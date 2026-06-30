"""STEP 1 — L_mem 재학습 전 사전측정 (read/compute only, no training, no file modification).

목적: X-target vs W_train-target score-space contrast 측정으로 L_mem target 확정을 위한 근거 수집.
구현 금지·학습 금지·파일수정 금지.

Usage:
    python scripts/step1_lmem_premeasure.py \
        --device cpu          # or cuda:0
        --n_samples 2000
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data
from src.training.train_hybrid import (
    TextEncoder,
    load_stage1_weights,
    _load_bank_full,
)


# ---------------------------------------------------------------------------
# Data helpers

def load_bank_with_evidence(bank_jsonl: str) -> dict[str, list[dict]]:
    """Load bank → {uid_str: [{mid, text, item_ids}, ...]}.

    Extends _load_bank_full with evidence.item_ids so we can resolve which
    memory cluster contains X or W_train.
    """
    user_memories: dict[str, list[dict]] = {}
    with open(bank_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))
            if not uid:
                continue
            if "intent_description" in rec:
                mid = str(rec.get("memory_id", ""))
                item_ids = set(rec.get("evidence", {}).get("item_ids", []))
                user_memories.setdefault(uid, []).append(
                    {"mid": mid, "text": rec["intent_description"], "item_ids": item_ids}
                )
    return user_memories


def load_align_pairs_with_uid(pairs_jsonl: str, mid_to_uid: dict[str, str]) -> list[dict]:
    """Load align_pairs.jsonl → list of {uid, query, positive_memory_id, source_item_id}.

    Resolves uid from positive_memory_id via mid_to_uid (align_pairs lack user_id).
    """
    pairs: list[dict] = []
    with open(pairs_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mid = str(rec.get("positive_memory_id", ""))
            if not mid:
                continue
            uid = mid_to_uid.get(mid, "")
            if not uid:
                continue
            x = rec.get("source_item_id")
            q = rec.get("query", "")
            if not q or x is None:
                continue
            pairs.append({
                "uid": uid,
                "query": q,
                "positive_memory_id": mid,
                "source_item_id": int(x),
            })
    return pairs


# ---------------------------------------------------------------------------
# Taxonomy classification

def classify_sample(
    x: int,
    w_train: int,
    uid: str,
    pos_mid: str,
    user_mems: list[dict],
) -> str:
    """Return A/B/C/D label for one sample.

    A: X == W_train
    B: X != W_train, but mid(X) == mid(W_train) (same cluster)
    C: X != W_train, W_train is in some memory cluster, but cluster differs from X's cluster
    D: W_train has no provenance in memory evidence
    """
    if x == w_train:
        return "A"

    # Find which cluster X belongs to (should be pos_mid cluster)
    mid_of_x = None
    mid_of_w = None
    for m in user_mems:
        if x in m["item_ids"]:
            mid_of_x = m["mid"]
        if w_train in m["item_ids"]:
            mid_of_w = m["mid"]

    if mid_of_w is None:
        return "D"
    if mid_of_x == mid_of_w:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Checkpoint loader

def load_ckpt(
    ckpt_path: str,
    device: str,
    is_stage1enc: bool,
) -> tuple[TextEncoder, IntentProjector]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    d_text = cfg.get("d_text", 768)
    d_sasrec = cfg.get("d_sasrec", 256)

    text_enc = TextEncoder(model_id="BAAI/bge-base-en-v1.5", device=device)
    if is_stage1enc and cfg.get("stage1_ckpt"):
        try:
            load_stage1_weights(cfg["stage1_ckpt"], text_enc)
        except Exception:
            pass
    enc_state = ckpt.get("text_encoder_state")
    if enc_state:
        text_enc.load_state_dict(enc_state)
    projector = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
    proj_state = ckpt.get("projector_state")
    if proj_state:
        projector.load_state_dict(proj_state)

    text_enc.eval()
    projector.eval()
    return text_enc, projector


# ---------------------------------------------------------------------------
# Sequence builder

def build_seq(user_train: dict[int, list[int]], uid_int: int, maxlen: int) -> np.ndarray:
    """Build padded sequence [maxlen] from user_train[uid][:-1] (LOO: last item excluded)."""
    items = user_train.get(uid_int, [])
    # seq = all items except last (W_train)
    seq_items = items[:-1]
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for i in reversed(seq_items):
        if idx < 0:
            break
        seq[idx] = i
        idx -= 1
    return seq


# ---------------------------------------------------------------------------
# Contrast computation for one checkpoint

@torch.no_grad()
def compute_contrasts(
    model: SASRec,
    text_enc: TextEncoder,
    projector: IntentProjector,
    samples: list[dict],  # [{uid, query, pos_mid, x, w_train, label, seq, m_pos, m_neg}, ...]
    device: str,
    batch_size: int = 32,
) -> list[dict]:
    """Compute ΔzX, ΔzW, cosX, cosW for each sample. Returns extended records."""
    model.eval()
    results = []

    for b_start in range(0, len(samples), batch_size):
        batch = samples[b_start : b_start + batch_size]

        # Encode queries and memories in one shot
        q_texts = [s["query"] for s in batch]
        m_pos_texts = [s["m_pos_text"] for s in batch]
        m_neg_texts = [s["m_neg_text"] for s in batch]

        h_q = text_enc.encode(q_texts, is_query=True)      # [B, d_text]
        h_mpos = text_enc.encode(m_pos_texts, is_query=False)  # [B, d_text]
        h_mneg = text_enc.encode(m_neg_texts, is_query=False)  # [B, d_text]

        # Router top-1: cosine(h_mems, h_q) per sample — done per sample since K varies
        # NOTE: _route_batch selects best_mem = mem_texts[sims.argmax()] — i.e., router top-1 by cosine.
        # This is the CURRENT training path's m+ (router top-1), which may differ from provenance m+.
        router_top1_mids = []
        for j, s in enumerate(batch):
            uid_str = s["uid"]
            all_mems = s["all_mems"]  # [{mid, text, item_ids}]
            if len(all_mems) == 1:
                router_top1_mids.append(all_mems[0]["mid"])
            else:
                mem_texts = [m["text"] for m in all_mems]
                h_mems_j = text_enc.encode(mem_texts, is_query=False)
                sims = (h_mems_j @ h_q[j : j + 1].T).squeeze(-1)
                router_top1_mids.append(all_mems[sims.argmax().item()]["mid"])

        pfx_pos = projector(h_q, h_mpos)  # [B, 1, d_sasrec]
        pfx_neg = projector(h_q, h_mneg)  # [B, 1, d_sasrec]

        # Stack seqs for batch log2feats calls
        seqs = np.stack([s["seq"] for s in batch])  # [B, maxlen]

        log_feats_pos, _ = model.log2feats(seqs, prefix_embeds=pfx_pos)
        log_feats_neg, _ = model.log2feats(seqs, prefix_embeds=pfx_neg)

        h_pos = log_feats_pos[:, -1, :]  # [B, d_sasrec]
        h_neg = log_feats_neg[:, -1, :]  # [B, d_sasrec]
        delta_h_x = h_pos - h_neg  # [B, d_sasrec] (for X target direction)
        delta_h_w = h_pos - h_neg  # same delta_h, different target item

        # Item embeddings for X and W_train
        x_ids = torch.tensor([s["x"] for s in batch], dtype=torch.long, device=device)
        w_ids = torch.tensor([s["w_train"] for s in batch], dtype=torch.long, device=device)
        emb_x = model.item_emb(x_ids)      # [B, d_sasrec]
        emb_w = model.item_emb(w_ids)      # [B, d_sasrec]

        # Score-space contrasts
        zX_pos = (emb_x * h_pos).sum(dim=-1)  # [B]
        zX_neg = (emb_x * h_neg).sum(dim=-1)
        zW_pos = (emb_w * h_pos).sum(dim=-1)
        zW_neg = (emb_w * h_neg).sum(dim=-1)

        dz_x = (zX_pos - zX_neg).cpu().numpy()   # [B]
        dz_w = (zW_pos - zW_neg).cpu().numpy()   # [B]

        # Cosine of delta-h with target item direction
        cos_x = F.cosine_similarity(delta_h_x, emb_x, dim=-1).cpu().numpy()  # [B]
        cos_w = F.cosine_similarity(delta_h_w, emb_w, dim=-1).cpu().numpy()  # [B]

        for j, s in enumerate(batch):
            results.append({
                **{k: v for k, v in s.items() if k not in ("seq", "all_mems")},
                "dz_x": float(dz_x[j]),
                "dz_w": float(dz_w[j]),
                "cos_x": float(cos_x[j]),
                "cos_w": float(cos_w[j]),
                "router_top1_mid": router_top1_mids[j],
                "router_matches_provenance": router_top1_mids[j] == s["pos_mid"],
            })

    return results


# ---------------------------------------------------------------------------
# Table printer

def print_table(title: str, columns: list[str], rows: list[tuple]) -> None:
    widths = [max(len(c), max((len(str(r[i])) for r in rows), default=0)) for i, c in enumerate(columns)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    sep = "  ".join("-" * w for w in widths)
    print(f"\n{title}")
    print(fmt.format(*columns))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


# ---------------------------------------------------------------------------
# Stats helper

def stats(vals: list[float]) -> tuple[float, float, float]:
    """Return (mean, SE, pos_frac) for a list of floats."""
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    se = (var / n) ** 0.5
    pf = sum(1 for v in vals if v > 0) / n
    return m, se, pf


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Books")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--sasrec_ckpt", default="checkpoints/Books/sasrec_pretrain.pt")
    parser.add_argument("--ckpt_2a", default="checkpoints/Books/stage2_stage1enc_best.pt")
    parser.add_argument("--ckpt_2b", default="checkpoints/Books/stage2_rawbge_best.pt")
    parser.add_argument("--bank_jsonl", default="data/memory/Books/f3_bank.jsonl")
    parser.add_argument("--pairs_jsonl", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    pairs_jsonl = args.pairs_jsonl or f"data/processed/{args.category}/align_pairs.jsonl"
    rng = random.Random(args.seed)
    device = args.device

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("[data] Loading dataset...")
    dataset = load_data(args.category, args.data_dir)
    user_train = dataset["user_train"]
    itemnum = dataset["itemnum"]

    ckpt_sas = torch.load(args.sasrec_ckpt, map_location="cpu")
    saved = ckpt_sas["args"]
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)
    model_args = SimpleNamespace(
        maxlen=saved.maxlen, hidden_units=saved.hidden_units,
        num_blocks=saved.num_blocks, num_heads=saved.num_heads,
        dropout_rate=saved.dropout_rate, norm_first=saved.norm_first,
        device=device,
    )
    model = SASRec(dataset["usernum"], dataset["itemnum"], model_args).to(device)
    model.load_state_dict(ckpt_sas["model_state_dict"])
    model.eval()
    print(f"  SASRec: users={dataset['usernum']} items={itemnum} maxlen={saved.maxlen}")

    # ── Load bank with evidence ───────────────────────────────────────────────
    print("[data] Loading bank with evidence item_ids...")
    user_mems_full = load_bank_with_evidence(args.bank_jsonl)  # {uid: [{mid, text, item_ids}]}
    mid_to_uid = {m["mid"]: uid for uid, mems in user_mems_full.items() for m in mems}
    print(f"  bank: {len(user_mems_full)} users, {len(mid_to_uid)} memories")

    # ── Load align_pairs ─────────────────────────────────────────────────────
    print("[data] Loading align_pairs...")
    all_pairs = load_align_pairs_with_uid(pairs_jsonl, mid_to_uid)
    print(f"  {len(all_pairs)} pairs with resolved uid")

    # ── Build K>=2 eligible set ───────────────────────────────────────────────
    # Only keep pairs where user has K>=2 memories
    eligible = []
    for p in all_pairs:
        uid = p["uid"]
        mems = user_mems_full.get(uid, [])
        if len(mems) < 2:
            continue
        uid_int = int(uid)
        if uid_int not in user_train or len(user_train[uid_int]) < 1:
            continue
        eligible.append(p)

    print(f"  {len(eligible)} pairs with K>=2 memories (eligible for contrast measurement)")
    print(f"  total pairs (all K): {len(all_pairs)}")

    # Sample N>=2000
    n_sample = min(args.n_samples, len(eligible))
    sampled_pairs = rng.sample(eligible, n_sample)
    print(f"  sampling {n_sample} pairs (seed={args.seed})")

    # ── Build sample records ──────────────────────────────────────────────────
    # For each pair:
    #   - w_train = user_train[uid][-1]
    #   - seq = user_train[uid][:-1] padded
    #   - m+ = provenance memory (positive_memory_id)
    #   - m- = different cluster_id memory from same user
    #   - label = A/B/C/D

    # Verify: W_train should NOT appear in seq
    w_in_seq_count = 0
    samples_full = []
    for p in sampled_pairs:
        uid_int = int(p["uid"])
        train_items = user_train[uid_int]
        w_train = train_items[-1]
        seq = build_seq(user_train, uid_int, model_args.maxlen)

        # Verify W_train not in seq
        if w_train in seq:
            w_in_seq_count += 1

        mems = user_mems_full[p["uid"]]
        pos_mid = p["positive_memory_id"]

        # m+ must be the provenance memory (positive_memory_id), NOT router top-1
        m_pos = next((m for m in mems if m["mid"] == pos_mid), None)
        if m_pos is None:
            continue  # provenance mid not in bank (should be rare)

        # m- = different cluster memory from same user
        # Exclude K=1 (already filtered), find one with different mid from m+
        m_neg_candidates = [m for m in mems if m["mid"] != pos_mid]
        if not m_neg_candidates:
            continue  # only one distinct cluster despite K>=2
        m_neg = rng.choice(m_neg_candidates)

        label = classify_sample(p["source_item_id"], w_train, p["uid"], pos_mid, mems)

        samples_full.append({
            "uid": p["uid"],
            "query": p["query"],
            "pos_mid": pos_mid,
            "x": p["source_item_id"],
            "w_train": w_train,
            "label": label,
            "seq": seq,
            "m_pos_text": m_pos["text"],
            "m_neg_text": m_neg["text"],
            "m_neg_mid": m_neg["mid"],
            "all_mems": mems,
        })

    print(f"\n  Samples after provenance m+ resolution: {len(samples_full)}")
    print(f"  W_train in seq check: {w_in_seq_count} / {n_sample}  (should be 0)")

    # ── STEP 1A — Taxonomy ────────────────────────────────────────────────────
    label_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for s in samples_full:
        label_counts[s["label"]] += 1

    n_k2 = len(samples_full)
    n_total_pairs = len(all_pairs)
    n_k2_eligible = len(eligible)
    a_b = label_counts["A"] + label_counts["B"]
    c_d = label_counts["C"] + label_counts["D"]

    def pct(n, d):
        return f"{100*n/d:.1f}%" if d > 0 else "n/a"

    print_table(
        "Table 1: Sample Taxonomy",
        ["group", "n", "ratio"],
        [
            ("total pairs (all K)", n_total_pairs, pct(n_total_pairs, n_total_pairs)),
            ("K>=2 eligible", n_k2_eligible, pct(n_k2_eligible, n_total_pairs)),
            ("sampled K>=2", n_k2, pct(n_k2, n_k2_eligible)),
            ("A: X == W_train", label_counts["A"], pct(label_counts["A"], n_k2)),
            ("B: X!=W, same cluster", label_counts["B"], pct(label_counts["B"], n_k2)),
            ("C: X!=W, diff cluster", label_counts["C"], pct(label_counts["C"], n_k2)),
            ("D: W_train no provenance", label_counts["D"], pct(label_counts["D"], n_k2)),
            ("A+B (W_train aligned)", a_b, pct(a_b, n_k2)),
            ("C+D (W_train misaligned)", c_d, pct(c_d, n_k2)),
        ],
    )

    # ── STEP 1B/1C/1D — contrast measurement per checkpoint ──────────────────
    ckpt_configs = [
        ("2a", args.ckpt_2a, True),
        ("2b", args.ckpt_2b, False),
    ]

    all_results: dict[str, list[dict]] = {}

    for ckpt_tag, ckpt_path, is_stage1enc in ckpt_configs:
        if not Path(ckpt_path).exists():
            print(f"\n[skip] ckpt {ckpt_tag}: {ckpt_path} not found")
            continue

        print(f"\n[ckpt {ckpt_tag}] Loading {ckpt_path} ...")
        text_enc, projector = load_ckpt(ckpt_path, device, is_stage1enc)

        print(f"  Computing contrasts on {len(samples_full)} samples ...")
        res = compute_contrasts(model, text_enc, projector, samples_full, device, args.batch_size)
        all_results[ckpt_tag] = res
        print(f"  Done.")

    # ── Print tables ──────────────────────────────────────────────────────────
    group_labels = ["K>=2 all", "A", "B", "C", "D", "A+B", "C+D"]

    def filter_group(res: list[dict], group: str) -> list[dict]:
        if group == "K>=2 all":
            return res
        if group == "A+B":
            return [r for r in res if r["label"] in ("A", "B")]
        if group == "C+D":
            return [r for r in res if r["label"] in ("C", "D")]
        return [r for r in res if r["label"] == group]

    # Table 2: X-target contrast
    t2_rows = []
    for ckpt_tag, res in all_results.items():
        for grp in group_labels:
            sub = filter_group(res, grp)
            if not sub:
                t2_rows.append((ckpt_tag, grp, 0, "n/a", "n/a", "n/a", "n/a"))
                continue
            m, se, pf_dz = stats([r["dz_x"] for r in sub])
            _, _, pf_cos = stats([r["cos_x"] for r in sub])
            t2_rows.append((
                ckpt_tag, grp, len(sub),
                f"{m:+.4f}", f"{se:.4f}", f"{pf_dz:.3f}", f"{pf_cos:.3f}",
            ))

    print_table(
        "\nTable 2: X-target contrast  (m+ = provenance positive_memory_id)",
        ["ckpt", "group", "n", "mean_ΔzX", "SE", "pos_frac(ΔzX>0)", "cosX_pos_frac"],
        t2_rows,
    )

    # Table 3: W_train-target contrast
    t3_rows = []
    for ckpt_tag, res in all_results.items():
        for grp in group_labels:
            sub = filter_group(res, grp)
            if not sub:
                t3_rows.append((ckpt_tag, grp, 0, "n/a", "n/a", "n/a", "n/a"))
                continue
            m, se, pf_dz = stats([r["dz_w"] for r in sub])
            _, _, pf_cos = stats([r["cos_w"] for r in sub])
            t3_rows.append((
                ckpt_tag, grp, len(sub),
                f"{m:+.4f}", f"{se:.4f}", f"{pf_dz:.3f}", f"{pf_cos:.3f}",
            ))

    print_table(
        "\nTable 3: W_train-target contrast  (m+ = provenance positive_memory_id)",
        ["ckpt", "group", "n", "mean_ΔzW", "SE", "pos_frac(ΔzW>0)", "cosW_pos_frac"],
        t3_rows,
    )

    # Table 4: router top-1 vs provenance match
    t4_rows = []
    for ckpt_tag, res in all_results.items():
        for grp in group_labels:
            sub = filter_group(res, grp)
            if not sub:
                t4_rows.append((ckpt_tag, grp, 0, "n/a", "n/a", "n/a"))
                continue
            router_match = [r for r in sub if r["router_matches_provenance"]]
            router_miss = [r for r in sub if not r["router_matches_provenance"]]
            match_rate = f"{len(router_match)/len(sub):.3f}" if sub else "n/a"

            # ΔzX pos_frac split by router match
            _, _, pf_x_match = stats([r["dz_x"] for r in router_match]) if router_match else (0, 0, float("nan"))
            _, _, pf_x_miss = stats([r["dz_x"] for r in router_miss]) if router_miss else (0, 0, float("nan"))
            _, _, pf_w_match = stats([r["dz_w"] for r in router_match]) if router_match else (0, 0, float("nan"))
            _, _, pf_w_miss = stats([r["dz_w"] for r in router_miss]) if router_miss else (0, 0, float("nan"))

            t4_rows.append((
                ckpt_tag, grp, len(sub),
                f"{match_rate} ({len(router_match)}/{len(sub)})",
                f"match={pf_x_match:.3f} miss={pf_x_miss:.3f}",
                f"match={pf_w_match:.3f} miss={pf_w_miss:.3f}",
            ))

    print_table(
        "\nTable 4: Router top-1 vs provenance m+ match",
        ["ckpt", "group", "n", "router==prov (n/N)", "ΔzX_pos_frac (match/miss)", "ΔzW_pos_frac (match/miss)"],
        t4_rows,
    )

    # ── Observation summary (no interpretation, observations only) ────────────
    if all_results:
        first_res = next(iter(all_results.values()))
        router_acc = sum(1 for r in first_res if r["router_matches_provenance"]) / len(first_res)
        dz_x_all_mean = sum(r["dz_x"] for r in first_res) / len(first_res)
        dz_w_all_mean = sum(r["dz_w"] for r in first_res) / len(first_res)
        ab_sub = [r for r in first_res if r["label"] in ("A", "B")]
        cd_sub = [r for r in first_res if r["label"] in ("C", "D")]
        dz_w_ab = sum(r["dz_w"] for r in ab_sub) / len(ab_sub) if ab_sub else float("nan")
        dz_w_cd = sum(r["dz_w"] for r in cd_sub) / len(cd_sub) if cd_sub else float("nan")

        print("\n" + "="*60)
        print("Observations (raw measurements, no conclusion drawn)")
        print("="*60)
        print(f"  Taxonomy: A={label_counts['A']} B={label_counts['B']} "
              f"C={label_counts['C']} D={label_counts['D']} "
              f"| A+B={a_b} ({pct(a_b,n_k2)}) C+D={c_d} ({pct(c_d,n_k2)})")
        print(f"  First ckpt ({list(all_results.keys())[0]}), K>=2 all (n={len(first_res)}):")
        print(f"    mean_ΔzX = {dz_x_all_mean:+.4f}")
        print(f"    mean_ΔzW = {dz_w_all_mean:+.4f}")
        print(f"    mean_ΔzW A+B = {dz_w_ab:+.4f}  "
              f"C+D = {dz_w_cd:+.4f}  "
              f"gap = {(dz_w_ab - dz_w_cd):+.4f}")
        print(f"    router top-1 == provenance m+ rate = {router_acc:.3f}")

        print("\n  Interpretation keys (from task spec):")
        print("  1. ΔzX >> ΔzW  →  L_mem_X more coherent than L_mem_Wtrain")
        print("  2. ΔzW[A+B] >> ΔzW[C+D]  →  L_mem_Wtrain_aligned feasible, Wtrain_all risky")
        print("  3. ΔzW[C+D] also large  →  user/sequence shortcut suspected")
        print("  4. Both ΔzX, ΔzW ≈0.5 pos_frac  →  no score-space signal at all")
        print("  5. router ≠ provenance high  →  L_mem with router top-1 as m+ is incorrect")


if __name__ == "__main__":
    main()
