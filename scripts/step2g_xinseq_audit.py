# -*- coding: utf-8 -*-
"""STEP 2G -- X_in_seq self-reconstruction audit. Read/compute only, no training, no file modification.

목적: L_mem_X의 target X(source_item_id)가 input sequence에 이미 존재하는 비율을 측정한다.
      X_in_seq=True인 경우 model이 memory recall 없이 history에서 X를 재현할 수 있으므로
      self-reconstruction 리스크 → full training 가능/불가 gate.

측정 항목:
  1. total samples / K≥2 samples / L_mem_X valid samples
  2. X_in_seq count & ratio (X ∈ user_train[uid][:-1])
  3. X == W_train count / X != W_train count
  4. source position of X (A: X==W_train / B: X in seq before W_train / C: X not found)
  5. L_mem_X valid 중 X_in_seq=True인 샘플: mean ΔzX, pos_frac ΔzX>0  (requires --checkpoint)
  6. L_mem_X valid 중 X_in_seq=False인 샘플: mean ΔzX, pos_frac ΔzX>0 (requires --checkpoint)

Usage (data-only, no model):
    python scripts/step2g_xinseq_audit.py --n_samples 2000 --seed 42

Usage (with ΔzX contrast):
    python scripts/step2g_xinseq_audit.py \\
        --n_samples 2000 --seed 42 --device cuda:0 \\
        --checkpoint checkpoints/Books/stage2_stage1enc_best.pt \\
        --stage1_ckpt checkpoints/Books/stage1_align_best.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dataloader import load_data
from src.training.train_hybrid import (
    TextEncoder,
    IntentProjector,
    _load_bank_full,
    _load_pseudo_queries,
    load_stage1_weights,
)


# ---------------------------------------------------------------------------
# Bank loader with item_ids (extended from step1_lmem_premeasure)

def _load_bank_with_item_ids(bank_jsonl: str) -> dict[str, list[dict]]:
    """Load bank → {uid_str: [{mid, text, item_ids}, ...]}."""
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
            mid = str(rec.get("memory_id", ""))
            text = rec.get("intent_description", "")
            item_ids = set(rec.get("evidence", {}).get("item_ids", []))
            if text:
                user_memories.setdefault(uid, []).append(
                    {"mid": mid, "text": text, "item_ids": item_ids}
                )
    return user_memories


# ---------------------------------------------------------------------------
# Sequence builder (same as step1)

def build_seq(user_train: dict[int, list[int]], uid_int: int, maxlen: int) -> np.ndarray:
    """Padded sequence [maxlen] from user_train[uid][:-1] (LOO: last item excluded)."""
    items = user_train.get(uid_int, [])
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
# ΔzX computation (requires model, projector, text_enc)

@torch.no_grad()
def compute_dzX_batch(
    model,
    text_enc: "TextEncoder",
    projector: "IntentProjector",
    samples: list[dict],
    device: str,
    batch_size: int = 32,
) -> list[float]:
    """Compute ΔzX = zX_pos - zX_neg for each sample. Returns list of floats (NaN on error)."""
    model.eval()
    results = []

    for b_start in range(0, len(samples), batch_size):
        batch = samples[b_start : b_start + batch_size]
        q_texts = [s["query"] for s in batch]
        pp_texts = [s["prov_pos_text"] for s in batch]
        in_texts = [s["intra_neg_text"] for s in batch]
        x_ids_list = [s["x"] for s in batch]
        seqs = np.stack([s["seq"] for s in batch])

        h_q = text_enc.encode(q_texts, is_query=True)
        h_pp = text_enc.encode(pp_texts, is_query=False)
        h_in = text_enc.encode(in_texts, is_query=False)

        pfx_pp = projector(h_q, h_pp)  # [B, 1, d_sasrec]
        pfx_in = projector(h_q, h_in)  # [B, 1, d_sasrec]

        log_pp, _ = model.log2feats(seqs, prefix_embeds=pfx_pp)
        log_in, _ = model.log2feats(seqs, prefix_embeds=pfx_in)

        h_pp_last = log_pp[:, -1, :]
        h_in_last = log_in[:, -1, :]

        x_t = torch.LongTensor(x_ids_list).to(device)
        emb_x = model.item_emb(x_t)

        zX_pos = (h_pp_last * emb_x).sum(-1)
        zX_neg = (h_in_last * emb_x).sum(-1)
        dz = (zX_pos - zX_neg).cpu().numpy()
        results.extend(dz.tolist())

    return results


# ---------------------------------------------------------------------------
# Main audit

def run_audit(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Paths
    bank_jsonl = args.bank_jsonl or "data/memory/Books/f3_bank.jsonl"
    pairs_jsonl = args.pairs_jsonl or f"data/processed/{args.category}/align_pairs.jsonl"

    print(f"\n[step2g] === X_in_seq Self-Reconstruction Audit ===")
    print(f"[step2g] bank={bank_jsonl}")
    print(f"[step2g] pairs={pairs_jsonl}")
    print(f"[step2g] n_samples={args.n_samples}  seed={args.seed}  device={args.device}")

    # Load bank with item_ids
    print("[step2g] Loading bank ...")
    user_memories = _load_bank_with_item_ids(bank_jsonl)
    mid_to_uid = {m["mid"]: uid for uid, mems in user_memories.items() for m in mems}
    print(f"[step2g] {len(user_memories)} users, {len(mid_to_uid)} mids")

    # Load pseudo-queries
    print("[step2g] Loading pseudo-queries ...")
    user_queries, user_source_ids, user_pos_mids = _load_pseudo_queries(
        pairs_jsonl, mid_to_uid=mid_to_uid
    )
    print(f"[step2g] {len(user_queries)} users with queries")

    # Load training data
    cat = args.category
    data_dir = args.data_dir or "data/processed"
    print(f"[step2g] Loading dataset {cat} ...")
    dataset = load_data(cat, data_dir)
    user_train = dataset["user_train"]   # {uid_int: [item_id, ...]}

    # Resolve maxlen from checkpoint args (must match pos_emb size) before building seqs
    maxlen = args.maxlen
    if args.checkpoint:
        try:
            _ck_tmp = torch.load(args.checkpoint, map_location="cpu")
            _saved_tmp = _ck_tmp.get("args", {})
            if isinstance(_saved_tmp, dict):
                maxlen = _saved_tmp.get("maxlen", args.maxlen)
            else:
                maxlen = getattr(_saved_tmp, "maxlen", args.maxlen)
            print(f"[step2g] Using maxlen={maxlen} from checkpoint args")
        except Exception:
            pass

    # Build sample pool: one pair per (uid, query_idx)
    pool: list[dict] = []
    uids_with_data = sorted(
        set(user_queries.keys()) & set(str(k) for k in user_train.keys())
    )
    rng.shuffle(uids_with_data)

    for uid_s in uids_with_data:
        if len(pool) >= args.n_samples * 5:  # oversample, then prune
            break
        uid_int = int(uid_s)
        mems = user_memories.get(uid_s, [])
        queries = user_queries.get(uid_s, [])
        src_ids = user_source_ids.get(uid_s, [])
        pos_mids = user_pos_mids.get(uid_s, [])

        items = user_train.get(uid_int, [])
        if len(items) < 2:
            continue
        w_train = items[-1]
        seq_items_set = set(items[:-1])  # input sequence items

        k_personal = len(mems)

        for q_idx, (q, x, pos_mid) in enumerate(zip(queries, src_ids, pos_mids)):
            if not q or not x or not pos_mid:
                continue
            prov_pos_m = next((m for m in mems if m["mid"] == pos_mid), None)
            neg_candidates = [m for m in mems if m["mid"] != pos_mid]
            intra_neg_m = (
                neg_candidates[rng.randint(0, len(neg_candidates) - 1)]
                if neg_candidates else None
            )
            valid_lmem = k_personal >= 2 and prov_pos_m is not None and intra_neg_m is not None

            x_in_seq = (x in seq_items_set)

            # Source position classification
            if x == w_train:
                pos_class = "A_eq_wtrain"
            elif x in seq_items_set:
                pos_class = "B_in_seq_before_wtrain"
            else:
                pos_class = "C_not_found"

            pool.append({
                "uid": uid_s,
                "query": q,
                "x": x,
                "w_train": w_train,
                "pos_mid": pos_mid,
                "prov_pos_text": prov_pos_m["text"] if prov_pos_m else None,
                "prov_pos_item_ids": prov_pos_m["item_ids"] if prov_pos_m else None,
                "intra_neg_text": intra_neg_m["text"] if intra_neg_m else None,
                "k_personal": k_personal,
                "valid_lmem": valid_lmem,
                "x_in_seq": x_in_seq,
                "pos_class": pos_class,
                "seq": build_seq(user_train, uid_int, maxlen),
            })

    rng.shuffle(pool)
    sample_n = min(args.n_samples, len(pool))
    pool = pool[:sample_n]
    print(f"[step2g] Collected {len(pool)} total samples (target {args.n_samples})")

    # ── Structural statistics (no model needed) ─────────────────────────────
    total_n = len(pool)
    kge2_n = sum(1 for s in pool if s["k_personal"] >= 2)
    valid_n = sum(1 for s in pool if s["valid_lmem"])

    valid_pool = [s for s in pool if s["valid_lmem"]]
    xinseq_n = sum(1 for s in valid_pool if s["x_in_seq"])
    xinseq_ratio = xinseq_n / valid_n if valid_n > 0 else float("nan")

    x_eq_w_n = sum(1 for s in valid_pool if s["x"] == s["w_train"])
    x_ne_w_n = valid_n - x_eq_w_n

    pos_class_counts = defaultdict(int)
    for s in valid_pool:
        pos_class_counts[s["pos_class"]] += 1

    print(f"\n{'='*60}")
    print(f"  STEP 2G — X_in_seq Audit Results")
    print(f"{'='*60}")
    print(f"\n[1] Sample Counts (seed={args.seed})")
    print(f"  total_samples      : {total_n}")
    print(f"  K>=2 samples       : {kge2_n}  ({100*kge2_n/total_n:.1f}%)")
    print(f"  L_mem_X valid      : {valid_n}  ({100*valid_n/total_n:.1f}%)")

    print(f"\n[2] X_in_seq (self-reconstruction risk)")
    print(f"  X_in_seq=True      : {xinseq_n}  ({100*xinseq_ratio:.1f}%)  ← RISK")
    print(f"  X_in_seq=False     : {valid_n - xinseq_n}  ({100*(1-xinseq_ratio):.1f}%)")

    print(f"\n[3] X == W_train")
    print(f"  X == W_train       : {x_eq_w_n}  ({100*x_eq_w_n/valid_n:.1f}%)")
    print(f"  X != W_train       : {x_ne_w_n}  ({100*x_ne_w_n/valid_n:.1f}%)")

    print(f"\n[4] Source Position of X in user_train (L_mem_X valid only)")
    for cls in ["A_eq_wtrain", "B_in_seq_before_wtrain", "C_not_found"]:
        cnt = pos_class_counts[cls]
        print(f"  {cls:<30}: {cnt:5d}  ({100*cnt/valid_n:.1f}%)")

    # ── ΔzX statistics (requires checkpoint) ────────────────────────────────
    if not args.checkpoint:
        print(f"\n[5-6] ΔzX stats: SKIPPED (--checkpoint not provided)")
        print(f"\n{'='*60}")
        _print_verdict(xinseq_ratio, args.xinseq_threshold, dzx_stats_available=False)
        return

    print(f"\n[step2g] Loading SASRec from {args.checkpoint} ...")
    from src.models.sasrec import SASRec
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    saved = ckpt.get("args", {})
    if isinstance(saved, dict):
        from types import SimpleNamespace
        saved = SimpleNamespace(**saved)
    sas_args = SimpleNamespace(
        maxlen=getattr(saved, "maxlen", maxlen),
        hidden_units=getattr(saved, "hidden_units", 128),
        num_blocks=getattr(saved, "num_blocks", 2),
        num_heads=getattr(saved, "num_heads", 1),
        dropout_rate=getattr(saved, "dropout_rate", 0.5),
        norm_first=getattr(saved, "norm_first", False),
        device=args.device,
    )
    usernum = ckpt.get("usernum", getattr(saved, "usernum", 100000))
    itemnum = ckpt.get("itemnum", getattr(saved, "itemnum", 100000))
    model = SASRec(usernum, itemnum, sas_args).to(args.device)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"'model_state_dict' not found in {args.checkpoint}. "
            "Pass the SASRec pretrain checkpoint (e.g. checkpoints/Books/sasrec_pretrain.pt) "
            "via --checkpoint, and stage2 enc+proj checkpoint via --proj_ckpt."
        )
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f"[step2g] SASRec frozen  d={sas_args.hidden_units}  items={itemnum}")

    print(f"[step2g] Loading text encoder ...")
    text_enc = TextEncoder(device=args.device)
    if args.stage1_ckpt:
        load_stage1_weights(args.stage1_ckpt, text_enc)
        print(f"[step2g] Stage1 weights loaded from {args.stage1_ckpt}")
    else:
        print(f"[step2g] stage1_ckpt not set — using raw BGE")
    text_enc.eval()

    projector = IntentProjector(text_enc.dim, sas_args.hidden_units).to(args.device)
    if args.proj_ckpt:
        pc = torch.load(args.proj_ckpt, map_location=args.device)
        projector.load_state_dict(pc.get("projector_state", pc))
        print(f"[step2g] projector loaded from {args.proj_ckpt}")
    projector.eval()

    # Compute ΔzX for valid samples with prov_pos and intra_neg available
    computable = [s for s in valid_pool if s["prov_pos_text"] and s["intra_neg_text"]]
    print(f"[step2g] Computing ΔzX for {len(computable)} valid samples ...")

    with torch.no_grad():
        dzx_vals = compute_dzX_batch(
            model, text_enc, projector, computable, args.device, batch_size=32
        )

    # Split by x_in_seq
    in_seq_dz = [dz for s, dz in zip(computable, dzx_vals) if s["x_in_seq"]]
    not_in_seq_dz = [dz for s, dz in zip(computable, dzx_vals) if not s["x_in_seq"]]

    def _stats(vals):
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.mean(np.array(vals) > 0))

    mean_in, pf_in = _stats(in_seq_dz)
    mean_out, pf_out = _stats(not_in_seq_dz)

    print(f"\n[5] X_in_seq=True  (n={len(in_seq_dz)}): mean ΔzX={mean_in:.4f}  pos_frac={pf_in:.3f}")
    print(f"[6] X_in_seq=False (n={len(not_in_seq_dz)}): mean ΔzX={mean_out:.4f}  pos_frac={pf_out:.3f}")

    print(f"\n{'='*60}")
    _print_verdict(xinseq_ratio, args.xinseq_threshold, dzx_stats_available=True,
                   mean_in=mean_in, mean_out=mean_out, pf_in=pf_in, pf_out=pf_out)


def _print_verdict(
    xinseq_ratio: float,
    threshold: float,
    dzx_stats_available: bool,
    mean_in: float = float("nan"),
    mean_out: float = float("nan"),
    pf_in: float = float("nan"),
    pf_out: float = float("nan"),
) -> None:
    print(f"  VERDICT")
    print(f"{'='*60}")
    if xinseq_ratio >= threshold:
        print(f"  X_in_seq ratio = {100*xinseq_ratio:.1f}% >= threshold {100*threshold:.0f}%")
        print(f"  => SELF-RECONSTRUCTION RISK HIGH")
        print(f"  => FULL TRAINING: FORBIDDEN")
        print(f"  => Resolution candidates:")
        print(f"     A) Restrict L_mem_X to X==W_train samples only (pos_class A)")
        print(f"     B) Rebuild prefix sequence to exclude X from input seq")
        print(f"     C) Demote L_mem_X to diagnostic/ablation; use Wtrain_aligned as main")
    else:
        print(f"  X_in_seq ratio = {100*xinseq_ratio:.1f}% < threshold {100*threshold:.0f}%")
        print(f"  => self-reconstruction risk LOW")
        if dzx_stats_available:
            print(f"  => mean ΔzX (in_seq={mean_in:.4f}, not_in_seq={mean_out:.4f})")
        print(f"  => X_in_seq gate: PASS")
        print(f"  => Pending: beta=0 compat check + val/test leak audit before full training")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--category", default="Books")
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--bank_jsonl", default=None)
    ap.add_argument("--pairs_jsonl", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--checkpoint", default=None, help="SASRec stage2 ckpt (for ΔzX)")
    ap.add_argument("--stage1_ckpt", default=None, help="Stage1 text encoder ckpt")
    ap.add_argument("--proj_ckpt", default=None, help="Projector ckpt (optional)")
    ap.add_argument("--xinseq_threshold", type=float, default=0.30,
                    help="X_in_seq ratio above which self-reconstruction risk is HIGH")
    args = ap.parse_args()

    from types import SimpleNamespace  # noqa: re-imported inside for clarity
    run_audit(args)


if __name__ == "__main__":
    main()
