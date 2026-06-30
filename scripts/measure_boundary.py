"""STEP 2, 3, 4: Boundary crossing / candidate augmentation / projector sensitivity.

Uses existing checkpoints only — no retraining.

STEP 2 — Cross@K / Drop@K / NetCross@K for K=10/20/50/100/200
  Cross@K = fraction(r_vanilla>K and r_steered<=K)  → vanilla miss, steered hit
  Drop@K  = fraction(r_vanilla<=K and r_steered>K)  → vanilla hit, steered miss
  NetCross@K = Cross - Drop

STEP 3 — fixed-budget candidate augmentation, M=100, b=10/20/50
  C_aug = vanilla_top_(M-b) ∪ steered_top_b
  Recall@M(C_aug) vs Recall@M(vanilla_top_M)

STEP 4 — projector memory sensitivity
  D_p   = ||g(h_q,h_m+) - g(h_q,h_m-)||_2  (projector output distance)
  Δz_y  = score_target(pfx_pos) - score_target(pfx_neg)  (rec score diff)
  Small D_p → projector ignores h_memory (uses h_query only) → B-death root cause
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data
from src.training.train_hybrid import TextEncoder, _load_bank_full, _load_pseudo_queries
from scripts.eval_steered import (
    make_steered_prefix_fn,
    _load_user_first_pos_mid,
)


# ---------------------------------------------------------------------------
# Checkpoint loading

def load_stage2(ckpt_path: str, device: str) -> tuple[TextEncoder, IntentProjector]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    d_text = cfg.get("d_text", 768)
    d_sasrec = cfg.get("d_sasrec", 256)

    text_enc = TextEncoder(model_id="BAAI/bge-base-en-v1.5", device=device)
    state = ckpt.get("text_encoder_state")
    if state:
        text_enc.load_state_dict(state)
    else:
        print(f"  [warn] no text_encoder_state in {ckpt_path}")

    proj = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
    pstate = ckpt.get("projector_state")
    if pstate:
        proj.load_state_dict(pstate)
    else:
        print(f"  [warn] no projector_state in {ckpt_path}")

    text_enc.eval()
    proj.eval()
    return text_enc, proj


# ---------------------------------------------------------------------------
# Core eval loop: returns {user_id: rank} and optionally top-K item sets

@torch.no_grad()
def run_full_eval(
    model,
    dataset: dict,
    eval_args,
    prefix_fn,
    return_topk: int = 0,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Returns (user_ranks, user_topk_items).

    user_topk_items is populated only when return_topk > 0.
    Item IDs are 1-indexed to match dataset conventions.
    """
    dev = eval_args.device
    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test  = dataset["user_test"]
    itemnum    = dataset["itemnum"]

    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)  # [I, d]

    all_users = [u for u in range(1, dataset["usernum"] + 1)
                 if user_train.get(u) and user_test.get(u)]

    user_ranks: dict[int, int] = {}
    user_topk: dict[int, list[int]] = {}

    for u in all_users:
        target = user_test[u][0]

        seq = np.zeros([eval_args.maxlen], dtype=np.int32)
        idx = eval_args.maxlen - 1
        if user_valid.get(u):
            for vi in reversed(user_valid[u]):
                seq[idx] = vi
                idx -= 1
                if idx == -1:
                    break
        for i in reversed(user_train[u]):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1

        seq_np = seq[np.newaxis, :]
        pfx = prefix_fn(u, seq_np) if prefix_fn is not None else None
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat = log_feats[0, -1, :]
        scores = all_item_embs.matmul(final_feat)

        seen = set(user_train[u]) | set(user_valid.get(u, []))
        seen.discard(0)
        seen_idx = torch.tensor([i - 1 for i in seen if 1 <= i <= itemnum],
                                dtype=torch.long, device=dev)
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        rank = int((scores > scores[target - 1]).sum().item())
        user_ranks[u] = rank

        if return_topk > 0:
            topk_idx = scores.topk(min(return_topk, itemnum)).indices.tolist()
            user_topk[u] = [i + 1 for i in topk_idx]  # 1-indexed item IDs

    return user_ranks, user_topk


# ---------------------------------------------------------------------------
# STEP 2: boundary crossing analysis

def step2_boundary_crossing(
    vanilla_ranks: dict[int, int],
    steered_ranks: dict[int, int],
    label: str,
    ks: list[int] | None = None,
) -> dict:
    if ks is None:
        ks = [10, 20, 50, 100, 200]

    common = sorted(set(vanilla_ranks) & set(steered_ranks))
    n = len(common)
    if n == 0:
        print(f"  [STEP2:{label}] n=0, skip")
        return {}

    print(f"\n  [STEP2:{label}] n={n}")
    print(f"  {'K':>5}  {'Cross':>8}  {'Drop':>8}  {'NetCross':>9}  "
          f"{'Cross_n':>8}  {'Drop_n':>7}")

    results = {}
    for k in ks:
        cross_n = sum(1 for u in common if vanilla_ranks[u] >= k and steered_ranks[u] < k)
        drop_n  = sum(1 for u in common if vanilla_ranks[u] < k  and steered_ranks[u] >= k)
        net_n   = cross_n - drop_n
        cross_r = cross_n / n
        drop_r  = drop_n / n
        net_r   = net_n / n
        print(f"  @{k:4d}  {cross_r:8.4f}  {drop_r:8.4f}  {net_r:+9.4f}  "
              f"{cross_n:8d}  {drop_n:7d}")
        results[k] = {"cross": cross_r, "drop": drop_r, "net": net_r,
                      "cross_n": cross_n, "drop_n": drop_n, "n": n}

    return results


# ---------------------------------------------------------------------------
# STEP 3: fixed-budget candidate augmentation

def step3_candidate_augmentation(
    vanilla_ranks: dict[int, int],
    vanilla_topM: dict[int, list[int]],
    steered_topb: dict[int, list[int]],
    dataset: dict,
    label: str,
    M: int = 100,
    bs: list[int] | None = None,
) -> dict:
    if bs is None:
        bs = [10, 20, 50]

    user_test = dataset["user_test"]
    common = sorted(set(vanilla_topM) & set(steered_topb) & set(vanilla_ranks))
    n = len(common)
    if n == 0:
        print(f"  [STEP3:{label}] n=0, skip")
        return {}

    # vanilla baseline: Recall@M(vanilla top-M)
    vanilla_recall_M = sum(
        1 for u in common
        if user_test.get(u) and user_test[u][0] in set(vanilla_topM[u][:M])
    ) / n

    print(f"\n  [STEP3:{label}] n={n}  M={M}")
    print(f"  vanilla Recall@{M} = {vanilla_recall_M:.4f}")
    print(f"  {'b':>5}  {'R@M(aug)':>10}  {'gain':>8}  {'new_slots_avg':>14}")

    results = {"vanilla_recall_M": vanilla_recall_M}
    for b in bs:
        if b >= M:
            continue
        budget_vanilla = M - b

        aug_hits = 0
        total_new_slots = 0
        for u in common:
            target = user_test[u][0] if user_test.get(u) else None
            if target is None:
                continue
            v_set = set(vanilla_topM[u][:budget_vanilla])
            s_top_b = steered_topb[u][:b]
            new_from_steered = [i for i in s_top_b if i not in v_set]
            total_new_slots += len(new_from_steered)
            c_aug = v_set | set(new_from_steered)
            if target in c_aug:
                aug_hits += 1

        aug_recall = aug_hits / n
        gain = aug_recall - vanilla_recall_M
        avg_new = total_new_slots / n
        sign = "+" if gain >= 0 else ""
        print(f"  b={b:4d}  {aug_recall:10.4f}  {sign}{gain:.4f}  {avg_new:14.1f}")
        results[b] = {"aug_recall": aug_recall, "gain": gain, "avg_new_slots": avg_new}

    return results


# ---------------------------------------------------------------------------
# STEP 4: projector memory sensitivity

@torch.no_grad()
def step4_projector_sensitivity(
    model,
    dataset: dict,
    eval_args,
    text_enc: TextEncoder,
    proj: IntentProjector,
    user_queries: dict[str, list[str]],
    user_memories: dict[str, list[dict]],
    user_first_mid: dict[str, str],
    label: str,
    n_sample: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute D_p and Δz_y to diagnose whether projector uses h_memory.

    D_p   = ||g(h_q, h_m+) - g(h_q, h_m-)||_2  (L2 distance in prefix space)
    Δz_y  = score(target | h_m+) - score(target | h_m-)

    Negative memory m- strategy:
      K≥2: pick a different memory from the same user (index 1)
      K=1: use a random other user's first memory
    """
    rng = random.Random(seed)
    dev = eval_args.device
    user_test = dataset["user_test"]
    itemnum = dataset["itemnum"]

    # Candidates: users with query, memory, first_mid, and test target
    candidates = [
        u for u in range(1, dataset["usernum"] + 1)
        if user_test.get(u)
        and user_queries.get(str(u))
        and user_memories.get(str(u))
        and user_first_mid.get(str(u))
    ]

    if len(candidates) > n_sample:
        rng_samp = random.Random(seed)
        candidates = rng_samp.sample(candidates, n_sample)

    # Build fallback: pool of (uid_str, mem_text) for K=1 negatives
    all_uid_strs = list(user_memories.keys())

    all_items_t = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items_t)  # [I, d]

    dp_values: list[float] = []
    dz_values: list[float] = []
    n_same_user_neg: int = 0

    for u in candidates:
        uid_s = str(u)
        query_text = user_queries[uid_s][0]
        mems = user_memories[uid_s]

        pos_mid = user_first_mid[uid_s]
        pos_mem = next((m for m in mems if m["mid"] == pos_mid), None)
        if pos_mem is None:
            continue

        # Negative memory selection
        if len(mems) >= 2:
            neg_mem = next((m for m in mems if m["mid"] != pos_mid), None)
            if neg_mem is None:
                continue
            n_same_user_neg += 1
        else:
            # Random other user's first memory
            other_uid = uid_s
            tries = 0
            while other_uid == uid_s and tries < 20:
                other_uid = rng.choice(all_uid_strs)
                tries += 1
            if other_uid == uid_s:
                continue
            other_mems = user_memories[other_uid]
            neg_mem = other_mems[0]

        # Encode
        h_q   = text_enc.encode([query_text], is_query=True)           # [1, d_text]
        h_mpos = text_enc.encode([pos_mem["text"]], is_query=False)    # [1, d_text]
        h_mneg = text_enc.encode([neg_mem["text"]], is_query=False)    # [1, d_text]

        pfx_pos = proj(h_q, h_mpos)  # [1, 1, d_sasrec]
        pfx_neg = proj(h_q, h_mneg)

        # D_p: L2 distance between prefix embeddings
        dp = (pfx_pos - pfx_neg).norm(dim=-1).mean().item()
        dp_values.append(dp)

        # Δz_y: recommendation score difference for the test target
        target = user_test[u][0]
        if not (1 <= target <= itemnum):
            continue

        seq = np.zeros([eval_args.maxlen], dtype=np.int32)
        idx = eval_args.maxlen - 1
        for i in reversed(dataset["user_train"].get(u, [])):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1
        seq_np = seq[np.newaxis, :]

        log_pos, _ = model.log2feats(seq_np, prefix_embeds=pfx_pos.to(dev))
        log_neg, _ = model.log2feats(seq_np, prefix_embeds=pfx_neg.to(dev))

        feat_pos = log_pos[0, -1, :]
        feat_neg = log_neg[0, -1, :]

        item_emb_y = all_item_embs[target - 1]  # [d]
        score_pos = item_emb_y.dot(feat_pos).item()
        score_neg = item_emb_y.dot(feat_neg).item()
        dz_values.append(score_pos - score_neg)

    n_valid = len(dp_values)
    if n_valid == 0:
        print(f"  [STEP4:{label}] n=0, skip")
        return {}

    dp_mean = sum(dp_values) / n_valid
    dp_std  = (sum((x - dp_mean)**2 for x in dp_values) / n_valid) ** 0.5
    dp_p10  = sorted(dp_values)[int(0.1 * n_valid)]
    dp_p50  = sorted(dp_values)[int(0.5 * n_valid)]
    dp_p90  = sorted(dp_values)[int(0.9 * n_valid)]

    dz_mean = sum(dz_values) / len(dz_values) if dz_values else float("nan")
    dz_std  = (sum((x - dz_mean)**2 for x in dz_values) / len(dz_values)) ** 0.5 if dz_values else float("nan")
    dz_pos  = sum(1 for x in dz_values if x > 0)
    dz_neg  = sum(1 for x in dz_values if x < 0)
    dz_n    = len(dz_values)

    print(f"\n  [STEP4:{label}] n_valid={n_valid}  same_user_neg={n_same_user_neg}")
    print(f"  D_p (||pfx_pos - pfx_neg||):  mean={dp_mean:.4f}  std={dp_std:.4f}  "
          f"p10={dp_p10:.4f}  p50={dp_p50:.4f}  p90={dp_p90:.4f}")
    if dp_mean < 0.01:
        print(f"  *** D_p≈0 → projector outputs identical regardless of memory → h_q only")
    elif dp_mean < 0.1:
        print(f"  *** D_p small → memory has minor influence on prefix")
    else:
        print(f"  *** D_p non-trivial → projector does differentiate memories")

    print(f"  Δz_y (score_target with m+ vs m-):  mean={dz_mean:+.4f}  std={dz_std:.4f}  "
          f"n={dz_n}  pos={dz_pos}({100*dz_pos/dz_n:.0f}%)  neg={dz_neg}({100*dz_neg/dz_n:.0f}%)")
    if abs(dz_mean) < 0.001:
        print(f"  *** Δz_y≈0 → routing change not transmitted to recommendation scores")
    elif dz_mean > 0:
        print(f"  *** Δz_y>0 → correct memory boosts target score (expected direction)")
    else:
        print(f"  *** Δz_y<0 → correct memory hurts target score (unexpected!)")

    return {
        "n_valid": n_valid,
        "dp_mean": dp_mean, "dp_std": dp_std,
        "dp_p10": dp_p10, "dp_p50": dp_p50, "dp_p90": dp_p90,
        "dz_mean": dz_mean, "dz_std": dz_std, "dz_n": dz_n,
        "dz_pos_frac": dz_pos / dz_n if dz_n > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    ap = argparse.ArgumentParser(description="STEP2+3+4: boundary crossing / candidate augmentation / projector sensitivity")
    ap.add_argument("--category",    default="Books")
    ap.add_argument("--data_dir",    default="data/processed")
    ap.add_argument("--sasrec_ckpt", default="checkpoints/Books/sasrec_pretrain.pt")
    ap.add_argument("--ckpt_2a",     default="checkpoints/Books/stage2_stage1enc_best.pt")
    ap.add_argument("--ckpt_2b",     default="checkpoints/Books/stage2_rawbge_best.pt")
    ap.add_argument("--bank_jsonl",  default="data/memory/Books/f3_bank.jsonl")
    ap.add_argument("--pairs_jsonl", default=None,
                    help="align_pairs.jsonl; defaults to data/processed/{category}/align_pairs.jsonl")
    ap.add_argument("--device",      default="cpu")
    ap.add_argument("--step4_n",     type=int, default=1000,
                    help="Users sampled for projector sensitivity (STEP4)")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    device = args.device
    pairs_jsonl = args.pairs_jsonl or f"data/processed/{args.category}/align_pairs.jsonl"
    M = 100
    B_VALUES = [10, 20, 50]
    K_BOUNDARY = [10, 20, 50, 100, 200]

    # ── SASRec backbone ──────────────────────────────────────────────────────
    print(f"\n[model] Loading SASRec from {args.sasrec_ckpt} ...")
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
    eval_args = SimpleNamespace(maxlen=saved.maxlen, device=device)
    dataset = load_data(args.category, args.data_dir)
    model = SASRec(dataset["usernum"], dataset["itemnum"], model_args).to(device)
    model.load_state_dict(ckpt_sas["model_state_dict"])
    model.eval()
    print(f"  users={dataset['usernum']}  items={dataset['itemnum']}")

    # ── Bank + queries ───────────────────────────────────────────────────────
    print(f"\n[data] bank={args.bank_jsonl}  pairs={pairs_jsonl}")
    user_memories = _load_bank_full(args.bank_jsonl)
    mid_to_uid = {m["mid"]: uid for uid, mems in user_memories.items() for m in mems}
    user_queries, _ = _load_pseudo_queries(pairs_jsonl, mid_to_uid=mid_to_uid)
    user_first_mid = _load_user_first_pos_mid(pairs_jsonl, mid_to_uid)
    print(f"  bank_users={len(user_memories)}  query_users={len(user_queries)}  "
          f"first_mid_users={len(user_first_mid)}")

    # ── Vanilla runs (need top-M items for STEP3) ────────────────────────────
    print("\n[STEP2+3] Running vanilla eval (top-M items) ...")
    vanilla_ranks, vanilla_topM = run_full_eval(model, dataset, eval_args,
                                                prefix_fn=None, return_topk=M)

    all_results = {}

    for tag, ckpt_path in [("2a", args.ckpt_2a), ("2b", args.ckpt_2b)]:
        if not Path(ckpt_path).exists():
            print(f"\n[skip] {tag}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Checkpoint: {tag}  ({ckpt_path})")
        print(f"{'='*60}")

        text_enc, proj = load_stage2(ckpt_path, device)
        prefix_fn = make_steered_prefix_fn(
            text_enc, proj, user_queries, user_memories, device, seed=args.seed,
        )

        # Steered eval — need top-M for STEP3
        print(f"\n[STEP2+3:{tag}] Running steered eval ...")
        steered_ranks, steered_topM = run_full_eval(
            model, dataset, eval_args, prefix_fn=prefix_fn, return_topk=M,
        )

        # ── STEP 2 ──────────────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  STEP 2 — Boundary Crossing  [{tag}]")
        print(f"{'─'*50}")
        s2 = step2_boundary_crossing(vanilla_ranks, steered_ranks, tag, ks=K_BOUNDARY)

        # ── STEP 3 ──────────────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  STEP 3 — Candidate Augmentation  [{tag}]")
        print(f"{'─'*50}")
        s3 = step3_candidate_augmentation(
            vanilla_ranks, vanilla_topM, steered_topM,
            dataset, label=tag, M=M, bs=B_VALUES,
        )

        # ── STEP 4 ──────────────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  STEP 4 — Projector Memory Sensitivity  [{tag}]")
        print(f"{'─'*50}")
        s4 = step4_projector_sensitivity(
            model, dataset, eval_args, text_enc, proj,
            user_queries, user_memories, user_first_mid,
            label=tag, n_sample=args.step4_n, seed=args.seed,
        )

        all_results[tag] = {"step2": s2, "step3": s3, "step4": s4}

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path("results/boundary")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.category}_boundary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
