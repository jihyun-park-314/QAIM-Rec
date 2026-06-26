"""STEP 1 — K_personal 재층화 eval.

Replaces the old warm/cold (eligible≥1) split with K=0 / K=1 / K≥2 buckets.
This is the true contribution axis of the paper (plan.md v0.4.5 §2.2-C).

Usage — smoke (20-user dump):
    docker exec qaim-rec python3 src/eval/stratify_kpersonal.py \\
        --bank_jsonl data/memory_full_test/memory_b_u20_seed42.jsonl \\
        --checkpoint checkpoints/Books/sasrec_pretrain.pt \\
        --category Books --device cpu

Usage — full bank (after extraction):
    docker exec qaim-rec python3 src/eval/stratify_kpersonal.py \\
        --bank_dir data/memory_full/bank/ \\
        --checkpoint checkpoints/Books/sasrec_pretrain.pt \\
        --category Books --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.sasrec import SASRec
from src.models.dataloader import load_data

_KS = [5, 10, 20]
_BUCKETS = ("k0", "k1", "k2plus")


# ---------------------------------------------------------------------------
# Bank loading — two source formats

def load_kpersonal_map(
    bank_jsonl: str | None = None,
    bank_dir: str | None = None,
) -> dict[int, int]:
    """Return {user_id (int): k_personal (int)} from bank source."""
    kmap: dict[int, int] = {}

    if bank_jsonl:
        with open(bank_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                uid = int(rec["user_id"])
                kmap[uid] = int(rec["k_personal"])

    elif bank_dir:
        bank_path = Path(bank_dir)
        for p in bank_path.glob("*.json"):
            if p.name.startswith("_"):
                continue
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            uid = int(rec["user_id"])
            kmap[uid] = int(rec["k_personal"])

    else:
        raise ValueError("Provide either --bank_jsonl or --bank_dir")

    return kmap


# ---------------------------------------------------------------------------
# K bucket assignment

def _bucket(k: int) -> str:
    if k == 0:
        return "k0"
    if k == 1:
        return "k1"
    return "k2plus"


# ---------------------------------------------------------------------------
# Stratified full-ranking eval

@torch.no_grad()
def evaluate_stratified_kpersonal(
    model: SASRec,
    dataset: dict,
    args: SimpleNamespace,
    kpersonal_map: dict[int, int],
    split: str = "test",
    restrict_to_bank: bool = True,
) -> dict:
    """Full-catalog ranking eval split by K=0 / K=1 / K≥2.

    Args:
        restrict_to_bank: if True, only evaluate users present in kpersonal_map
                          (smoke mode). If False, assign unmapped users to k0.
    Returns:
        {
            "overall": {metric: float},
            "k0": {metric: float},
            "k1": {metric: float},
            "k2plus": {metric: float},
            "counts": {"overall": n, "k0": n, "k1": n, "k2plus": n},
        }
    """
    model.eval()
    dev = args.device

    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test = dataset["user_test"]
    itemnum = dataset["itemnum"]

    if split == "test":
        target_dict = user_test
        history_extra = user_valid
    else:
        target_dict = user_valid
        history_extra = {}

    all_users = [u for u in user_train if target_dict.get(u)]
    if restrict_to_bank:
        all_users = [u for u in all_users if u in kpersonal_map]

    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)  # [I, d]

    def _empty():
        return {f"{m}@{k}": 0.0 for m in ["Recall", "NDCG", "MRR"] for k in _KS}

    accum = {b: _empty() for b in ("overall", *_BUCKETS)}
    counts = {b: 0 for b in ("overall", *_BUCKETS)}

    for u in all_users:
        target_items = target_dict[u]
        if not target_items:
            continue
        target = target_items[0]

        # Build right-aligned input sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if split == "test" and history_extra.get(u):
            for extra in reversed(history_extra[u]):
                seq[idx] = extra
                idx -= 1
                if idx == -1:
                    break
        for i in reversed(user_train[u]):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1

        seq_np = seq[np.newaxis, :]
        log_feats, _ = model.log2feats(seq_np)
        final_feat = log_feats[0, -1, :]

        scores = all_item_embs.matmul(final_feat)  # [I]

        seen = set(user_train[u])
        seen.discard(0)
        if split == "test" and history_extra.get(u):
            seen.update(history_extra[u])
        seen_idx = torch.tensor(
            [i - 1 for i in seen if 1 <= i <= itemnum],
            dtype=torch.long, device=dev,
        )
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        rank = (scores > scores[target - 1]).sum().item()

        k_val = kpersonal_map.get(u, 0)
        bucket = _bucket(k_val)

        for grp in ("overall", bucket):
            counts[grp] += 1
            for k in _KS:
                if rank < k:
                    accum[grp][f"Recall@{k}"] += 1.0
                    accum[grp][f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)
                if rank < k:
                    accum[grp][f"MRR@{k}"] += 1.0 / (rank + 1)

    def _norm(a: dict, n: int) -> dict[str, float]:
        if n == 0:
            return {mk: 0.0 for mk in a}
        return {mk: round(v / n, 6) for mk, v in a.items()}

    return {
        "overall": _norm(accum["overall"], counts["overall"]),
        "k0":      _norm(accum["k0"],      counts["k0"]),
        "k1":      _norm(accum["k1"],      counts["k1"]),
        "k2plus":  _norm(accum["k2plus"],  counts["k2plus"]),
        "counts":  counts,
    }


# ---------------------------------------------------------------------------
# Pretty-print

def print_results(results: dict) -> None:
    counts = results["counts"]
    print("\n=== K_personal Stratified Eval ===")
    print(f"  Users evaluated: overall={counts['overall']}  "
          f"k0={counts['k0']}  k1={counts['k1']}  k≥2={counts['k2plus']}")
    print()
    label_map = {"overall": "Overall", "k0": "K=0   ", "k1": "K=1   ", "k2plus": "K≥2   "}
    for grp in ("overall", "k0", "k1", "k2plus"):
        m = results[grp]
        print(f"  [{label_map[grp]}]  (n={counts[grp]})")
        for k in _KS:
            r = m.get(f"Recall@{k}", 0)
            n = m.get(f"NDCG@{k}", 0)
            mrr = m.get(f"MRR@{k}", 0)
            print(f"    @{k:2d}: Recall={r:.4f}  NDCG={n:.4f}  MRR={mrr:.4f}")
        print()


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank_jsonl", type=str, default=None,
                        help="JSONL with user_id+k_personal per line (smoke/full)")
    parser.add_argument("--bank_dir", type=str, default=None,
                        help="Directory of per-user JSON files (from save_bank)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/Books/sasrec_pretrain.pt")
    parser.add_argument("--category", type=str, default="Books")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--restrict_to_bank", action="store_true", default=True,
                        help="Only eval users present in bank (auto-enabled for smoke)")
    cli = parser.parse_args()

    # ── load k_personal map ──────────────────────────────────────────────────
    print(f"[stratify] Loading bank k_personal map ...")
    kmap = load_kpersonal_map(bank_jsonl=cli.bank_jsonl, bank_dir=cli.bank_dir)
    k_counts = {"k0": 0, "k1": 0, "k2plus": 0}
    for k in kmap.values():
        k_counts[_bucket(k)] += 1
    print(f"[stratify] Bank users: {len(kmap)}  "
          f"k0={k_counts['k0']}  k1={k_counts['k1']}  k≥2={k_counts['k2plus']}")

    # ── load dataset ─────────────────────────────────────────────────────────
    print(f"[stratify] Loading dataset ({cli.category}) ...")
    dataset = load_data(cli.category, cli.data_dir)
    print(f"[stratify] Dataset: {dataset['usernum']} users, {dataset['itemnum']} items")

    # ── load checkpoint ──────────────────────────────────────────────────────
    print(f"[stratify] Loading checkpoint: {cli.checkpoint}")
    ckpt = torch.load(cli.checkpoint, map_location="cpu")
    saved_args = ckpt["args"]
    if isinstance(saved_args, dict):
        saved_args = SimpleNamespace(**saved_args)

    # Override device for eval
    eval_args = SimpleNamespace(
        maxlen=saved_args.maxlen,
        hidden_units=saved_args.hidden_units,
        num_blocks=saved_args.num_blocks,
        num_heads=saved_args.num_heads,
        dropout_rate=saved_args.dropout_rate,
        norm_first=saved_args.norm_first,
        device=cli.device,
    )

    model = SASRec(
        user_num=dataset["usernum"],
        item_num=dataset["itemnum"],
        args=eval_args,
    ).to(cli.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[stratify] Model loaded (epoch {ckpt.get('epoch', '?')})")

    # ── run eval ──────────────────────────────────────────────────────────────
    print(f"[stratify] Running {cli.split} eval (restrict_to_bank={cli.restrict_to_bank}) ...")
    results = evaluate_stratified_kpersonal(
        model=model,
        dataset=dataset,
        args=eval_args,
        kpersonal_map=kmap,
        split=cli.split,
        restrict_to_bank=cli.restrict_to_bank,
    )

    print_results(results)
    print("NOTE: smoke results are meaningless (20 users). "
          "Numbers will be valid after full bank build.")


if __name__ == "__main__":
    main()
