"""Full-catalog ranking evaluation for SASRec.

Replaces pmixer's 100-negative-sampling evaluation with full-catalog ranking:
  Recall@{5,10,20}, NDCG@{5,10,20}, MRR@{5,10,20}

No negative sampling — scores all items, filters past interactions.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

_KS = [5, 10, 20]


@torch.no_grad()
def evaluate_full(
    model,
    dataset: dict,
    args,
    split: str = "test",
    max_users: int = 10_000,
) -> dict[str, float]:
    """Full-catalog ranking evaluation.

    Args:
        model: SASRec instance (eval mode)
        dataset: output of dataloader.load_data()
        args: namespace with maxlen, device
        split: "test" or "val"
        max_users: cap evaluation at this many users (sample if larger)

    Returns:
        dict of metric_name → float, e.g. {"Recall@10": 0.123, ...}
    """
    model.eval()
    dev = args.device

    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test = dataset["user_test"]
    usernum = dataset["usernum"]
    itemnum = dataset["itemnum"]

    if split == "test":
        target_dict = user_test
        # For test: include val item in the "seen" history for sequence building
        history_extra = user_valid
    else:
        target_dict = user_valid
        history_extra = {}

    all_users = [u for u in range(1, usernum + 1) if user_train.get(u) and target_dict.get(u)]
    if len(all_users) > max_users:
        import random
        rng = random.Random(42)
        all_users = rng.sample(all_users, max_users)

    # All item indices [1..I] as tensor
    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)  # [I]
    all_item_embs = model.item_emb(all_items)  # [I, d]

    accum = {f"{m}@{k}": 0.0 for m in ["Recall", "NDCG", "MRR"] for k in _KS}
    n_valid = 0

    for u in all_users:
        target_items = target_dict[u]
        if not target_items:
            continue
        target = target_items[0]  # single target for LOO

        # Build input sequence (right-aligned, maxlen)
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        # For test: include val item in sequence
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

        seq_np = seq[np.newaxis, :]  # [1, maxlen] — keep as ndarray for log2feats

        # Compute user representation
        log_feats, _ = model.log2feats(seq_np)
        final_feat = log_feats[0, -1, :]  # [d]

        # Score all items
        scores = all_item_embs.matmul(final_feat)  # [I]

        # Mask past interactions (set to -inf)
        seen = set(user_train[u])
        seen.discard(0)
        if split == "test" and history_extra.get(u):
            seen.update(history_extra[u])
        seen_idx = torch.tensor([i - 1 for i in seen if 1 <= i <= itemnum],
                                dtype=torch.long, device=dev)
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        # Rank (descending)
        rank_of_target = (scores > scores[target - 1]).sum().item()  # 0-indexed rank

        n_valid += 1
        for k in _KS:
            if rank_of_target < k:
                accum[f"Recall@{k}"] += 1.0
                accum[f"NDCG@{k}"] += 1.0 / math.log2(rank_of_target + 2)
            accum[f"MRR@{k}"] += 1.0 / (rank_of_target + 1) if rank_of_target < k else 0.0

    if n_valid == 0:
        return {k: 0.0 for k in accum}

    return {k: round(v / n_valid, 6) for k, v in accum.items()}


def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    header = f"{prefix} " if prefix else ""
    for k in _KS:
        r = metrics.get(f"Recall@{k}", 0)
        n = metrics.get(f"NDCG@{k}", 0)
        m = metrics.get(f"MRR@{k}", 0)
        print(f"  {header}@{k:2d}: Recall={r:.4f}  NDCG={n:.4f}  MRR={m:.4f}")
