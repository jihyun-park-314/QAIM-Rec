"""Full-catalog ranking evaluation for SASRec.

Replaces pmixer's 100-negative-sampling evaluation with full-catalog ranking:
  Recall@{5,10,20}, NDCG@{5,10,20}, MRR@{5,10,20}

No negative sampling — scores all items, filters past interactions.
"""
from __future__ import annotations

import copy
import math
from typing import Callable

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
    prefix_fn: Callable[[int, "np.ndarray"], "torch.Tensor | None"] | None = None,
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

        # Compute user representation (with optional steering prefix)
        pfx = prefix_fn(u, seq_np) if prefix_fn is not None else None
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat = log_feats[0, -1, :]  # [d] — last item pos (prefix prepended, so [-1] stays correct)

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


@torch.no_grad()
def evaluate_full_with_ranks(
    model,
    dataset: dict,
    args,
    split: str = "test",
    max_users: int = 10_000,
    prefix_fn: Callable[[int, "np.ndarray"], "torch.Tensor | None"] | None = None,
) -> tuple[dict[str, float], dict[int, int]]:
    """Like evaluate_full but also returns {user_id: rank} for rank-delta analysis."""
    model.eval()
    dev = args.device

    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test = dataset["user_test"]
    usernum = dataset["usernum"]
    itemnum = dataset["itemnum"]

    if split == "test":
        target_dict = user_test
        history_extra = user_valid
    else:
        target_dict = user_valid
        history_extra = {}

    all_users = [u for u in range(1, usernum + 1) if user_train.get(u) and target_dict.get(u)]
    if len(all_users) > max_users:
        import random
        rng = random.Random(42)
        all_users = rng.sample(all_users, max_users)

    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)

    accum = {f"{m}@{k}": 0.0 for m in ["Recall", "NDCG", "MRR"] for k in _KS}
    n_valid = 0
    user_ranks: dict[int, int] = {}

    for u in all_users:
        target_items = target_dict[u]
        if not target_items:
            continue
        target = target_items[0]

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
        pfx = prefix_fn(u, seq_np) if prefix_fn is not None else None
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat = log_feats[0, -1, :]
        scores = all_item_embs.matmul(final_feat)

        seen = set(user_train[u])
        seen.discard(0)
        if split == "test" and history_extra.get(u):
            seen.update(history_extra[u])
        seen_idx = torch.tensor([i - 1 for i in seen if 1 <= i <= itemnum],
                                dtype=torch.long, device=dev)
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        rank_of_target = (scores > scores[target - 1]).sum().item()
        user_ranks[u] = rank_of_target

        n_valid += 1
        for k in _KS:
            if rank_of_target < k:
                accum[f"Recall@{k}"] += 1.0
                accum[f"NDCG@{k}"] += 1.0 / math.log2(rank_of_target + 2)
            accum[f"MRR@{k}"] += 1.0 / (rank_of_target + 1) if rank_of_target < k else 0.0

    if n_valid == 0:
        return {k: 0.0 for k in accum}, {}

    metrics = {k: round(v / n_valid, 6) for k, v in accum.items()}
    return metrics, user_ranks


def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    header = f"{prefix} " if prefix else ""
    for k in _KS:
        r = metrics.get(f"Recall@{k}", 0)
        n = metrics.get(f"NDCG@{k}", 0)
        m = metrics.get(f"MRR@{k}", 0)
        print(f"  {header}@{k:2d}: Recall={r:.4f}  NDCG={n:.4f}  MRR={m:.4f}")


def load_warm_users(sequences_jsonl_path: str) -> set[int]:
    """Return set of user IDs (int) that have ≥1 memory-eligible interaction.

    'Warm' = has at least one interaction with rating≥4 + ≥10 words in train.
    'Cold' = no eligible interactions → prototype/DEFAULT_INTENT only.
    """
    import json as _json
    warm: set[int] = set()
    try:
        with open(sequences_jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                if any(it.get("is_eligible") for it in rec.get("items", [])):
                    warm.add(rec["user_id"])
    except FileNotFoundError:
        pass
    return warm


@torch.no_grad()
def evaluate_full_stratified(
    model,
    dataset: dict,
    args,
    split: str = "test",
    max_users: int = 10_000,
    sequences_jsonl_path: str | None = None,
    prefix_fn: Callable[[int, "np.ndarray"], "torch.Tensor | None"] | None = None,
) -> dict[str, dict[str, float]]:
    """Full-catalog ranking with warm/cold stratification.

    Returns:
        {
            "overall": {metric: float, ...},
            "warm":    {metric: float, ...},   # users with ≥1 eligible interaction
            "cold":    {metric: float, ...},   # users with 0 eligible interactions
            "counts":  {"overall": int, "warm": int, "cold": int},
        }
    """
    warm_users: set[int] = set()
    if sequences_jsonl_path:
        warm_users = load_warm_users(sequences_jsonl_path)

    model.eval()
    dev = args.device

    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test = dataset["user_test"]
    usernum = dataset["usernum"]
    itemnum = dataset["itemnum"]

    if split == "test":
        target_dict = user_test
        history_extra = user_valid
    else:
        target_dict = user_valid
        history_extra = {}

    all_users = [u for u in range(1, usernum + 1) if user_train.get(u) and target_dict.get(u)]
    if len(all_users) > max_users:
        import random
        rng = random.Random(42)
        all_users = rng.sample(all_users, max_users)

    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)

    def _empty_accum():
        return {f"{m}@{k}": 0.0 for m in ["Recall", "NDCG", "MRR"] for k in _KS}

    accum = {"overall": _empty_accum(), "warm": _empty_accum(), "cold": _empty_accum()}
    n_valid = {"overall": 0, "warm": 0, "cold": 0}

    for u in all_users:
        target_items = target_dict[u]
        if not target_items:
            continue
        target = target_items[0]

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
        pfx = prefix_fn(u, seq_np) if prefix_fn is not None else None
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat = log_feats[0, -1, :]
        scores = all_item_embs.matmul(final_feat)

        seen = set(user_train[u])
        seen.discard(0)
        if split == "test" and history_extra.get(u):
            seen.update(history_extra[u])
        seen_idx = torch.tensor([i - 1 for i in seen if 1 <= i <= itemnum],
                                dtype=torch.long, device=dev)
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        rank_of_target = (scores > scores[target - 1]).sum().item()

        tier = "warm" if (warm_users and u in warm_users) else "cold"
        for group in ("overall", tier):
            n_valid[group] += 1
            for k in _KS:
                if rank_of_target < k:
                    accum[group][f"Recall@{k}"] += 1.0
                    accum[group][f"NDCG@{k}"] += 1.0 / math.log2(rank_of_target + 2)
                accum[group][f"MRR@{k}"] += 1.0 / (rank_of_target + 1) if rank_of_target < k else 0.0

    def _normalise(a: dict, n: int) -> dict[str, float]:
        if n == 0:
            return {k: 0.0 for k in a}
        return {k: round(v / n, 6) for k, v in a.items()}

    return {
        "overall": _normalise(accum["overall"], n_valid["overall"]),
        "warm":    _normalise(accum["warm"],    n_valid["warm"]),
        "cold":    _normalise(accum["cold"],    n_valid["cold"]),
        "counts":  n_valid,
    }
