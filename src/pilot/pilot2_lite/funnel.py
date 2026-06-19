"""Task 2-A: Eligibility funnel — no LLM, all users.

For each domain:
  - Stream ALL reviews.
  - Per user: sort interactions by timestamp.
  - Split: train_history = interactions[:-2], val = [-2], test = [-1].
  - eligible interaction: user_id + parent_asin + timestamp exist, token_count >= 10.
  - eligible user: train_history has >= 8 eligible interactions.

Reports: total_users, users_with_eligible_ge_8, eligibility_ratio,
         eligible-history length stats.

Also saves eligible user interaction data (for smoke.py):
  compact list of {user_id, eligible_train_interactions=[{timestamp, parent_asin, review_text, ...}]}
  — only for users selected by pick_eligible_users().
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


_MIN_TOKENS = 10


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _review_text(row: dict) -> str:
    return row.get("text") or row.get("review_text") or row.get("content") or ""


def _is_eligible_interaction(row: dict) -> bool:
    if not (row.get("user_id") and row.get("parent_asin") and row.get("timestamp")):
        return False
    text = _review_text(row)
    return len(text.split()) >= _MIN_TOKENS


def run_funnel(domain: str, data_dir: str = "data/raw") -> dict:
    reviews_path = str(Path(data_dir) / domain / "reviews.jsonl")

    print(f"[funnel] Streaming {domain} ...")

    # Per-user: list of (timestamp, is_eligible)
    # Using compact tuples to minimize memory.
    user_interactions: dict[str, list[tuple[int, bool]]] = {}

    line_count = 0
    for row in _iter_jsonl(reviews_path):
        line_count += 1
        if line_count % 2_000_000 == 0:
            print(f"  ... {line_count:,} lines, {len(user_interactions):,} users")

        uid = row.get("user_id") or ""
        if not uid:
            continue
        ts = row.get("timestamp")
        if ts is None:
            ts = 0
        pa = row.get("parent_asin") or ""
        text = _review_text(row)
        eligible = bool(pa and ts and len(text.split()) >= _MIN_TOKENS)

        if uid not in user_interactions:
            user_interactions[uid] = []
        user_interactions[uid].append((int(ts), eligible))

    print(f"[funnel] {domain}: {line_count:,} reviews, {len(user_interactions):,} unique users")

    # Compute funnel stats
    total_users = len(user_interactions)
    eligible_user_count = 0
    eligible_history_lengths: list[int] = []

    eligible_users: list[str] = []

    for uid, interactions in user_interactions.items():
        if len(interactions) < 3:
            # Need at least 3 to have a non-empty train set (last 2 go to val/test)
            continue
        interactions_sorted = sorted(interactions, key=lambda x: x[0])
        train = interactions_sorted[:-2]
        n_eligible_train = sum(1 for _, elig in train if elig)
        if n_eligible_train >= 8:
            eligible_user_count += 1
            eligible_history_lengths.append(n_eligible_train)
            eligible_users.append(uid)

    eligibility_ratio = eligible_user_count / total_users if total_users else 0.0

    def _pct(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        data_s = sorted(data)
        idx = int(len(data_s) * p)
        return float(data_s[min(idx, len(data_s) - 1)])

    history_stats = {
        "median": _pct(eligible_history_lengths, 0.50),
        "p25": _pct(eligible_history_lengths, 0.25),
        "p75": _pct(eligible_history_lengths, 0.75),
        "mean": round(statistics.mean(eligible_history_lengths), 1) if eligible_history_lengths else 0.0,
    }

    return {
        "domain": domain,
        "total_users": total_users,
        "users_with_eligible_ge_8": eligible_user_count,
        "eligibility_ratio": round(eligibility_ratio, 4),
        "eligible_history_length_stats": history_stats,
        "_eligible_user_ids": eligible_users,
        "_user_interactions": user_interactions,
    }


def save_funnel_cache(funnel_result: dict, cache_path: str) -> None:
    """Save eligible user IDs and stats to disk (lightweight, no review text)."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    payload = {k: v for k, v in funnel_result.items() if not k.startswith("_")}
    payload["_eligible_user_ids"] = funnel_result.get("_eligible_user_ids", [])
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_funnel_cache(cache_path: str) -> dict | None:
    """Load funnel cache from disk. Returns None if not found."""
    if not Path(cache_path).is_file():
        return None
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


_COUNT_THRESHOLDS = [3, 5, 8]
_LENGTH_GATES = [1, 3, 10]   # non-empty, ≥3 words, ≥10 words
_RATING_GATES = ["all", "ge4", "eq5"]


def _length_label(lg: int) -> str:
    if lg == 1:
        return "nonempty"
    return f"ge{lg}w"


def run_funnel_sweep(domain: str, data_dir: str = "data/raw") -> dict:
    """3-axis funnel sweep: count ∈ {3,5,8} × length ∈ {1,3,10} × rating ∈ {all,≥4,=5}.

    Streams ALL reviews once. Per-user: keeps (timestamp, rating, word_count) for
    valid interactions (user_id + parent_asin + timestamp present).
    Applies leave-one-out split: train = all but last 2 (by timestamp).
    Returns per-domain, per-rating 3×3 table (count × length).
    """
    reviews_path = str(Path(data_dir) / domain / "reviews.jsonl")

    print(f"[funnel_sweep] Streaming {domain} ...")

    # Compact per-user store: list of (timestamp, rating, word_count)
    user_interactions: dict[str, list[tuple[int, float, int]]] = {}

    line_count = 0
    for row in _iter_jsonl(reviews_path):
        line_count += 1
        if line_count % 5_000_000 == 0:
            print(f"  ... {line_count:,} lines, {len(user_interactions):,} users")

        uid = row.get("user_id") or ""
        if not uid:
            continue
        ts = row.get("timestamp")
        if not ts:
            continue
        pa = row.get("parent_asin") or ""
        if not pa:
            continue
        text = _review_text(row)
        wc = len(text.split())
        rating = float(row.get("rating") or 0.0)

        if uid not in user_interactions:
            user_interactions[uid] = []
        user_interactions[uid].append((int(ts), rating, wc))

    total_users = len(user_interactions)
    print(f"[funnel_sweep] {domain}: {line_count:,} reviews, {total_users:,} users")

    def _pct(data: list, p: float) -> float:
        if not data:
            return 0.0
        ds = sorted(data)
        idx = int(len(ds) * p)
        return float(ds[min(idx, len(ds) - 1)])

    # Pre-sort and split each user's interactions once
    user_trains: dict[str, list[tuple[int, float, int]]] = {}
    for uid, ints in user_interactions.items():
        if len(ints) < 3:
            continue
        sorted_ints = sorted(ints, key=lambda x: x[0])
        user_trains[uid] = sorted_ints[:-2]  # train only

    # Sweep all axis combinations
    rating_tables: dict[str, dict] = {}
    for rg in _RATING_GATES:
        cells: dict[str, dict] = {}
        for ct in _COUNT_THRESHOLDS:
            for lg in _LENGTH_GATES:
                eligible_users = 0
                history_lens: list[int] = []

                for uid, train in user_trains.items():
                    eligible_count = 0
                    for ts, rating, wc in train:
                        # Rating gate
                        if rg == "ge4" and rating < 4.0:
                            continue
                        if rg == "eq5" and rating != 5.0:
                            continue
                        # Length gate
                        if wc < lg:
                            continue
                        eligible_count += 1

                    if eligible_count >= ct:
                        eligible_users += 1
                        history_lens.append(eligible_count)

                cell_key = f"count{ct}_{_length_label(lg)}"
                cells[cell_key] = {
                    "count_threshold": ct,
                    "length_gate": _length_label(lg),
                    "eligible_users": eligible_users,
                    "eligibility_ratio": round(eligible_users / total_users, 6) if total_users else 0.0,
                    "history_len_median": _pct(history_lens, 0.50),
                    "history_len_p25": _pct(history_lens, 0.25),
                    "history_len_p75": _pct(history_lens, 0.75),
                }

        rating_tables[rg] = {
            "total_users": total_users,
            "cells": cells,
        }

    return {
        "domain": domain,
        "total_users": total_users,
        "rating_tables": rating_tables,
    }


def pick_eligible_users(
    funnel_result: dict,
    n: int = 10,
    seed: int = 42,
) -> list[str]:
    """Pick n eligible users deterministically from funnel result."""
    import random
    eligible = funnel_result["_eligible_user_ids"]
    rng = random.Random(seed)
    sampled = rng.sample(eligible, min(n, len(eligible)))
    return sampled


def collect_eligible_user_ids(
    domain: str,
    count_threshold: int = 5,
    min_words: int = 10,
    min_rating: float = 4.0,
    data_dir: str = "data/raw",
) -> list[str]:
    """One-pass stream → eligible user IDs for a specific count/length/rating combo.
    Used by smoke Task 2-B to build the count5+ge4+ge10w pool.
    Leave-one-out split applied: train = all but last 2 by timestamp.
    """
    reviews_path = str(Path(data_dir) / domain / "reviews.jsonl")
    print(f"[funnel_ids] {domain}: streaming (count≥{count_threshold},"
          f" ge{min_words}w, rating≥{min_rating}) ...")

    user_data: dict[str, list[tuple[int, float, int]]] = {}
    line_count = 0
    for row in _iter_jsonl(reviews_path):
        line_count += 1
        if line_count % 5_000_000 == 0:
            print(f"  ... {line_count:,} lines, {len(user_data):,} users")
        uid = row.get("user_id") or ""
        if not uid:
            continue
        ts = row.get("timestamp")
        if not ts:
            continue
        pa = row.get("parent_asin") or ""
        if not pa:
            continue
        text = _review_text(row)
        wc = len(text.split())
        rating = float(row.get("rating") or 0.0)
        if uid not in user_data:
            user_data[uid] = []
        user_data[uid].append((int(ts), rating, wc))

    eligible_ids: list[str] = []
    for uid, ints in user_data.items():
        if len(ints) < 3:
            continue
        sorted_ints = sorted(ints, key=lambda x: x[0])
        train = sorted_ints[:-2]
        eligible_count = sum(
            1 for _, rating, wc in train
            if rating >= min_rating and wc >= min_words
        )
        if eligible_count >= count_threshold:
            eligible_ids.append(uid)

    print(f"[funnel_ids] {domain}: {len(eligible_ids):,} eligible users")
    return eligible_ids


def collect_user_interactions(
    domain: str,
    user_ids: list[str],
    funnel_result: dict,
    max_per_user: int = 8,
    min_rating: float = 0.0,
    data_dir: str = "data/raw",
) -> list[dict[str, Any]]:
    """Collect full interaction data for selected users.

    Streams reviews once to collect ALL interactions for the given user_ids,
    sorts by timestamp, removes last 2 (val/test), picks most recent max_per_user
    eligible from train. Does not require the full _user_interactions compact dict.

    Returns list of user dicts:
      {user_id, total_eligible_train_count, selected_interaction_count,
       eligible_train_interactions: [{timestamp, parent_asin, review_text, rating,
                                      title, category, brand, price}]}
    """
    user_id_set = set(user_ids)

    # Stream reviews to collect ALL interactions for selected users
    reviews_path = str(Path(data_dir) / domain / "reviews.jsonl")
    # Collect ALL interactions (eligible or not) to correctly determine train/val/test split.
    user_buckets: dict[str, list[dict]] = {uid: [] for uid in user_ids}

    for row in _iter_jsonl(reviews_path):
        uid = row.get("user_id") or ""
        if uid not in user_id_set:
            continue
        pa = row.get("parent_asin") or ""
        ts = row.get("timestamp") or 0
        text = _review_text(row)
        user_buckets[uid].append({
            "timestamp": int(ts),
            "parent_asin": pa,
            "review_text": text,
            "rating": row.get("rating") or 0.0,
            "_eligible": bool(pa and ts and len(text.split()) >= _MIN_TOKENS),
        })

    # Build meta lookup for eligible interactions
    all_asins = set()
    for interactions in user_buckets.values():
        for it in interactions:
            if it.get("_eligible"):
                all_asins.add(it["parent_asin"])

    meta_path = str(Path(data_dir) / domain / "meta.jsonl")
    meta_lookup: dict[str, dict] = {}
    for row in _iter_jsonl(meta_path):
        pa = row.get("parent_asin") or row.get("asin") or ""
        if pa in all_asins:
            cats = row.get("categories") or []
            cat0 = cats[0] if isinstance(cats, list) and cats else (row.get("main_category") or "")
            meta_lookup[pa] = {
                "title": row.get("title") or "",
                "category": cat0,
                "brand": row.get("brand") or row.get("store") or "",
                "price": str(row.get("price") or "unknown"),
            }

    # For each user: sort all interactions by timestamp, remove last 2 (val/test),
    # then filter to eligible within train, take most recent max_per_user.
    results = []
    for uid in user_ids:
        all_sorted = sorted(user_buckets[uid], key=lambda x: x["timestamp"])
        if len(all_sorted) < 3:
            continue
        train_all = all_sorted[:-2]
        train_eligible = [
            it for it in train_all
            if it.get("_eligible") and float(it.get("rating") or 0.0) >= min_rating
        ]
        if not train_eligible:
            continue

        recent_n = train_eligible[-max_per_user:]

        enriched = []
        for it in recent_n:
            meta = meta_lookup.get(it["parent_asin"], {})
            enriched.append({
                "timestamp": it["timestamp"],
                "parent_asin": it["parent_asin"],
                "review_text": it["review_text"],
                "rating": it["rating"],
                "title": meta.get("title") or "(unknown)",
                "category": meta.get("category") or "",
                "brand": meta.get("brand") or "",
                "price": meta.get("price") or "unknown",
            })

        results.append({
            "user_id": uid,
            "total_eligible_train_count": len(train_eligible),
            "selected_interaction_count": len(enriched),
            "eligible_train_interactions": enriched,
        })

    return results
