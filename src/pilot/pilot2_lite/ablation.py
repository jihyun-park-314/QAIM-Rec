"""Task 2-C: Short-review ablation.

Selects ~N_TARGET short positive reviews (3–9 words, rating≥4) per domain
from the train partition only (leave-one-out split: all but last 2 interactions).
Runs p1_aspect v4 + product metadata on each.
Saves raw LLM outputs for qualitative comparison vs long-review results.

Selection strategy:
  - Two-pass: first pass collects per-user timestamps to determine train/val/test
    boundaries; second pass collects candidate rows.
  - Candidates: word_count in [3, 9], rating≥4, not in last-2 (val/test).
  - Sample N_TARGET deterministically with seed=42.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient, load_llm_config
from src.llm.prompts import p1_aspect

_MIN_SHORT_WORDS = 3
_MAX_SHORT_WORDS = 9
_MIN_RATING = 4.0
_N_TARGET = 20
_SEED = 42
_LLM_CONFIG_PATH = "configs/llm/p1_aspect.yaml"
_ESTIMATED_SECONDS_PER_CALL = 8.0


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _review_text(row: dict) -> str:
    return row.get("text") or row.get("review_text") or row.get("content") or ""


def collect_short_reviews(
    domain: str,
    n_target: int = _N_TARGET,
    seed: int = _SEED,
    data_dir: str = "data/raw",
) -> list[dict]:
    """Two-pass collection of short positive train reviews.

    Pass 1: collect per-user all timestamps → determine val/test boundary.
    Pass 2: collect candidate (short+positive+train) rows + product ASINs.
    Then load meta, sample, return enriched list.
    """
    reviews_path = str(Path(data_dir) / domain / "reviews.jsonl")

    # Pass 1: per-user sorted timestamps → val/test set of (user_id, timestamp) pairs
    print(f"[ablation] {domain}: pass 1 — building train/val/test boundaries ...")
    user_timestamps: dict[str, list[int]] = {}
    for row in _iter_jsonl(reviews_path):
        uid = row.get("user_id") or ""
        ts = row.get("timestamp")
        if not uid or not ts:
            continue
        if uid not in user_timestamps:
            user_timestamps[uid] = []
        user_timestamps[uid].append(int(ts))

    # For each user with ≥3 interactions: val+test = last 2 timestamps
    val_test_keys: set[tuple[str, int]] = set()
    for uid, tss in user_timestamps.items():
        if len(tss) < 3:
            continue
        sorted_tss = sorted(tss)
        val_test_keys.add((uid, sorted_tss[-1]))
        val_test_keys.add((uid, sorted_tss[-2]))

    # Pass 2: collect candidates
    print(f"[ablation] {domain}: pass 2 — collecting short positive train reviews ...")
    candidates: list[dict] = []
    for row in _iter_jsonl(reviews_path):
        uid = row.get("user_id") or ""
        ts = row.get("timestamp")
        if not uid or not ts:
            continue
        if (uid, int(ts)) in val_test_keys:
            continue  # skip val/test
        if len(user_timestamps.get(uid, [])) < 3:
            continue  # user has no train set

        rating = float(row.get("rating") or 0.0)
        if rating < _MIN_RATING:
            continue

        text = _review_text(row)
        wc = len(text.split())
        if not (_MIN_SHORT_WORDS <= wc <= _MAX_SHORT_WORDS):
            continue

        pa = row.get("parent_asin") or ""
        candidates.append({
            "user_id": uid,
            "parent_asin": pa,
            "timestamp": int(ts),
            "rating": rating,
            "review_text": text,
            "word_count": wc,
        })

    print(f"[ablation] {domain}: {len(candidates):,} candidates ({_MIN_SHORT_WORDS}–{_MAX_SHORT_WORDS}w, ≥{_MIN_RATING}★, train-only)")

    if len(candidates) <= n_target:
        selected = candidates
    else:
        rng = random.Random(seed)
        selected = rng.sample(candidates, n_target)

    # Load meta for selected ASINs
    asin_set = {r["parent_asin"] for r in selected if r["parent_asin"]}
    meta_path = str(Path(data_dir) / domain / "meta.jsonl")
    meta_lookup: dict[str, dict] = {}
    for row in _iter_jsonl(meta_path):
        pa = row.get("parent_asin") or row.get("asin") or ""
        if pa in asin_set:
            cats = row.get("categories") or []
            cat0 = cats[0] if isinstance(cats, list) and cats else (row.get("main_category") or "")
            meta_lookup[pa] = {
                "title": row.get("title") or "",
                "category": cat0,
                "brand": row.get("brand") or row.get("store") or "",
                "price": str(row.get("price") or "unknown"),
            }

    for r in selected:
        meta = meta_lookup.get(r["parent_asin"], {})
        r["title"] = meta.get("title") or "(unknown)"
        r["category"] = meta.get("category") or ""
        r["brand"] = meta.get("brand") or ""
        r["price"] = meta.get("price") or "unknown"

    return selected


def estimate_ablation_calls(reviews_per_domain: dict[str, list]) -> dict:
    total = sum(len(v) for v in reviews_per_domain.values())
    secs = total * _ESTIMATED_SECONDS_PER_CALL
    return {
        "total_estimated_calls": total,
        "per_domain": {d: len(v) for d, v in reviews_per_domain.items()},
        "estimated_seconds": round(secs),
        "estimated_minutes": round(secs / 60, 1),
    }


def run_ablation_for_domain(
    domain: str,
    reviews: list[dict],
    domain_type: str = "lifestyle",
) -> list[dict[str, Any]]:
    """Run p1_aspect v4 on each short review. Returns per-review result dicts."""
    import time

    llm_cfg = load_llm_config(_LLM_CONFIG_PATH)
    llm_cfg.retry_max = 2
    llm_cfg.prompt_version = p1_aspect.PROMPT_VERSION
    client = LLMClient(llm_cfg)

    results = []
    for r in reviews:
        item = {
            "title": r["title"],
            "category": r["category"],
            "brand": r["brand"],
            "price": r["price"],
            "rating": r["rating"],
            "review_text": r["review_text"],
            "parent_asin": r["parent_asin"],
            "user_id": r["user_id"],
        }
        t0 = time.time()
        res = p1_aspect.run_p1_aspect(client, item, retry_max=2)
        latency = round(time.time() - t0, 3)

        results.append({
            "user_id": r["user_id"],
            "parent_asin": r["parent_asin"],
            "title": r["title"],
            "rating": r["rating"],
            "word_count": r["word_count"],
            "review_text": r["review_text"],
            "parse_success": res.parsed is not None,
            "cache_hit": res.cache_hit,
            "latency_s": latency,
            "retry_count": res.retry_count,
            "llm_output": res.parsed,
        })

    client.close()
    return results
