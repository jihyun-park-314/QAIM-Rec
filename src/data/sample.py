"""M0: Local JSONL sampler for (item_meta, review) pairs.

Reads from data/raw/{category}/ — HF streaming/download is strictly forbidden.
Fails clearly if files are missing.

Expected layout (Amazon Reviews 2023 JSONL export):
  data/raw/{category}/reviews.jsonl   — one review JSON per line
  data/raw/{category}/meta.jsonl      — one item-meta JSON per line

Review fields used (flexible): text, rating, user_id, parent_asin (or asin)
Meta fields used (flexible): title, price, store/brand, main_category/categories, parent_asin
"""

from __future__ import annotations

import json
import os
import random
from typing import Any


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _review_text(row: dict) -> str:
    return (
        row.get("text")
        or row.get("review_text")
        or row.get("content")
        or row.get("reviewText")
        or ""
    )


def _parent_asin(row: dict) -> str | None:
    """Returns parent_asin if present, else None. Falls back to asin is NOT allowed
    (asin is the variant ASIN, not the parent — plan.md M0: skip if parent_asin missing)."""
    pa = row.get("parent_asin") or row.get("parentAsin")
    return pa or None


def _meta_asin(row: dict) -> str:
    return row.get("parent_asin") or row.get("asin") or row.get("parentAsin") or ""


def _meta_title(row: dict) -> str:
    return row.get("title") or row.get("productTitle") or ""


def _meta_price(row: dict) -> str:
    v = row.get("price")
    if v is None:
        return "unknown"
    return str(v)


def _meta_brand(row: dict) -> str:
    return (
        row.get("brand")
        or row.get("store")
        or (row.get("details") or {}).get("Brand", "")
        or "unknown"
    )


def _meta_category(row: dict, config_category: str = "") -> str:
    c = row.get("main_category") or row.get("category") or row.get("categories")
    if isinstance(c, list):
        return c[0] if c else (config_category or "unknown")
    return str(c) if c else (config_category or "unknown")


def _locate(category: str, data_dir: str = "data/raw") -> tuple[str, str]:
    base = os.path.join(data_dir, category)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"Category directory not found: {base!r}. "
            "Place local JSONL files under data/raw/{category}/ before running."
        )
    reviews_path = os.path.join(base, "reviews.jsonl")
    meta_path = os.path.join(base, "meta.jsonl")
    if not os.path.isfile(reviews_path):
        raise FileNotFoundError(
            f"Reviews file not found: {reviews_path!r}. "
            "Expected data/raw/{category}/reviews.jsonl"
        )
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Meta file not found: {meta_path!r}. "
            "Expected data/raw/{category}/meta.jsonl"
        )
    return reviews_path, meta_path


def sample_review_pairs(
    category: str,
    n_samples: int,
    seed: int,
    min_review_tokens: int = 10,
    data_dir: str = "data/raw",
) -> list[dict[str, Any]]:
    """Sample n_samples (item_meta, review) pairs from local JSONL files.

    Fails immediately if data/raw/{category}/ or its JSONL files are missing.
    Uses reservoir sampling with the given seed for reproducibility.
    """
    reviews_path, meta_path = _locate(category, data_dir)

    rng = random.Random(seed)
    reservoir: list[dict] = []

    for row in _iter_jsonl(reviews_path):
        text = _review_text(row)
        if len(text.split()) < min_review_tokens:
            continue
        pa = _parent_asin(row)
        if not pa:
            continue  # skip: no parent_asin — cannot link to item meta (plan.md M0)
        entry = {
            "_parent_asin": pa,
            "rating": row.get("rating") or row.get("overall") or 0,
            "review_text": text,
            "user_id": row.get("user_id") or row.get("reviewerID") or "",
        }
        if len(reservoir) < n_samples:
            reservoir.append(entry)
        else:
            j = rng.randint(0, len(reservoir))
            if j < n_samples:
                reservoir[j] = entry

    if not reservoir:
        raise ValueError(
            f"No reviews with >= {min_review_tokens} tokens found in {reviews_path!r}."
        )

    if len(reservoir) < n_samples:
        import warnings
        warnings.warn(
            f"Only {len(reservoir)} reviews available (requested {n_samples}).",
            stacklevel=2,
        )

    asins_needed = {e["_parent_asin"] for e in reservoir}
    meta_by_asin: dict[str, dict] = {}
    for row in _iter_jsonl(meta_path):
        asin = _meta_asin(row)
        if asin in asins_needed:
            meta_by_asin[asin] = row

    result = []
    for entry in reservoir:
        asin = entry["_parent_asin"]
        meta = meta_by_asin.get(asin, {})
        result.append(
            {
                "title": _meta_title(meta) or "(unknown)",
                "category": _meta_category(meta, config_category=category),
                "brand": _meta_brand(meta),
                "price": _meta_price(meta),
                "rating": entry["rating"],
                "review_text": entry["review_text"],
                "parent_asin": asin,
                "user_id": entry["user_id"],
            }
        )

    return result
