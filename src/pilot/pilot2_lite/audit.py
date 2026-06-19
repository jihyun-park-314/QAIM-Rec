"""Task 1: Books composition audit — no LLM.

Streams up to 50k reviews from Books/reviews.jsonl.
Builds meta lookup from Books/meta.jsonl.
Reports: scanned_review_count, unique_user_count, unique_parent_asin_count,
         review length stats, rating distribution, top categories,
         children-related item/review ratio.
Also checks the 20 pilot1_aspect Books parent_asins for children-related ratio.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


_CHILDREN_CATEGORY_KEYWORDS = frozenset([
    "child", "children", "kids", "baby", "toddler", "preschool",
    "infant", "juvenile", "school", "education", "educational",
    "bedtime", "picture book",
])

_CHILDREN_TITLE_KEYWORDS = frozenset([
    "child", "children", "kids", "baby", "toddler", "preschool",
    "infant", "juvenile", "school", "education", "educational",
    "bedtime", "picture book",
])


def _is_children_text(text: str) -> bool:
    """Keyword match on lowercased text. 'picture' alone is NOT matched."""
    t = text.lower()
    for kw in _CHILDREN_CATEGORY_KEYWORDS:
        if kw in t:
            return True
    return False


def _is_children_meta(meta: dict) -> bool:
    """Category-first check. Falls back to title keyword."""
    categories = meta.get("categories") or []
    main_cat = meta.get("main_category") or ""
    cat_strings = [main_cat] + (categories if isinstance(categories, list) else [str(categories)])
    combined_cat = " ".join(str(c) for c in cat_strings)
    if _is_children_text(combined_cat):
        return True
    title = meta.get("title") or ""
    return _is_children_text(title)


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_books_audit(
    data_dir: str = "data/raw",
    max_reviews: int = 50_000,
    pilot1_asins: list[str] | None = None,
) -> dict:
    reviews_path = str(Path(data_dir) / "Books" / "reviews.jsonl")
    meta_path = str(Path(data_dir) / "Books" / "meta.jsonl")

    # --- Pass 1: stream reviews (up to max_reviews) ---
    review_count = 0
    users: set[str] = set()
    asins: set[str] = set()
    ratings: list[float] = []
    token_counts: list[int] = []

    for row in _iter_jsonl(reviews_path):
        if review_count >= max_reviews:
            break
        text = row.get("text") or row.get("review_text") or ""
        uid = row.get("user_id") or ""
        pa = row.get("parent_asin") or ""
        rating = row.get("rating") or 0.0
        if uid:
            users.add(uid)
        if pa:
            asins.add(pa)
        tc = len(text.split())
        token_counts.append(tc)
        ratings.append(float(rating))
        review_count += 1

    # --- Pass 2: build meta lookup for scanned asins + pilot1 asins ---
    all_asins_needed = set(asins)
    if pilot1_asins:
        all_asins_needed.update(pilot1_asins)

    meta_lookup: dict[str, dict] = {}
    cat_counter: dict[str, int] = {}

    for row in _iter_jsonl(meta_path):
        pa = row.get("parent_asin") or row.get("asin") or ""
        if pa in all_asins_needed:
            meta_lookup[pa] = row
        cats = row.get("categories") or []
        if isinstance(cats, list) and len(cats) > 1:
            sub = cats[1] if len(cats) > 1 else ""
            if sub:
                cat_counter[sub] = cat_counter.get(sub, 0) + 1

    # --- Children-related stats on scanned reviews ---
    # Re-stream to count children-related reviews (we already know asins)
    children_asins = {a for a, m in meta_lookup.items() if a in asins and _is_children_meta(m)}
    # For asins with no meta: fallback check not possible here (title unknown)
    children_review_count = 0
    total_rechecked = 0
    for row in _iter_jsonl(reviews_path):
        if total_rechecked >= max_reviews:
            break
        pa = row.get("parent_asin") or ""
        if pa in children_asins:
            children_review_count += 1
        total_rechecked += 1

    n_children_asins = len(children_asins)
    n_asins_with_meta = sum(1 for a in asins if a in meta_lookup)

    # --- Pilot1 asins children check ---
    pilot1_children_ratio = None
    if pilot1_asins:
        n_p1_children = sum(
            1 for a in pilot1_asins
            if a in meta_lookup and _is_children_meta(meta_lookup[a])
        )
        pilot1_children_ratio = round(n_p1_children / len(pilot1_asins), 4)

    # --- Rating distribution ---
    rating_dist: dict[str, int] = {}
    for r in ratings:
        k = str(int(r)) if r == int(r) else str(r)
        rating_dist[k] = rating_dist.get(k, 0) + 1

    # --- Review length stats ---
    def _pct(data: list[float], p: float) -> float:
        data_s = sorted(data)
        idx = int(len(data_s) * p)
        return round(data_s[min(idx, len(data_s) - 1)], 1)

    # --- Top subcategories ---
    top_cats = sorted(cat_counter.items(), key=lambda x: -x[1])[:20]

    return {
        "scanned_review_count": review_count,
        "unique_user_count": len(users),
        "unique_parent_asin_count": len(asins),
        "asins_with_meta": n_asins_with_meta,
        "review_length_tokens": {
            "median": _pct(token_counts, 0.50),
            "p25": _pct(token_counts, 0.25),
            "p75": _pct(token_counts, 0.75),
            "mean": round(statistics.mean(token_counts), 1),
        },
        "rating_distribution": rating_dist,
        "top_subcategories": [{"category": c, "count": n} for c, n in top_cats],
        "children_related": {
            "children_asins_count": n_children_asins,
            "children_asins_ratio_among_scanned": round(n_children_asins / max(len(asins), 1), 4),
            "children_review_count": children_review_count,
            "children_review_ratio": round(children_review_count / max(review_count, 1), 4),
        },
        "pilot1_aspect_asins_children_check": {
            "pilot1_asins_count": len(pilot1_asins) if pilot1_asins else 0,
            "children_ratio": pilot1_children_ratio,
            "note": "Children ratio among Books pilot1_aspect N=20 sample",
        },
    }
