#!/usr/bin/env python3
"""STEP 1A' Gate Check — cache restoration feasibility.

Proves (or refutes) that item_id ↔ cluster_label mapping can be deterministically
recovered from parsed_success_cache WITHOUT any LLM re-execution.

Gate criteria:
  (1) eligible review cache coverage = 100%
  (2) source_text matching is 1:1 deterministic per user
  (3) item_id restoration rate per user and globally

Stops and reports regardless of outcome. Does NOT rebuild anything.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants (must match the original build exactly)

PROMPT_VERSION = "p1_books_b_2adc8af9"
MODEL_ID = "gemma4:26b"
TEMPERATURE = 0.0
CACHE_DB = PROJECT_ROOT / "data/llm_cache/llm_cache.sqlite"
CANDIDATES_PATH = PROJECT_ROOT / "data/processed/Books/books_memory_candidates.jsonl"
MERGED_PATH = PROJECT_ROOT / "data/memory_full/memory_b_u8924_seed42.jsonl"

# ---------------------------------------------------------------------------
# Exact replicas of build-time logic (from pipeline.py / p1_books.py)

def _category_str(category) -> str:
    if isinstance(category, list):
        return " > ".join(str(c) for c in category)
    return str(category) if category else ""


def _words_head_tail(text: str, head: int, tail: int) -> str:
    words = text.split()
    total = head + tail
    if len(words) <= total:
        return " ".join(words)
    return " ".join(words[:head]) + " [...] " + " ".join(words[-tail:])


def _truncate_review_b(text: str) -> str:
    words = text.split()
    if len(words) <= 400:
        return text
    return _words_head_tail(text, 250, 150)


_USER_TEMPLATE_WITH_DESC = """\
Item title: {title}
Category: {category}
Description: {description}
Rating: {rating}/5
Review: \"\"\"{review_text}\"\"\""""

_USER_TEMPLATE_NO_DESC = """\
Item title: {title}
Category: {category}
Rating: {rating}/5
Review: \"\"\"{review_text}\"\"\""""

# Load system prompt (same path logic as loader.py)
_SYSTEM_B_PATH = PROJECT_ROOT / "config/prompts/amazon/_default/p1_books_b.txt"
with open(_SYSTEM_B_PATH, encoding="utf-8") as _f:
    _SYSTEM_B = _f.read()


def build_messages_b(item: dict) -> list[dict]:
    review_text = _truncate_review_b(item.get("review_text", ""))
    description = (item.get("item_description") or "").strip()
    category = _category_str(item.get("item_category", ""))
    if description:
        user_block = _USER_TEMPLATE_WITH_DESC.format(
            title=item.get("item_title", ""),
            category=category,
            description=description[:400],
            rating=item.get("rating", ""),
            review_text=review_text,
        )
    else:
        user_block = _USER_TEMPLATE_NO_DESC.format(
            title=item.get("item_title", ""),
            category=category,
            rating=item.get("rating", ""),
            review_text=review_text,
        )
    return [
        {"role": "system", "content": _SYSTEM_B},
        {"role": "user", "content": user_block},
    ]


def parsed_cache_key(messages: list[dict]) -> str:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    raw = f"{PROMPT_VERSION}|{MODEL_ID}|{TEMPERATURE}|{serialized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_eligible(p: dict) -> bool:
    if not p.get("is_discriminative"):
        return False
    if p.get("grounding_level") == "metadata_dominant":
        return False
    if not (p.get("contextual_intent") or "").strip():
        return False
    if len(p.get("evidence_span") or []) < 1:
        return False
    return True


def make_source_text(p: dict) -> str:
    ci = (p.get("contextual_intent") or "").strip()
    ps = (p.get("preference_summary") or "").strip()
    parts = [x for x in [ci, ps] if x]
    return " ".join(parts)


_REQUIRED_FIELDS = {"contextual_intent", "preference_summary", "evidence_span",
                    "is_discriminative", "grounding_level"}


def parse_cached_response(raw: str) -> dict | None:
    """Parse cached JSON; returns None on failure."""
    if not raw:
        return None
    try:
        # strip markdown fences if any
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            text = m.group(1)
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not _REQUIRED_FIELDS.issubset(data):
        return None
    if not isinstance(data.get("evidence_span"), list):
        data["evidence_span"] = []
    return data


# ---------------------------------------------------------------------------
# Load chunk source_texts per user

def load_merged_source_texts() -> dict[str, dict]:
    """Returns {user_id: {"cluster_summaries": [...], "source_text_to_label": {str: int}}}"""
    user_data: dict = {}
    with open(MERGED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row["user_id"])
            st_to_label: dict[str, int] = {}
            label_to_sts: dict[int, list[str]] = {}
            for cs in row.get("cluster_summaries", []):
                label = cs["label"]
                label_to_sts[label] = cs["source_texts"]
                for st in cs["source_texts"]:
                    st_to_label[st] = label
            user_data[uid] = {
                "cluster_summaries": row.get("cluster_summaries", []),
                "source_text_to_label": st_to_label,
                "label_to_sts": label_to_sts,
                "k_personal": row.get("k_personal", 0),
                "total_cluster_members": sum(len(v) for v in label_to_sts.values()),
            }
    return user_data


# ---------------------------------------------------------------------------
# Main gate check

def main():
    print("=" * 70, flush=True)
    print("STEP 1A' Gate Check — cache restoration feasibility", flush=True)
    print("=" * 70, flush=True)

    # Load chunk data
    print("\n[1] Loading merged cluster data ...", flush=True)
    user_chunk_data = load_merged_source_texts()
    print(f"    Loaded {len(user_chunk_data)} users from merged file", flush=True)

    # Open cache
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # Load candidates
    print("\n[2] Loading candidates ...", flush=True)
    candidates_by_user: dict[str, list[dict]] = defaultdict(list)
    total_reviews = 0
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uid = str(r["user_id"])
            if uid in user_chunk_data:  # only process users that are in the build
                candidates_by_user[uid].append(r)
                total_reviews += 1
    print(f"    {total_reviews} reviews across {len(candidates_by_user)} users (in-build only)", flush=True)

    # Per-review: reconstruct messages → cache lookup → is_eligible → source_text
    print("\n[3] Running cache lookups ...", flush=True)

    # Stats
    cache_hit_eligible = 0
    cache_miss_eligible = 0  # reviews that ARE eligible (in chunks) but not in cache
    cache_hit_total = 0
    cache_miss_total = 0
    parse_fail = 0

    # Per-user matching
    # For each user: list of (item_id, source_text) for eligible reviews from cache
    user_eligible_from_cache: dict[str, list[tuple]] = defaultdict(list)

    # Track which users have cache misses on eligible reviews
    cache_miss_users: dict[str, list[str]] = defaultdict(list)  # uid → [item_ids]

    processed = 0
    for uid, reviews in candidates_by_user.items():
        chunk_st_to_label = user_chunk_data[uid]["source_text_to_label"]
        # Build set of all source_texts in this user's chunks (ground truth)
        gt_source_texts = set(chunk_st_to_label.keys())

        for item in reviews:
            messages = build_messages_b(item)
            key = parsed_cache_key(messages)
            row = conn.execute(
                "SELECT response FROM parsed_success_cache WHERE key=?", (key,)
            ).fetchone()

            if row is None:
                cache_miss_total += 1
                # Check if this review's source_text would be in the chunks
                # (We can't know without the parsed output — mark as unknown)
            else:
                cache_hit_total += 1
                parsed = parse_cached_response(row[0])
                if parsed is None:
                    parse_fail += 1
                    continue
                if is_eligible(parsed):
                    st = make_source_text(parsed)
                    user_eligible_from_cache[uid].append((str(item["item_id"]), st))
                    if st in gt_source_texts:
                        cache_hit_eligible += 1
                    # else: cache hit but source_text not in chunks (shouldn't happen for build reviews)

        processed += 1
        if processed % 1000 == 0:
            print(f"    ... {processed}/{len(candidates_by_user)} users done", flush=True)

    conn.close()

    print(f"\n[4] Cache stats:", flush=True)
    print(f"    Total reviews in-build    : {total_reviews}", flush=True)
    print(f"    parsed_success_cache hits : {cache_hit_total}", flush=True)
    print(f"    parsed_success_cache miss : {cache_miss_total}", flush=True)
    print(f"    parse fail on hit         : {parse_fail}", flush=True)
    print(f"    Overall cache coverage    : {cache_hit_total/max(total_reviews,1)*100:.2f}%", flush=True)

    # ---------------------------------------------------------------------------
    # Gate (1): eligible review cache coverage
    # We need to verify: for every source_text in chunk cluster_summaries,
    # there exists exactly one cache-reconstructed (item_id, source_text) pair.

    print("\n[5] Source text matching per user ...", flush=True)

    total_cluster_members = 0   # total source_texts in all user chunks
    total_matched = 0           # chunk source_texts matched by cache reconstruction
    total_unmatched_in_chunks = 0  # chunk source_texts NOT matched from cache
    total_extra_in_cache = 0    # cache eligible source_texts NOT in chunks

    # 1:1 determinism
    many_to_one = 0  # multiple cache items → same source_text (same label)
    zero_to_one = 0  # source_text in chunk but no cache item matched

    per_user_restoration: list[tuple[str, int, int, float]] = []  # (uid, matched, total, rate)

    # Users where chunk source_texts are NOT fully covered by cache
    users_with_gap: list[tuple[str, list[str]]] = []

    for uid, chunk_data in user_chunk_data.items():
        st_to_label = chunk_data["source_text_to_label"]
        chunk_sts = set(st_to_label.keys())
        n_chunk = len(chunk_sts)
        total_cluster_members += n_chunk

        # Source texts reconstructed from cache for this user
        cache_items = user_eligible_from_cache.get(uid, [])
        cache_sts = [st for (_, st) in cache_items]
        cache_st_set = set(cache_sts)

        # Count duplicates in cache_sts (same source_text from multiple reviews)
        from collections import Counter
        st_counts = Counter(cache_sts)
        n_dups = sum(c - 1 for c in st_counts.values() if c > 1)
        many_to_one += n_dups

        matched = chunk_sts & cache_st_set
        n_matched = len(matched)
        unmatched = chunk_sts - cache_st_set
        n_unmatched = len(unmatched)
        extra = cache_st_set - chunk_sts
        n_extra = len(extra)

        total_matched += n_matched
        total_unmatched_in_chunks += n_unmatched
        total_extra_in_cache += n_extra
        zero_to_one += n_unmatched

        rate = n_matched / n_chunk if n_chunk > 0 else 1.0
        per_user_restoration.append((uid, n_matched, n_chunk, rate))

        if n_unmatched > 0:
            users_with_gap.append((uid, list(unmatched)))

    print(f"\n{'='*70}", flush=True)
    print(f"GATE REPORT", flush=True)
    print(f"{'='*70}", flush=True)

    overall_coverage = total_matched / max(total_cluster_members, 1) * 100
    print(f"\n(1) Eligible review cache coverage:", flush=True)
    print(f"    Total cluster members (source_texts in chunks) : {total_cluster_members}", flush=True)
    print(f"    Matched from cache                             : {total_matched}", flush=True)
    print(f"    Unmatched (in chunks, NOT in cache)            : {total_unmatched_in_chunks}", flush=True)
    print(f"    Extra (in cache, NOT in chunks)                : {total_extra_in_cache}", flush=True)
    print(f"    COVERAGE                                       : {overall_coverage:.2f}%", flush=True)

    print(f"\n(2) Source text 1:1 matching determinism:", flush=True)
    print(f"    Many-to-one (dup source_texts from different item_ids) : {many_to_one}", flush=True)
    print(f"    Zero-to-one (chunk source_text not in cache)           : {zero_to_one}", flush=True)
    is_deterministic = (many_to_one == 0 and zero_to_one == 0)
    print(f"    1:1 DETERMINISTIC                                      : {is_deterministic}", flush=True)

    print(f"\n(3) item_id restoration rate:", flush=True)
    rates = [r for (_, _, _, r) in per_user_restoration]
    if rates:
        import statistics
        print(f"    Per-user mean  : {statistics.mean(rates)*100:.2f}%", flush=True)
        print(f"    Per-user median: {statistics.median(rates)*100:.2f}%", flush=True)
        n_perfect = sum(1 for r in rates if r == 1.0)
        print(f"    Users 100%     : {n_perfect}/{len(rates)} ({n_perfect/len(rates)*100:.1f}%)", flush=True)
        n_zero = sum(1 for r in rates if r == 0.0)
        print(f"    Users 0%       : {n_zero}/{len(rates)}", flush=True)
    print(f"    Global rate    : {overall_coverage:.2f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
    if overall_coverage >= 100.0 and is_deterministic:
        print(f"GATE: PASS — proceed to STEP 1B (rebuild with evidence_map)", flush=True)
    else:
        print(f"GATE: FAIL — stop here, report gap, proceed to STEP 1C", flush=True)
        if users_with_gap:
            print(f"\nUsers with unmatched cluster source_texts (up to 20 shown):", flush=True)
            for uid, unmatched_sts in users_with_gap[:20]:
                print(f"  uid={uid}  unmatched_count={len(unmatched_sts)}", flush=True)
                for st in unmatched_sts[:2]:
                    print(f"    source_text: {st[:80]!r}...", flush=True)
    print(f"{'='*70}", flush=True)

    # Save detailed per-user restoration stats
    out = {
        "cache_coverage_pct": round(overall_coverage, 4),
        "total_cluster_members": total_cluster_members,
        "total_matched": total_matched,
        "total_unmatched": total_unmatched_in_chunks,
        "total_extra_in_cache": total_extra_in_cache,
        "many_to_one_dups": many_to_one,
        "zero_to_one_gaps": zero_to_one,
        "is_deterministic": is_deterministic,
        "n_users": len(user_chunk_data),
        "n_users_with_gap": len(users_with_gap),
        "users_with_gap_sample": [
            {"uid": uid, "unmatched_count": len(sts), "sample_sts": sts[:3]}
            for uid, sts in users_with_gap[:50]
        ],
        "per_user_restoration_summary": {
            "mean_rate": round(statistics.mean(rates) if rates else 0, 6),
            "median_rate": round(statistics.median(rates) if rates else 0, 6),
            "n_perfect": sum(1 for r in rates if r == 1.0),
            "n_zero": sum(1 for r in rates if r == 0.0),
        },
    }
    out_path = PROJECT_ROOT / "reports/gate_1a_report.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
