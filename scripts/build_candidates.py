"""
Build candidates.jsonl for Books (pre-P1 filter, frozen artifact).

Filter: rating>=4 AND n_words>=10 AND item in train-history (from splits.json).
Selection: timestamp ascending (train-history chronological order).
is_discriminative is P1 output — NOT applied here.

Output:
  data/processed/Books/candidates.jsonl
  data/processed/Books/candidates.manifest.json
"""

import json
import hashlib
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

CATEGORY = "Books"
BASE = f"data/processed/{CATEGORY}"
OUT_FILE = f"{BASE}/candidates.jsonl"
MANIFEST_FILE = f"{BASE}/candidates.manifest.json"

SPLITS_PATH = f"{BASE}/splits.json"
SEQUENCES_PATH = f"{BASE}/sequences.jsonl"
RAW_REVIEWS_PATH = f"data/raw/{CATEGORY}/reviews.jsonl"
RAW_META_PATH = f"data/raw/{CATEGORY}/meta.jsonl"
ID_MAPS_PATH = f"{BASE}/id_maps.json"

SPLITS_MD5_EXPECTED = "d2762a011a6801d7aa7d70fe65f32957"

RATING_MIN = 4.0
WORDS_MIN = 10


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_str(s: bytes):
    return hashlib.md5(s).hexdigest()


def main():
    # --- STEP 0: splits.json MD5 guard ---
    actual_md5 = md5_file(SPLITS_PATH)
    if actual_md5 != SPLITS_MD5_EXPECTED:
        sys.exit(f"ABORT: splits.json MD5 mismatch. expected={SPLITS_MD5_EXPECTED} actual={actual_md5}")
    print(f"[0] splits.json MD5 OK: {actual_md5}")

    with open(SPLITS_PATH) as f:
        splits_raw = json.load(f)
    splits = splits_raw["users"]  # {str(user_id): {train:[item_ids], val:int, test:int}}

    # Build per-user train-item sets (internal integer IDs)
    user_train_items: dict[str, set] = {}
    for uid, entry in splits.items():
        user_train_items[uid] = set(entry["train"])

    print(f"[0] Users in splits: {len(user_train_items):,}")

    # --- Load id_maps ---
    with open(ID_MAPS_PATH) as f:
        id_maps = json.load(f)
    id2item = id_maps["id2item"]   # str(item_id) -> orig_item_id (ASIN)
    id2user = id_maps["id2user"]   # str(user_id) -> orig_user_id
    item2id = id_maps["item2id"]   # orig_item_id -> int item_id
    user2id = id_maps["user2id"]   # orig_user_id -> int user_id

    # Build ASIN -> item_title from raw meta
    print("[1] Loading item meta...")
    asin2title: dict[str, str] = {}
    with open(RAW_META_PATH) as f:
        for line in f:
            m = json.loads(line)
            asin = m.get("parent_asin") or m.get("asin", "")
            title = m.get("title", "")
            if asin:
                asin2title[asin] = title
    print(f"    meta titles loaded: {len(asin2title):,}")

    # --- Build sequences lookup: (user_id, item_id) -> {ts, rating, n_words} from sequences.jsonl ---
    # We use this for n_words (word count already computed) and rating/ts alignment.
    # The raw reviews have the text, but we need to match them to internal IDs.
    print("[2] Loading sequences metadata...")
    seq_meta: dict[tuple, dict] = {}  # (int_user_id, int_item_id) -> {ts, rating, n_words}
    with open(SEQUENCES_PATH) as f:
        for line in f:
            r = json.loads(line)
            uid = r["user_id"]
            for item in r["items"]:
                if item.get("has_review"):
                    seq_meta[(uid, item["item_id"])] = {
                        "ts": item["ts"],
                        "rating": item["rating"],
                        "n_words": item.get("n_words", 0),
                    }
    print(f"    seq_meta entries: {len(seq_meta):,}")

    # --- Load raw reviews and index by (orig_user_id, orig_item_id) ---
    # We need review text. Map to internal IDs on the fly.
    print("[3] Loading raw reviews...")
    # Index: orig_user_id -> list of {orig_item_id, rating, ts, text}
    raw_review_index: dict[str, list] = defaultdict(list)
    with open(RAW_REVIEWS_PATH) as f:
        for line in f:
            r = json.loads(line)
            orig_uid = r.get("user_id", "")
            # Use parent_asin if available, else asin
            orig_iid = r.get("parent_asin") or r.get("asin", "")
            text = r.get("text", "")
            rating = float(r.get("rating", 0))
            ts = r.get("timestamp", 0)
            if orig_uid and orig_iid and text:
                raw_review_index[orig_uid].append({
                    "orig_item_id": orig_iid,
                    "rating": rating,
                    "ts": ts,
                    "text": text,
                })
    print(f"    raw review users indexed: {len(raw_review_index):,}")

    # --- STEP 1: Build candidates ---
    print("[4] Building candidates...")

    # Guard: abort if output already exists (frozen artifact)
    if os.path.exists(OUT_FILE):
        print(f"WARNING: {OUT_FILE} already exists. Verifying existing manifest...")
        if os.path.exists(MANIFEST_FILE):
            with open(MANIFEST_FILE) as f:
                manifest = json.load(f)
            existing_md5 = md5_file(OUT_FILE)
            if existing_md5 == manifest.get("md5"):
                print(f"  Existing candidates.jsonl is valid (md5={existing_md5}). Skipping regeneration.")
                # Proceed to step 2 reporting only
                with open(OUT_FILE) as f:
                    rows = [json.loads(l) for l in f]
                report_distribution(rows)
                return
            else:
                sys.exit("ABORT: candidates.jsonl exists but MD5 mismatches manifest. Manual inspection required.")
        else:
            sys.exit("ABORT: candidates.jsonl exists without manifest. Manual inspection required.")

    rows = []
    skipped_no_text = 0
    skipped_filter = 0

    for uid_str, entry in splits.items():
        int_uid = int(uid_str)
        orig_uid = id2user.get(uid_str, "")
        if not orig_uid:
            continue

        train_item_set = user_train_items[uid_str]
        user_reviews = raw_review_index.get(orig_uid, [])

        # Build lookup: orig_item_id -> review data for this user
        review_by_asin: dict[str, dict] = {}
        for rv in user_reviews:
            review_by_asin[rv["orig_item_id"]] = rv

        # Iterate train items in timestamp ascending order (use seq_meta for ts)
        train_candidates = []
        for item_id in train_item_set:
            orig_iid = id2item.get(str(item_id), "")
            if not orig_iid:
                continue

            meta_key = (int_uid, item_id)
            smeta = seq_meta.get(meta_key)
            if smeta is None:
                continue  # no review in sequences

            rating = smeta["rating"]
            n_words = smeta["n_words"]

            # Pre-P1 filter: rating>=4 AND n_words>=10
            if rating < RATING_MIN or n_words < WORDS_MIN:
                skipped_filter += 1
                continue

            # Get review text from raw
            rv = review_by_asin.get(orig_iid)
            if rv is None:
                skipped_no_text += 1
                continue

            item_title = asin2title.get(orig_iid, "")
            train_candidates.append({
                "user_id": int_uid,
                "item_id": item_id,
                "timestamp": smeta["ts"],
                "rating": rating,
                "review_text": rv["text"],
                "item_title": item_title,
            })

        # Sort by timestamp ascending (plan: train-history chronological)
        train_candidates.sort(key=lambda x: (x["timestamp"], x["item_id"]))
        rows.extend(train_candidates)

    print(f"    Rows collected: {len(rows):,}")
    print(f"    Skipped (filter): {skipped_filter:,}")
    print(f"    Skipped (no text): {skipped_no_text:,}")

    # Write candidates.jsonl
    print("[5] Writing candidates.jsonl...")
    content_bytes = b""
    with open(OUT_FILE, "w") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            f.write(line + "\n")
            content_bytes += (line + "\n").encode("utf-8")

    file_md5 = md5_file(OUT_FILE)
    n_users = len(set(r["user_id"] for r in rows))

    manifest = {
        "n_rows": len(rows),
        "n_users": n_users,
        "md5": file_md5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filter": f"rating>={RATING_MIN} AND n_words>={WORDS_MIN} AND in_train_history",
        "sort": "timestamp_asc, item_id_tiebreak",
        "splits_md5": SPLITS_MD5_EXPECTED,
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[5] candidates.jsonl written: {len(rows):,} rows, {n_users:,} users, md5={file_md5}")
    print(f"[5] manifest written: {MANIFEST_FILE}")

    report_distribution(rows)


def report_distribution(rows):
    """STEP 2: eligible distribution report."""
    import statistics

    print("\n" + "="*60)
    print("STEP 2 — Eligible distribution report")
    print("="*60)

    from collections import Counter
    user_counts: dict[int, int] = Counter()
    for r in rows:
        user_counts[r["user_id"]] += 1

    counts = sorted(user_counts.values())
    n = len(counts)
    total_users_in_splits = 10566  # from splits meta

    def pct(p):
        idx = min(int(p / 100 * n), n - 1)
        return counts[idx]

    mean_val = statistics.mean(counts) if counts else 0
    print(f"\nUsers with ≥1 eligible review: {n:,} / {total_users_in_splits:,} ({100*n/total_users_in_splits:.1f}%)")
    print(f"Total eligible rows: {len(rows):,}")
    print(f"\nEligible per user distribution:")
    print(f"  mean : {mean_val:.2f}")
    print(f"  p50  : {pct(50)}")
    print(f"  p75  : {pct(75)}")
    print(f"  p90  : {pct(90)}")
    print(f"  p95  : {pct(95)}")
    print(f"  p99  : {pct(99)}")
    print(f"  max  : {counts[-1]}")

    print(f"\nCap analysis:")
    print(f"{'cap':>5} | {'capped_users':>15} | {'capped_ratio':>12} | {'dropped_reviews':>15} | {'total_calls(P1)':>16}")
    print("-" * 70)
    for cap in [12, 20, 25, 30]:
        capped = sum(1 for c in counts if c > cap)
        dropped = sum(max(c - cap, 0) for c in counts)
        total_calls = sum(min(c, cap) for c in counts)
        # Include users with 0 eligible (no calls needed)
        zero_users = total_users_in_splits - n
        print(f"{cap:>5} | {capped:>10,} ({100*capped/n:4.1f}%) | {capped/n:>11.3f} | {dropped:>15,} | {total_calls:>16,}")

    # Top 10% eligible users
    top10_threshold_idx = int(0.9 * n)
    top10_counts = counts[top10_threshold_idx:]
    print(f"\nTop 10% eligible users (n={len(top10_counts):,}, threshold eligible≥{counts[top10_threshold_idx]}):")
    if top10_counts:
        print(f"  mean : {statistics.mean(top10_counts):.2f}")
        print(f"  p50  : {statistics.median(top10_counts):.1f}")
        print(f"  p90  : {top10_counts[int(0.9*len(top10_counts))]}")
        print(f"  max  : {top10_counts[-1]}")
        print(f"  (These users are K_personal≥2 candidates — actual K confirmed post-F3 clustering)")


if __name__ == "__main__":
    main()
