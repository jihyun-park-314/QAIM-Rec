"""Smoke 2D — Merge integrity: 2 workers → row_count, dup-free, manifest md5.

Pass criteria:
  1. p1_extractions.jsonl row_count == Σ shard row counts.
  2. No duplicate (user_id, item_id) pairs.
  3. manifest md5 == actual file md5.
  4. manifest row_count/user_count match file.

Runs a mini 2-worker extraction (5 users each) to produce shard + merge artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

P1_PATH = ROOT / "data/processed/Books/p1_extractions.jsonl"
MANIFEST_PATH = ROOT / "data/processed/Books/p1_extractions.manifest.json"
SHARD_DIR = "data/p1_shards_smoke2d"


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    print("=" * 60)
    print("SMOKE 2D — Merge integrity (2 workers)")
    print("=" * 60)

    # Run mini 2-worker extraction
    cmd = (
        f"python3 scripts/parallel_memory_build.py "
        f"--category Books --variant B "
        f"--n_workers 2 --max_users_per_worker 5 "
        f"--output_dir {SHARD_DIR} "
        f"--report_dir {SHARD_DIR}"
    )
    print(f"[2D] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(f"[FAIL] Mini extraction failed (exit {ret})")

    # Check shards exist
    shard_paths = sorted(Path(SHARD_DIR).glob("worker_*.jsonl"))
    if not shard_paths:
        sys.exit(f"[FAIL] No shard files found in {SHARD_DIR}")
    print(f"[2D] Found {len(shard_paths)} shards: {[s.name for s in shard_paths]}")

    # Count rows per shard
    shard_row_counts = {str(sp): count_jsonl(str(sp)) for sp in shard_paths}
    total_shard_rows = sum(shard_row_counts.values())
    print(f"[2D] Shard row counts: {shard_row_counts}")
    print(f"[2D] Total shard rows: {total_shard_rows}")

    # Check canonical file
    if not P1_PATH.exists():
        sys.exit(f"[FAIL] p1_extractions.jsonl not found: {P1_PATH}")

    canonical_rows = count_jsonl(str(P1_PATH))
    print(f"[2D] Canonical rows: {canonical_rows}")

    # Check 1: row_count == Σ shard rows
    if canonical_rows != total_shard_rows:
        print(f"[FAIL] Row count mismatch: canonical={canonical_rows} != shards={total_shard_rows}")
        sys.exit(1)
    print(f"[2D] Row count OK: {canonical_rows} == Σ shard rows ✓")

    # Check 2: no duplicate (user_id, item_id)
    with open(P1_PATH, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    pair_counts = Counter((r["user_id"], r["item_id"]) for r in records)
    dupes = {k: v for k, v in pair_counts.items() if v > 1}
    if dupes:
        print(f"[FAIL] {len(dupes)} duplicate (user_id, item_id) pairs!")
        for pair, count in list(dupes.items())[:5]:
            print(f"  {pair}: {count}")
        sys.exit(1)
    print(f"[2D] No duplicates: {len(pair_counts)} unique (user, item) pairs ✓")

    # Check 3: manifest md5 matches actual file
    if not MANIFEST_PATH.exists():
        sys.exit(f"[FAIL] Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    actual_md5 = file_md5(str(P1_PATH))
    manifest_md5 = manifest.get("md5", "")
    if actual_md5 != manifest_md5:
        print(f"[FAIL] MD5 mismatch: manifest={manifest_md5} actual={actual_md5}")
        sys.exit(1)
    print(f"[2D] Manifest MD5 OK: {actual_md5} ✓")

    # Check 4: manifest row_count/user_count match
    manifest_rows = manifest.get("row_count", -1)
    manifest_users = manifest.get("user_count", -1)
    actual_users = len(set(r["user_id"] for r in records))

    if manifest_rows != canonical_rows:
        print(f"[FAIL] Manifest row_count={manifest_rows} != actual={canonical_rows}")
        sys.exit(1)
    if manifest_users != actual_users:
        print(f"[FAIL] Manifest user_count={manifest_users} != actual={actual_users}")
        sys.exit(1)
    print(f"[2D] Manifest counts OK: rows={manifest_rows}, users={manifest_users} ✓")

    print(f"\n[PASS] 2D — rows={canonical_rows}, no dupes, manifest md5 consistent.")


if __name__ == "__main__":
    main()
