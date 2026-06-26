"""Merge two GPU shard output directories into single canonical p1_extractions.jsonl.

Usage:
  python scripts/merge_shards.py \\
      --dir_a data/p1_shards_gpu01 \\
      --dir_b data/p1_shards_gpu23 \\
      --output data/processed/Books/p1_extractions.jsonl \\
      --manifest data/processed/Books/p1_extractions.manifest.json

Asserts:
  - merged_rows == sum(rows in each shard file)
  - zero (user_id, item_id) duplicate pairs across all shards
  - user sets of dir_a and dir_b are disjoint
  - is_discriminative=false records are present (canonical completeness)
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_dir(dir_path: str) -> tuple[list[dict], list[str]]:
    """Load all worker_*.jsonl from dir_path. Returns (rows, shard_file_list)."""
    pattern = os.path.join(dir_path, "worker_*.jsonl")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        sys.exit(f"[merge] ERROR: no worker_*.jsonl found in {dir_path}")
    rows = []
    for sf in shard_files:
        with open(sf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows, shard_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", default="data/p1_shards_gpu01",
                        help="Shard dir for GPU0+1 (shard 0_OF_2)")
    parser.add_argument("--dir_b", default="data/p1_shards_gpu23",
                        help="Shard dir for GPU2+3 (shard 1_OF_2)")
    parser.add_argument("--output", default="data/processed/Books/p1_extractions.jsonl")
    parser.add_argument("--manifest", default="data/processed/Books/p1_extractions.manifest.json")
    parser.add_argument("--prompt_version", default="p1_v2")
    parser.add_argument("--cap", type=int, default=20)
    args = parser.parse_args()

    print(f"[merge] Loading dir_a: {args.dir_a}")
    rows_a, files_a = _load_dir(args.dir_a)
    print(f"  {len(files_a)} shard files, {len(rows_a)} rows")

    print(f"[merge] Loading dir_b: {args.dir_b}")
    rows_b, files_b = _load_dir(args.dir_b)
    print(f"  {len(files_b)} shard files, {len(rows_b)} rows")

    # ── Assert 1: row count from each dir is non-zero ──
    if len(rows_a) == 0:
        sys.exit("[merge] ASSERT FAILED: dir_a has 0 rows")
    if len(rows_b) == 0:
        sys.exit("[merge] ASSERT FAILED: dir_b has 0 rows")

    # ── Assert 2: user sets are disjoint ──
    users_a = set(r["user_id"] for r in rows_a)
    users_b = set(r["user_id"] for r in rows_b)
    overlap = users_a & users_b
    if overlap:
        sys.exit(
            f"[merge] ASSERT FAILED: user sets are NOT disjoint. "
            f"{len(overlap)} overlapping users: {list(overlap)[:10]}"
        )
    print(f"[merge] Disjoint OK: {len(users_a)} users in dir_a, {len(users_b)} users in dir_b")

    all_rows = rows_a + rows_b
    expected_total = len(rows_a) + len(rows_b)

    # ── Assert 3: merged rows == Σshard ──
    assert len(all_rows) == expected_total, \
        f"[merge] ASSERT FAILED: merged={len(all_rows)} != expected={expected_total}"
    print(f"[merge] Row count OK: {len(all_rows)} total")

    # ── Assert 4: zero (user_id, item_id) duplicate pairs ──
    uid_item_pairs = [(r["user_id"], r["item_id"]) for r in all_rows]
    n_dupes = len(uid_item_pairs) - len(set(uid_item_pairs))
    if n_dupes:
        sys.exit(f"[merge] ASSERT FAILED: {n_dupes} duplicate (user_id, item_id) pairs found")
    print(f"[merge] Duplicate check OK: 0 duplicate (user_id, item_id) pairs")

    # ── Assert 5: is_discriminative=false records present (canonical completeness) ──
    n_not_discrim = sum(1 for r in all_rows if not r.get("is_discriminative", True))
    if n_not_discrim == 0:
        print("[merge] WARNING: is_discriminative=false records absent — "
              "only eligible records present? Canonical must include all extractions.")
    else:
        print(f"[merge] Canonical completeness OK: {n_not_discrim} is_discriminative=false records present")

    # ── Write canonical ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for row in all_rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    row_count = len(all_rows)
    user_count = len(users_a | users_b)
    canonical_md5 = _file_md5(args.output)

    # ── Write manifest ──
    n_eligible = sum(1 for r in all_rows if r.get("eligible", False))
    n_leakage = sum(1 for r in all_rows if r.get("leakage_detected", False))
    manifest = {
        "row_count": row_count,
        "user_count": user_count,
        "n_eligible": n_eligible,
        "eligible_rate": round(n_eligible / max(row_count, 1), 4),
        "leakage_rate_v1": round(n_leakage / max(row_count, 1), 4),
        "md5": canonical_md5,
        "prompt_version": args.prompt_version,
        "cap": args.cap,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "canonical_path": args.output,
        "sources": {
            "dir_a": args.dir_a,
            "dir_b": args.dir_b,
            "n_rows_a": len(rows_a),
            "n_rows_b": len(rows_b),
            "n_users_a": len(users_a),
            "n_users_b": len(users_b),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"  canonical          : {args.output}")
    print(f"  row_count          : {row_count}")
    print(f"  user_count         : {user_count}")
    print(f"  n_eligible         : {n_eligible}  (rate={manifest['eligible_rate']:.3f})")
    print(f"  leakage_rate_v1    : {manifest['leakage_rate_v1']:.3f}  (recompute with v2 detector)")
    print(f"  md5                : {canonical_md5}")
    print(f"  manifest           : {args.manifest}")
    print(f"\nNext: python scripts/recompute_leakage.py --input {args.output}  # apply v2 detector")


if __name__ == "__main__":
    main()
