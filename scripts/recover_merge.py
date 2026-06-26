"""One-shot recovery merge for corrupt chunk_0001.jsonl.

Reads chunk_0000 (all clean) + chunk_0001 (12 corrupt lines, 1 unrecoverable).
For each corrupt line, finds the second embedded JSON object starting with
{"user_id" and recovers it; the truncated first-half is discarded.
Writes data/memory_full/memory_b_u{N}_seed42.jsonl and reports losses.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CHUNK_0 = Path("data/memory_full/chunk_0000.jsonl")
CHUNK_1 = Path("data/memory_full/chunk_0001.jsonl")
# Partial file from failed merge — delete it before writing new output
PARTIAL = Path("data/memory_full/memory_b_u8924_seed42.jsonl")
OUT_DIR = Path("data/memory_full")


def try_recover(line: str) -> list[dict]:
    """Find all embedded {"user_id":...} objects in a corrupt line."""
    recovered = []
    for m in re.finditer(r'\{"user_id"', line):
        candidate = line[m.start():]
        try:
            obj = json.loads(candidate)
            recovered.append(obj)
        except json.JSONDecodeError:
            pass
    return recovered


def read_chunk(path: Path) -> tuple[list[dict], list[dict], list[int]]:
    """Read chunk JSONL. Returns (valid_rows, recovered_rows, lost_lines)."""
    valid, recovered, lost = [], [], []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                valid.append(json.loads(line))
            except json.JSONDecodeError:
                objs = try_recover(line)
                if objs:
                    recovered.extend(objs)
                    print(f"  [recover] {path.name}:{lineno} → "
                          f"{len(objs)} obj(s) rescued "
                          f"(uid={[o['user_id'] for o in objs]})", flush=True)
                else:
                    lost.append(lineno)
                    print(f"  [lost]    {path.name}:{lineno} — no JSON found", flush=True)
    return valid, recovered, lost


def main() -> None:
    if PARTIAL.exists():
        print(f"Removing incomplete partial file: {PARTIAL}")
        import subprocess
        subprocess.run(["sudo", "rm", str(PARTIAL)], check=True)

    print(f"Reading {CHUNK_0} ...")
    v0, r0, l0 = read_chunk(CHUNK_0)
    print(f"  chunk_0000: valid={len(v0)}  recovered={len(r0)}  lost_lines={len(l0)}")

    print(f"Reading {CHUNK_1} ...")
    v1, r1, l1 = read_chunk(CHUNK_1)
    print(f"  chunk_0001: valid={len(v1)}  recovered={len(r1)}  lost_lines={len(l1)}")

    all_rows = v0 + r0 + v1 + r1
    # Check for duplicate user_ids (shouldn't happen but sanity check)
    seen_uids: set[int] = set()
    deduped = []
    dups = []
    for row in all_rows:
        uid = row["user_id"]
        if uid in seen_uids:
            dups.append(uid)
        else:
            seen_uids.add(uid)
            deduped.append(row)

    if dups:
        print(f"  WARNING: {len(dups)} duplicate user_ids found: {dups[:20]}")

    n = len(deduped)
    out_path = OUT_DIR / f"memory_b_u{n}_seed42.jsonl"
    print(f"\nWriting {out_path} ({n} users) ...")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    total_input = len(v0) + len(r0) + len(v1) + len(r1)
    lost_total = l0 + l1
    print(f"\n=== Recovery Summary ===")
    print(f"  chunk_0000 clean:      {len(v0)}")
    print(f"  chunk_0000 recovered:  {len(r0)}")
    print(f"  chunk_0001 clean:      {len(v1)}")
    print(f"  chunk_0001 recovered:  {len(r1)}")
    print(f"  Total written:         {n}")
    print(f"  Truly lost lines:      {len(lost_total)} (no JSON recoverable)")
    print(f"  First-half truncated:  ~{len(r1)} (one user lost per merged-line recovery)")
    print(f"  Approximate total loss: ~{len(lost_total) + len(r1)} / 8924 users")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
