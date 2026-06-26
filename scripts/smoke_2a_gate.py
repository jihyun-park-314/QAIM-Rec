"""Smoke 2A — Determinism GATE: shuffled candidates → identical selected_review_ids.

Pass criteria: selected_review_ids from shuffled input == original (bit-for-bit).
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from parallel_memory_build import (
    CANDIDATES_MD5,
    verify_md5,
    select_and_split,
)

CANDIDATES_PATH = "data/processed/Books/candidates.jsonl"
N_WORKERS = 4
CAP = 20
ELIGIBLE_MIN = 2
SEED = 42
N_USERS = 200  # small sample for smoke


def main() -> None:
    print("=" * 60)
    print("SMOKE 2A — Determinism GATE (shuffle invariant)")
    print("=" * 60)

    if not os.path.isfile(CANDIDATES_PATH):
        sys.exit(f"[FAIL] candidates.jsonl not found: {CANDIDATES_PATH}")

    verify_md5(CANDIDATES_PATH, CANDIDATES_MD5)

    # Original selection
    _, _, sel_orig = select_and_split(
        CANDIDATES_PATH, ELIGIBLE_MIN, CAP, N_USERS, N_WORKERS, SEED
    )
    print(f"[2A] Original: {len(sel_orig)} users selected")

    # Load lines and shuffle
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    rng = random.Random(999)
    shuffled_lines = lines[:]
    rng.shuffle(shuffled_lines)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.writelines(shuffled_lines)
        tmp_path = tmp.name

    try:
        _, _, sel_shuffled = select_and_split(
            tmp_path, ELIGIBLE_MIN, CAP, N_USERS, N_WORKERS, SEED
        )
    finally:
        os.unlink(tmp_path)

    print(f"[2A] Shuffled: {len(sel_shuffled)} users selected")

    # Compare
    users_orig = sorted(sel_orig.keys(), key=str)
    users_shuffled = sorted(sel_shuffled.keys(), key=str)

    if users_orig != users_shuffled:
        print(f"[GATE FAIL] User sets differ!")
        print(f"  orig={users_orig[:5]}  shuffled={users_shuffled[:5]}")
        sys.exit(1)

    n_mismatch = 0
    for uid in users_orig:
        if sel_orig[uid] != sel_shuffled[uid]:
            n_mismatch += 1
            if n_mismatch <= 3:
                print(f"  [mismatch] uid={uid}")
                print(f"    orig    : {sel_orig[uid][:5]}")
                print(f"    shuffled: {sel_shuffled[uid][:5]}")

    if n_mismatch > 0:
        print(f"[GATE FAIL] {n_mismatch}/{len(users_orig)} users have different selected_review_ids!")
        sys.exit(1)

    print(f"\n[PASS] 2A — {len(users_orig)} users: selected_review_ids bit-identical after shuffle.")


if __name__ == "__main__":
    main()
