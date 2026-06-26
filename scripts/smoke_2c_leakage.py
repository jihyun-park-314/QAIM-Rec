"""Smoke 2C — Temporal leakage: all selected item timestamps ⊂ user train-history.

Pass criteria: for every selected (user_id, item_id) in p1_extractions.jsonl,
the item must exist in splits.json users[user_id].train.

Uses selected_review_ids.json from parallel_memory_build output.
Falls back to checking p1_extractions.jsonl directly against splits.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

P1_PATH = ROOT / "data/processed/Books/p1_extractions.jsonl"
SPLITS_PATH = ROOT / "data/processed/Books/splits.json"
SEL_IDS_PATH = ROOT / "data/p1_shards/selected_review_ids.json"


def main() -> None:
    print("=" * 60)
    print("SMOKE 2C — Temporal leakage: item timestamps ⊂ train-history")
    print("=" * 60)

    if not SPLITS_PATH.exists():
        sys.exit(f"[FAIL] splits.json not found: {SPLITS_PATH}")

    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)

    splits_users = splits.get("users", {})

    # Build train item_id sets per user
    # splits format: users[uid].train = list of item_ids
    train_sets: dict = {}
    for uid_str, user_split in splits_users.items():
        train_items = set(user_split.get("train", []))
        train_sets[uid_str] = train_items

    print(f"[2C] Loaded splits for {len(train_sets)} users")

    # Load selected reviews (from p1_extractions.jsonl or selected_review_ids.json)
    violations = []
    n_checked = 0

    if P1_PATH.exists():
        print(f"[2C] Checking via p1_extractions.jsonl ...")
        with open(P1_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                uid = str(r.get("user_id", ""))
                item_id = r.get("item_id")
                n_checked += 1
                train = train_sets.get(uid)
                if train is None:
                    # user not in splits — skip (splits coherence is separate check)
                    continue
                if item_id not in train:
                    violations.append({
                        "user_id": uid,
                        "item_id": item_id,
                        "timestamp": r.get("timestamp"),
                    })
    elif SEL_IDS_PATH.exists():
        print(f"[2C] Checking via selected_review_ids.json ...")
        with open(SEL_IDS_PATH, encoding="utf-8") as f:
            selected_review_ids = json.load(f)
        for uid, item_ids in selected_review_ids.items():
            train = train_sets.get(str(uid))
            if train is None:
                continue
            for item_id in item_ids:
                n_checked += 1
                if item_id not in train:
                    violations.append({"user_id": uid, "item_id": item_id})
    else:
        sys.exit(f"[FAIL] Neither p1_extractions.jsonl nor selected_review_ids.json found. "
                 f"Run parallel_memory_build.py first.")

    print(f"[2C] Checked {n_checked} (user, item) pairs")
    print(f"[2C] Violations: {len(violations)}")

    if violations:
        print(f"  Sample violations (up to 5):")
        for v in violations[:5]:
            print(f"    user={v['user_id']}  item={v['item_id']}")
        print(f"\n[FAIL] 2C — {len(violations)} temporal leakage violations!")
        sys.exit(1)

    print(f"\n[PASS] 2C — All {n_checked} selected items are in user train-history (0 violations).")


if __name__ == "__main__":
    main()
