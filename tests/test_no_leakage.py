"""A7: No-leakage tests.

Verifies that:
1. val/test items are not in the train set for each user (temporal integrity)
2. Memory-eligible interactions are strictly in train history (not val/test targets)
3. sasrec.txt only contains interactions present in splits.json

Run: pytest tests/test_no_leakage.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

CATEGORIES = ["Books", "Beauty_and_Personal_Care"]
DATA_DIR = "data/processed"


def _load_splits(category: str) -> dict:
    path = os.path.join(DATA_DIR, category, "splits.json")
    if not os.path.exists(path):
        pytest.skip(f"splits.json not found for {category}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_sequences(category: str) -> list[dict]:
    path = os.path.join(DATA_DIR, category, "sequences.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"sequences.jsonl not found for {category}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.mark.parametrize("category", CATEGORIES)
def test_eligible_interactions_in_train_only(category):
    """All is_eligible=True interactions must be in the train split (not val/test target)."""
    splits = _load_splits(category)
    sequences = _load_sequences(category)

    violations = []
    for rec in sequences:
        uid_str = str(rec["user_id"])
        split = splits["users"].get(uid_str)
        if split is None:
            violations.append(f"user {uid_str} not found in splits")
            continue

        val_id = split["val"]
        test_id = split["test"]

        for it in rec["items"]:
            if not it.get("is_eligible"):
                continue
            item_id = it["item_id"]
            # Eligible must not be val or test target
            if item_id == val_id:
                violations.append(
                    f"user {uid_str}: eligible item {item_id} is val target — LEAKAGE"
                )
            if item_id == test_id:
                violations.append(
                    f"user {uid_str}: eligible item {item_id} is test target — LEAKAGE"
                )

    assert not violations, f"{category}: {len(violations)} eligibility leakage violations:\n" + "\n".join(violations[:15])


@pytest.mark.parametrize("category", CATEGORIES)
def test_val_test_temporal_order(category):
    """For each user: timestamp of test target ≥ timestamp of val target.

    Verifies that LOO split respects temporal ordering.
    """
    sequences = _load_sequences(category)
    splits = _load_splits(category)

    violations = []
    for rec in sequences:
        uid_str = str(rec["user_id"])
        split = splits["users"].get(uid_str)
        if split is None:
            continue
        items_by_id = {it["item_id"]: it for it in rec["items"]}

        val_item = items_by_id.get(split["val"])
        test_item = items_by_id.get(split["test"])
        if val_item is None or test_item is None:
            continue
        if test_item["ts"] < val_item["ts"]:
            violations.append(
                f"user {uid_str}: test_ts={test_item['ts']} < val_ts={val_item['ts']}"
            )

    assert not violations, f"{category}: {len(violations)} temporal order violations:\n" + "\n".join(violations[:10])


@pytest.mark.parametrize("category", CATEGORIES)
def test_sasrec_txt_interactions_in_splits(category):
    """Every (user_id, item_id) in sasrec.txt must appear in splits.json train/val/test."""
    sasrec_path = os.path.join(DATA_DIR, category, "sasrec.txt")
    if not os.path.exists(sasrec_path):
        pytest.skip(f"sasrec.txt not found for {category}")

    splits = _load_splits(category)

    # Build set of all (user_id, item_id) from splits
    valid_pairs: set[tuple[int, int]] = set()
    for uid_str, split in splits["users"].items():
        uid = int(uid_str)
        for iid in split["train"]:
            valid_pairs.add((uid, iid))
        valid_pairs.add((uid, split["val"]))
        valid_pairs.add((uid, split["test"]))

    violations = []
    with open(sasrec_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                violations.append(f"line {lineno}: malformed '{line}'")
                continue
            uid, iid = int(parts[0]), int(parts[1])
            if (uid, iid) not in valid_pairs:
                violations.append(f"line {lineno}: ({uid},{iid}) not in splits")
                if len(violations) > 20:
                    break

    assert not violations, f"{category}: {len(violations)} sasrec.txt entries not in splits:\n" + "\n".join(violations[:10])


@pytest.mark.parametrize("category", CATEGORIES)
def test_no_future_items_in_train(category):
    """Train items must all come before val/test in each user's sequence (position check)."""
    sequences = _load_sequences(category)
    splits = _load_splits(category)

    violations = []
    for rec in sequences:
        uid_str = str(rec["user_id"])
        split = splits["users"].get(uid_str)
        if split is None:
            continue

        items = rec["items"]
        val_pos = next((i for i, it in enumerate(items) if it["item_id"] == split["val"]), None)
        test_pos = next((i for i, it in enumerate(items) if it["item_id"] == split["test"]), None)

        if val_pos is None or test_pos is None:
            continue

        # val should be second-to-last, test should be last
        n = len(items)
        if val_pos != n - 2:
            violations.append(f"user {uid_str}: val at position {val_pos}, expected {n-2}")
        if test_pos != n - 1:
            violations.append(f"user {uid_str}: test at position {test_pos}, expected {n-1}")

    assert not violations, f"{category}: {len(violations)} position violations:\n" + "\n".join(violations[:10])
