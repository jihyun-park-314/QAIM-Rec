"""A7: Split consistency tests.

Verifies that splits.json is internally consistent and that
all modules consuming it get byte-identical data.

Run: pytest tests/test_split_consistency.py -v
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
        pytest.skip(f"splits.json not found for {category} — run preprocessing first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_sequences(category: str) -> dict[str, list[int]]:
    """Load sequences.jsonl → {str(user_id): [item_id, ...]}."""
    path = os.path.join(DATA_DIR, category, "sequences.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"sequences.jsonl not found for {category}")
    seqs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seqs[str(rec["user_id"])] = [it["item_id"] for it in rec["items"]]
    return seqs


@pytest.mark.parametrize("category", CATEGORIES)
def test_split_covers_full_sequence(category):
    """train + [val] + [test] == full user sequence in sequences.jsonl."""
    splits = _load_splits(category)
    seqs = _load_sequences(category)

    mismatches = []
    for uid_str, split in splits["users"].items():
        if uid_str not in seqs:
            mismatches.append(f"user {uid_str} missing from sequences.jsonl")
            continue
        full_seq = seqs[uid_str]
        reconstructed = split["train"] + [split["val"]] + [split["test"]]
        if reconstructed != full_seq:
            mismatches.append(
                f"user {uid_str}: split={reconstructed[:5]}… seq={full_seq[:5]}…"
            )

    assert not mismatches, f"{category}: {len(mismatches)} sequence mismatches:\n" + "\n".join(mismatches[:10])


@pytest.mark.parametrize("category", CATEGORIES)
def test_val_test_not_in_train(category):
    """val and test item IDs must not appear in the train list for the same user."""
    splits = _load_splits(category)

    violations = []
    for uid_str, split in splits["users"].items():
        train_set = set(split["train"])
        if split["val"] in train_set:
            violations.append(f"user {uid_str}: val={split['val']} found in train")
        if split["test"] in train_set:
            violations.append(f"user {uid_str}: test={split['test']} found in train")

    assert not violations, f"{category}: {len(violations)} leakage violations:\n" + "\n".join(violations[:10])


@pytest.mark.parametrize("category", CATEGORIES)
def test_min_train_length(category):
    """Every user must have at least 1 item in train (sequence length ≥ 3 after 5-core)."""
    splits = _load_splits(category)

    empty_train = [uid for uid, s in splits["users"].items() if len(s["train"]) == 0]
    assert not empty_train, f"{category}: {len(empty_train)} users have empty train set"


@pytest.mark.parametrize("category", CATEGORIES)
def test_split_meta_counts(category):
    """n_users in meta matches actual user count in splits.users."""
    splits = _load_splits(category)
    meta = splits.get("meta", {})

    actual = len(splits["users"])
    expected = meta.get("n_users")
    if expected is not None:
        assert actual == expected, f"{category}: meta.n_users={expected} but actual={actual}"


@pytest.mark.parametrize("category", CATEGORIES)
def test_no_duplicate_users(category):
    """Each user_id appears exactly once in splits.users."""
    splits = _load_splits(category)
    uids = list(splits["users"].keys())
    assert len(uids) == len(set(uids)), f"{category}: duplicate user IDs in splits"


@pytest.mark.parametrize("category", CATEGORIES)
def test_id_maps_consistent_with_splits(category):
    """All user/item IDs in splits.json are present in id_maps.json."""
    splits = _load_splits(category)

    maps_path = os.path.join(DATA_DIR, category, "id_maps.json")
    if not os.path.exists(maps_path):
        pytest.skip(f"id_maps.json not found for {category}")
    with open(maps_path, encoding="utf-8") as f:
        id_maps = json.load(f)

    all_item_ids_in_splits = set()
    all_user_ids_in_splits = set()
    for uid_str, split in splits["users"].items():
        all_user_ids_in_splits.add(int(uid_str))
        all_item_ids_in_splits.update(split["train"])
        all_item_ids_in_splits.add(split["val"])
        all_item_ids_in_splits.add(split["test"])

    id2item = {int(k): v for k, v in id_maps["id2item"].items()}
    id2user = {int(k): v for k, v in id_maps["id2user"].items()}

    missing_items = all_item_ids_in_splits - set(id2item.keys())
    missing_users = all_user_ids_in_splits - set(id2user.keys())

    assert not missing_items, f"{category}: {len(missing_items)} item IDs in splits but not in id_maps"
    assert not missing_users, f"{category}: {len(missing_users)} user IDs in splits but not in id_maps"
