"""SASRec data loader using our sasrec.txt + splits.json format.

Replaces pmixer's data_partition() + WarpSampler with a version that:
- Reads sasrec.txt (our format: "user_id item_id\\n" per interaction, time-sorted)
- Uses splits.json as the single source of truth for train/val/test
- Produces identical sample format (uid, seq, pos, neg) for BPR training
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np


def load_data(category: str, data_dir: str = "data/processed") -> dict:
    """Load sasrec.txt and splits.json → unified data structure.

    Returns:
        {
          "user_train": {uid: [item_ids...]},  # train items only (time-sorted)
          "user_valid": {uid: [val_item_id]},
          "user_test": {uid: [test_item_id]},
          "usernum": int,
          "itemnum": int,
        }
    """
    splits_path = os.path.join(data_dir, category, "splits.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"splits.json not found: {splits_path!r}. Run preprocessing first."
        )

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    user_train: dict[int, list[int]] = {}
    user_valid: dict[int, list[int]] = {}
    user_test: dict[int, list[int]] = {}

    for uid_str, split in splits["users"].items():
        uid = int(uid_str)
        user_train[uid] = split["train"]
        user_valid[uid] = [split["val"]]
        user_test[uid] = [split["test"]]

    meta = splits.get("meta", {})
    usernum = meta.get("n_users", max(user_train.keys()))
    itemnum = meta.get("n_items", max(
        max(v) for v in user_train.values() if v
    ))

    return {
        "user_train": user_train,
        "user_valid": user_valid,
        "user_test": user_test,
        "usernum": usernum,
        "itemnum": itemnum,
    }


def random_neq(l: int, r: int, s: set) -> int:
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def _sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, seed):
    """Worker process: generate BPR training samples.

    Matches pmixer's WarpSampler format: (uid, seq, pos, neg) numpy arrays.
    """
    def sample(uid):
        while len(user_train.get(uid, [])) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return uid, seq, pos, neg

    np.random.seed(seed)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler:
    """Multiprocess batch sampler — same API as pmixer's WarpSampler."""

    def __init__(self, user_train, usernum, itemnum,
                 batch_size=64, maxlen=50, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            p = Process(
                target=_sample_function,
                args=(user_train, usernum, itemnum, batch_size, maxlen,
                      self.result_queue, np.random.randint(int(2e9))),
            )
            p.daemon = True
            p.start()
            self.processors.append(p)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
