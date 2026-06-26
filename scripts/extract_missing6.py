"""One-shot re-extraction for 6 missing users from parallel_memory_build.

Missing user_ids: [503, 4718, 4951, 6229, 6822, 7969]
Writes to data/memory_full/chunk_missing6.jsonl, then appends to
memory_b_u8918_seed42.jsonl → memory_b_u8924_seed42.jsonl.

Usage:
    docker exec -e PYTHONPATH=/qaim-rec qaim-rec python3 \\
        scripts/extract_missing6.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MISSING_UIDS = [503, 4718, 4951, 6229, 6822, 7969]

CANDIDATES_PATH = "data/processed/Books/books_memory_candidates.jsonl"
LLM_CONFIG_PATH = "configs/llm/p1.yaml"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
VARIANT = "p1_base"
MAX_REVIEWS = 12
K_MIN = 1
K_MAX = 5
TAU = 0.3

CHUNK_OUT = "data/memory_full/chunk_missing6.jsonl"
FULL_IN = "data/memory_full/memory_b_u8918_seed42.jsonl"
FINAL_OUT = "data/memory_full/memory_b_u8924_seed42.jsonl"


def load_user_groups(candidates_path: str, uid_set: set) -> dict:
    from collections import defaultdict
    groups: dict = defaultdict(list)
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["user_id"] in uid_set:
                groups[r["user_id"]].append(r)
    return dict(groups)


def main() -> None:
    from src.llm.client import LLMClient, load_llm_config
    from src.memory.embed import EmbeddingModel
    from src.memory.run_user_pilot import run_user_pipeline

    print(f"[missing6] Extracting {len(MISSING_UIDS)} users: {MISSING_UIDS}")

    uid_set = set(MISSING_UIDS)
    user_groups = load_user_groups(CANDIDATES_PATH, uid_set)
    found = set(user_groups.keys())
    not_found = uid_set - found
    if not_found:
        print(f"  WARNING: {not_found} not found in candidates file")

    llm_cfg = load_llm_config(LLM_CONFIG_PATH)
    client = LLMClient(llm_cfg)
    emb_model = EmbeddingModel(EMBED_MODEL, device="cpu")

    rows = []
    t_start = time.perf_counter()

    for i, uid in enumerate(MISSING_UIDS):
        reviews = user_groups.get(uid, [])
        if not reviews:
            print(f"  [skip] uid={uid} — no reviews in candidates")
            continue
        ur = run_user_pipeline(
            client, emb_model, uid, reviews, VARIANT,
            MAX_REVIEWS, K_MIN, K_MAX, TAU,
        )
        k_str = f"K={ur.k_personal}" if ur.k_personal > 0 else "K=0"
        elapsed = time.perf_counter() - t_start
        print(
            f"  [{i+1}/{len(MISSING_UIDS)}] uid={uid}  "
            f"in={ur.n_reviews_input}  elig={ur.n_eligible}  {k_str}  ({elapsed:.0f}s)",
            flush=True,
        )
        rows.append({
            "user_id": ur.user_id,
            "variant": ur.variant,
            "n_reviews_input": ur.n_reviews_input,
            "n_parse_success": ur.n_parse_success,
            "n_eligible": ur.n_eligible,
            "k_personal": ur.k_personal,
            "n_leakage": ur.n_leakage,
            "latency_total_s": ur.latency_total_s,
            "n_cache_hit": ur.n_cache_hit,
            "cluster_summaries": [
                {
                    "label": c["label"],
                    "size": c["size"],
                    "intents": c["intents"],
                    "source_texts": c["source_texts"],
                }
                for c in ur.cluster_summaries
            ],
        })

    client.close()

    # Write chunk
    os.makedirs(os.path.dirname(os.path.abspath(CHUNK_OUT)), exist_ok=True)
    with open(CHUNK_OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[missing6] Wrote {len(rows)} records → {CHUNK_OUT}")

    # Merge: full(8918) + new 6 → final
    all_rows: dict[int, dict] = {}
    with open(FULL_IN, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            all_rows[rec["user_id"]] = rec

    n_before = len(all_rows)
    for rec in rows:
        all_rows[rec["user_id"]] = rec
    n_added = len(all_rows) - n_before

    with open(FINAL_OUT, "w", encoding="utf-8") as f:
        for rec in all_rows.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[missing6] Merged: {n_before} + {n_added} new = {len(all_rows)} users")
    print(f"[missing6] Final bank: {FINAL_OUT}")


if __name__ == "__main__":
    main()
