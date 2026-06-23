"""Pilot 3 — Leakage probe / leakage floor (plan.md §4 Stage P3, v0.2 redefinition).

Per plan.md §1 design principle, this pilot is a pure
`config -> input path -> output path` CLI that reuses M1/`prompts/p2_pseudo_query.py`
and full-catalog retrieval with a small config (no separate implementation).

go/no-go: masking_pass_rate >= config.masking_pass_rate_threshold (default 0.70).
r_query_only (Recall@k) is recorded as a leakage floor reference, NOT a pass/fail criterion.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.pilot.common import load_config, write_report


@dataclass
class Pilot3Config:
    """Config for Pilot 3 (plan.md §4 Stage P3).

    Note: P1's `purpose` text is NOT used here, since it is not guaranteed to
    be decontaminated (plan.md §4 P3, requirement [필수]#3) — only P2
    pseudo-queries that pass the masking/validation check are used.
    """

    category: str = "Office_Products"
    n_users: int = 100
    recall_ks: list = None  # e.g. [10, 50]

    llm_config_path: str = "configs/llm/p2.yaml"
    embedding_model_id: str = "BAAI/bge-base-en-v1.5"  # plan.md §7 #8

    masking_pass_rate_threshold: float = 0.70

    seed: int = 42
    output_path: str = "results/pilot/pilot3_report.json"


@dataclass
class Pilot3Report:
    """Pilot 3 results (plan.md §4 Stage P3, v0.2 redefinition).

    `r_query_only` (Recall@k for k in config.recall_ks, keyed by k) is NOT an
    absolute pass/fail threshold. It is recorded as a "leakage floor" — the
    baseline that F8's `query_only` comparison group is measured against, used
    to interpret the *additional* lift a steered model provides over
    query-only retrieval (plan.md §4 Stage P3).

    go_nogo is True iff `masking_pass_rate` is measurable and
    >= config.masking_pass_rate_threshold (default 0.70). A high
    `r_query_only` is NOT a no-go condition.
    """

    r_query_only: dict  # {k: recall@k} for k in config.recall_ks

    masking_pass_rate: float
    n_total: int
    n_passed_masking: int

    go_nogo: bool
    notes: str


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_sequences(sequences_path: str) -> dict:
    """Load sequences.jsonl → {user_id(int): {orig_user_id, eligible_items[]}}."""
    result = {}
    with open(sequences_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec["user_id"]
            eligible = [
                it for it in rec.get("items", [])
                if it.get("is_eligible") and it.get("has_review")
            ]
            result[uid] = {
                "orig_user_id": rec["orig_user_id"],
                "eligible_items": eligible,  # train-only (is_eligible=True never set on val/test)
            }
    return result


def _load_splits(splits_path: str) -> dict:
    """Load splits.json → {user_id(int): {train, val, test}}."""
    with open(splits_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(uid): v for uid, v in raw["users"].items()}


def _load_id_maps(id_maps_path: str) -> dict:
    """Load id_maps.json → id2item {int → asin_str}."""
    with open(id_maps_path, encoding="utf-8") as f:
        maps = json.load(f)
    return {int(k): v for k, v in maps["id2item"].items()}


def _build_asin2title(meta_path: str, needed_asins: set) -> dict:
    """Stream meta.jsonl → {asin: title} for needed ASINs only."""
    asin2title = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("parent_asin", "")
            if asin in needed_asins:
                title = rec.get("title", "")
                if title:
                    asin2title[asin] = title
    return asin2title


def _collect_reviews(reviews_path: str, needed: dict) -> dict:
    """Stream reviews.jsonl to collect needed (orig_user_id, asin) → review_text.

    needed: {orig_user_id: {asin: True, ...}, ...}
    Returns: {(orig_user_id, asin): review_text}
    """
    found = {}
    total_needed = sum(len(v) for v in needed.values())
    print(f"  [reviews] streaming {reviews_path} for {total_needed} (user,item) pairs ...",
          flush=True)

    with open(reviews_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0 and i > 0:
                print(f"  [reviews] scanned {i//1_000_000}M lines, "
                      f"found {len(found)}/{total_needed}", flush=True)
            if len(found) >= total_needed:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("user_id", "")
            asin = rec.get("parent_asin") or rec.get("asin", "")
            if uid in needed and asin in needed[uid]:
                key = (uid, asin)
                if key not in found:
                    text = rec.get("text", "").strip()
                    if text:
                        found[key] = text
    print(f"  [reviews] found {len(found)}/{total_needed} reviews", flush=True)
    return found


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_pilot3(config: Pilot3Config) -> Pilot3Report:
    """Run Pilot 3: leakage floor probe over `config.n_users` users from
    `config.category` (plan.md §4 Stage P3).

    Procedure:
      1. Select `config.n_users` users (subset of Pilot 2's candidate users).
      2. M1/`prompts/p2_pseudo_query.py` (configured by `config.llm_config_path`):
         generate decontaminated pseudo-queries per user, validating that no
         `title` tokens (brand/model/exact spec) leak into the query, with
         retry on failure (plan.md §5 P2 parsing/validation).
      3. Embed pseudo-queries with `config.embedding_model_id` and search the
         full item catalog for each user's held-out test target item ->
         `r_query_only[k]` = Recall@k for each k in `config.recall_ks`.
      4. Compute `masking_pass_rate` = n_passed_masking / n_total.
      5. go/no-go: masking_pass_rate >= config.masking_pass_rate_threshold
         (plan.md §4 Stage P3 — `r_query_only` itself is not a go/no-go
         criterion, only a recorded leakage floor).
    """
    from src.llm.client import load_llm_config, LLMClient
    from src.llm.prompts.p2_pseudo_query import run_p2_query
    from src.memory.embed import EmbeddingModel

    recall_ks = config.recall_ks or [10, 50]
    rng = random.Random(config.seed)

    # ------------------------------------------------------------------
    # 1. Load data structures
    # ------------------------------------------------------------------
    category = config.category
    data_dir = Path(f"data/processed/{category}")
    raw_dir = Path(f"data/raw/{category}")

    print(f"[pilot3] loading data for {category} ...", flush=True)

    sequences = _load_sequences(str(data_dir / "sequences.jsonl"))
    splits = _load_splits(str(data_dir / "splits.json"))
    id2item = _load_id_maps(str(data_dir / "id_maps.json"))

    # ------------------------------------------------------------------
    # 2. Select eligible users (≥1 eligible train item + has test target)
    # ------------------------------------------------------------------
    eligible_users = [
        uid for uid, info in sequences.items()
        if info["eligible_items"] and splits.get(uid, {}).get("test")
    ]
    print(f"[pilot3] eligible users: {len(eligible_users)} / {len(sequences)}", flush=True)

    n_sample = min(config.n_users, len(eligible_users))
    sampled_users = sorted(rng.sample(eligible_users, n_sample))
    print(f"[pilot3] sampled {n_sample} users (seed={config.seed})", flush=True)

    # ------------------------------------------------------------------
    # 3. For each user, pick one eligible review item (most recent in train)
    #    and identify the test target ASIN.
    # ------------------------------------------------------------------
    user_review_items: dict[int, dict] = {}  # uid → {orig_item_id, item_id, orig_user_id}
    for uid in sampled_users:
        info = sequences[uid]
        eligible = info["eligible_items"]
        # Pick most recent eligible item (ts descending)
        best = max(eligible, key=lambda it: it.get("ts", 0))
        user_review_items[uid] = {
            "orig_user_id": info["orig_user_id"],
            "orig_item_id": best["orig_item_id"],
            "item_id": best["item_id"],
        }

    # Identify test target ASIN for each user
    user_test_asin: dict[int, str] = {}
    for uid in sampled_users:
        test_iid = splits[uid]["test"]
        test_asin = id2item.get(test_iid, "")
        if test_asin:
            user_test_asin[uid] = test_asin

    # ------------------------------------------------------------------
    # 4. Build asin2title from meta.jsonl (all catalog items + review items)
    # ------------------------------------------------------------------
    all_catalog_asins = set(id2item.values())
    review_asins = {v["orig_item_id"] for v in user_review_items.values()}
    needed_asins = all_catalog_asins | review_asins

    print(f"[pilot3] building asin2title for {len(needed_asins)} items ...", flush=True)
    asin2title = _build_asin2title(str(raw_dir / "meta.jsonl"), needed_asins)
    print(f"[pilot3] asin2title: {len(asin2title)} / {len(needed_asins)} found", flush=True)

    # ------------------------------------------------------------------
    # 5. Collect review texts (stream reviews.jsonl once)
    # ------------------------------------------------------------------
    needed_reviews: dict[str, dict] = {}  # orig_user_id → {asin: True}
    for uid, rv in user_review_items.items():
        orig_uid = rv["orig_user_id"]
        asin = rv["orig_item_id"]
        needed_reviews.setdefault(orig_uid, {})[asin] = True

    review_texts = _collect_reviews(str(raw_dir / "reviews.jsonl"), needed_reviews)
    print(f"[pilot3] collected {len(review_texts)} review texts", flush=True)

    # ------------------------------------------------------------------
    # 6. Generate P2 pseudo-queries for each user
    # ------------------------------------------------------------------
    llm_cfg = load_llm_config(config.llm_config_path)
    client = LLMClient(llm_cfg)

    print(f"[pilot3] generating P2 pseudo-queries ({n_sample} users) ...", flush=True)

    user_queries: dict[int, str] = {}    # uid → query string (masking-passed only)
    n_total = 0
    n_null_query = 0
    n_llm_fail = 0
    n_mask_fail = 0
    n_passed = 0

    for i, uid in enumerate(sampled_users):
        rv = user_review_items[uid]
        orig_uid = rv["orig_user_id"]
        asin = rv["orig_item_id"]
        review_text = review_texts.get((orig_uid, asin), "")

        if not review_text:
            n_total += 1
            n_llm_fail += 1
            continue

        title = asin2title.get(asin, "")
        item_dict = {
            "title": title,
            "category": category,
            "review_text": review_text,
        }

        result = run_p2_query(client, item_dict, retry_max=1)
        n_total += 1

        if result.failed_reason == "null_query":
            n_null_query += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_sample}] uid={uid} null_query "
                      f"(cache={result.cache_hit})", flush=True)
        elif result.failed_reason == "llm_fail":
            n_llm_fail += 1
        elif result.failed_reason == "mask_fail":
            n_mask_fail += 1
            # Still record the query for Recall computation (it's a leakage floor)
            # but don't count it toward masking_pass_rate
            user_queries[uid] = result.query  # type: ignore[assignment]
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_sample}] uid={uid} MASK_FAIL "
                      f"query='{result.query}' (cache={result.cache_hit})", flush=True)
        else:
            n_passed += 1
            user_queries[uid] = result.query  # type: ignore[assignment]
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_sample}] uid={uid} OK "
                      f"query='{result.query}' (cache={result.cache_hit})", flush=True)

    client.close()
    print(f"[pilot3] P2 done: total={n_total}, passed={n_passed}, "
          f"mask_fail={n_mask_fail}, null={n_null_query}, llm_fail={n_llm_fail}",
          flush=True)

    # masking_pass_rate counts only properly-generated queries (excludes null/llm_fail)
    n_attempted = n_total - n_null_query - n_llm_fail
    masking_pass_rate = n_passed / n_attempted if n_attempted > 0 else 0.0

    # ------------------------------------------------------------------
    # 7. Build catalog embeddings (all item titles)
    # ------------------------------------------------------------------
    print(f"[pilot3] loading embedding model {config.embedding_model_id} ...", flush=True)
    embed_model = EmbeddingModel(model_id=config.embedding_model_id)

    # Ordered catalog: item_id 1..N
    max_item_id = max(id2item.keys())
    catalog_ids: list[int] = []
    catalog_titles: list[str] = []
    for iid in range(1, max_item_id + 1):
        asin = id2item.get(iid)
        if asin is None:
            continue
        title = asin2title.get(asin, asin)  # fallback to asin if no title
        catalog_ids.append(iid)
        catalog_titles.append(title)

    print(f"[pilot3] embedding {len(catalog_ids)} catalog items ...", flush=True)
    catalog_embs = embed_model.encode_corpus(catalog_titles)  # [N, d]
    print(f"[pilot3] catalog embeddings: {catalog_embs.shape}", flush=True)

    # Build item_id → row_index in catalog_embs
    iid2row = {iid: idx for idx, iid in enumerate(catalog_ids)}

    # ------------------------------------------------------------------
    # 8. Embed queries and compute Recall@k
    # ------------------------------------------------------------------
    # Only users with a passed query AND a known test target
    eval_users = [
        uid for uid in sampled_users
        if uid in user_queries and uid in user_test_asin
    ]
    print(f"[pilot3] eval users (have query + test target): {len(eval_users)}", flush=True)

    if not eval_users:
        print("[pilot3] WARNING: no users to evaluate — returning zeros", flush=True)
        r_query_only = {str(k): 0.0 for k in recall_ks}
        return Pilot3Report(
            r_query_only=r_query_only,
            masking_pass_rate=masking_pass_rate,
            n_total=n_total,
            n_passed_masking=n_passed,
            go_nogo=False,
            notes=f"No eval users. null={n_null_query} llm_fail={n_llm_fail}",
        )

    queries = [user_queries[uid] for uid in eval_users]
    print(f"[pilot3] embedding {len(queries)} queries ...", flush=True)
    query_embs = embed_model.encode_queries(queries)  # [M, d]

    # Scores: [M, N] = query_embs @ catalog_embs.T
    import numpy as np
    scores = query_embs @ catalog_embs.T  # [M, N]

    recall_hits = {k: 0 for k in recall_ks}
    for i, uid in enumerate(eval_users):
        test_asin = user_test_asin[uid]
        test_iid = None
        # Find the item_id for this test ASIN
        for iid, a in id2item.items():
            if a == test_asin:
                test_iid = iid
                break
        if test_iid is None or test_iid not in iid2row:
            continue

        target_row = iid2row[test_iid]
        target_score = scores[i, target_row]
        rank = int((scores[i] > target_score).sum())  # 0-indexed rank

        for k in recall_ks:
            if rank < k:
                recall_hits[k] += 1

    n_eval = len(eval_users)
    r_query_only = {str(k): round(recall_hits[k] / n_eval, 6) for k in recall_ks}

    print(f"[pilot3] Recall@k over {n_eval} users:", flush=True)
    for k in recall_ks:
        print(f"  Recall@{k} = {r_query_only[str(k)]:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 9. go/no-go decision
    # ------------------------------------------------------------------
    go_nogo = masking_pass_rate >= config.masking_pass_rate_threshold

    notes_parts = [
        f"n_attempted={n_attempted}",
        f"n_passed={n_passed}",
        f"n_mask_fail={n_mask_fail}",
        f"n_null_query={n_null_query}",
        f"n_llm_fail={n_llm_fail}",
        f"n_eval={n_eval}",
    ]
    notes = ", ".join(notes_parts)
    if not go_nogo:
        notes = f"FAIL masking_pass_rate={masking_pass_rate:.3f} < {config.masking_pass_rate_threshold}. " + notes

    print(f"[pilot3] masking_pass_rate={masking_pass_rate:.3f} "
          f"(threshold={config.masking_pass_rate_threshold}) → go_nogo={go_nogo}", flush=True)

    return Pilot3Report(
        r_query_only=r_query_only,
        masking_pass_rate=round(masking_pass_rate, 4),
        n_total=n_total,
        n_passed_masking=n_passed,
        go_nogo=go_nogo,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 3: leakage probe (plan.md §4 Stage P3)")
    parser.add_argument("--config", required=True, help="Path to Pilot 3 YAML config")
    args = parser.parse_args()

    config = load_config(args.config, Pilot3Config)
    if config.recall_ks is None:
        config.recall_ks = [10, 50]

    print(f"[pilot3] config: {config}", flush=True)
    report = run_pilot3(config)
    write_report(report, config.output_path)
    print(f"[pilot3] report written to {config.output_path}", flush=True)
    print(f"[pilot3] go_nogo={report.go_nogo}, "
          f"masking_pass_rate={report.masking_pass_rate}, "
          f"r_query_only={report.r_query_only}", flush=True)


if __name__ == "__main__":
    main()
