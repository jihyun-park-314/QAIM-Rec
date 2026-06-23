# -*- coding: utf-8 -*-
"""Parallel memory build: split users across N worker processes.

Each worker:
  - Loads candidates file independently (no shared memory).
  - Creates its own LLMClient (shared SQLite cache, WAL mode).
  - Creates its own EmbeddingModel (own CUDA context via spawn).
  - Writes results to chunk_{worker_id:04d}.jsonl.

Main process merges chunk files into a single output JSONL + report.

Usage (full run, 4 workers):
  python scripts/parallel_memory_build.py \\
      --category Books --variant B \\
      --n_workers 4 \\
      --output_dir data/memory_full \\
      --report_dir reports

Usage (smoke test, 2 workers × 10 users):
  python scripts/parallel_memory_build.py \\
      --category Books --variant B \\
      --n_workers 2 --max_users_per_worker 10 \\
      --output_dir data/memory_full_test \\
      --report_dir reports
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Config

@dataclass
class WorkerConfig:
    category: str
    variant: str
    max_reviews_per_user: int
    eligible_min: int
    k_min: int
    k_max: int
    tau: float
    llm_config_path: str
    embed_model_name: str
    data_dir: str
    candidates_path: str


# ---------------------------------------------------------------------------
# Worker

def _load_user_groups(candidates_path: str, user_ids: set) -> dict:
    groups: dict = defaultdict(list)
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["user_id"] in user_ids:
                groups[r["user_id"]].append(r)
    return dict(groups)


def _worker(worker_id: int, user_ids: list, cfg: WorkerConfig, chunk_path: str) -> dict:
    """Entry point for each worker process (spawned, no shared state)."""
    from src.llm.client import LLMClient, load_llm_config
    from src.memory.embed import EmbeddingModel
    from src.memory.run_user_pilot import run_user_pipeline

    uid_set = set(user_ids)
    user_groups = _load_user_groups(cfg.candidates_path, uid_set)

    llm_cfg = load_llm_config(cfg.llm_config_path)
    client = LLMClient(llm_cfg)
    emb_model = EmbeddingModel(cfg.embed_model_name, device="cpu")

    rows = []
    t_start = time.perf_counter()

    for i, uid in enumerate(user_ids):
        reviews = user_groups.get(uid, [])
        ur = run_user_pipeline(
            client, emb_model, uid, reviews, cfg.variant,
            cfg.max_reviews_per_user, cfg.k_min, cfg.k_max, cfg.tau,
        )
        k_str = f"K={ur.k_personal}" if ur.k_personal > 0 else "K=0"
        elapsed = time.perf_counter() - t_start
        print(
            f"  [W{worker_id:02d} {i+1:4d}/{len(user_ids)}] "
            f"uid={str(uid)[:12]:12s} in={ur.n_reviews_input:2d} "
            f"elig={ur.n_eligible:2d} {k_str}  ({elapsed:.0f}s)",
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

    os.makedirs(os.path.dirname(os.path.abspath(chunk_path)), exist_ok=True)
    with open(chunk_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    client.close()

    total_reviews = sum(r["n_reviews_input"] for r in rows)
    total_eligible = sum(r["n_eligible"] for r in rows)
    total_cache = sum(r["n_cache_hit"] for r in rows)
    return {
        "worker_id": worker_id,
        "n_users": len(rows),
        "n_reviews": total_reviews,
        "n_eligible": total_eligible,
        "n_cache_hit": total_cache,
        "elapsed_s": time.perf_counter() - t_start,
        "chunk_path": chunk_path,
    }


# ---------------------------------------------------------------------------
# User selection (same logic as run_user_pilot.select_users)

def select_and_split(
    candidates_path: str,
    eligible_min: int,
    max_reviews_per_user: int,
    n_users: int,
    n_workers: int,
    seed: int,
) -> tuple[list[list], int]:
    """Return (chunks, total_selected) where chunks[i] is user_id list for worker i."""
    groups: dict = defaultdict(list)
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                groups[r["user_id"]].append(r)

    eligible = [u for u, recs in groups.items() if len(recs) >= eligible_min]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:n_users]

    # Split into n_workers chunks (nearly equal size)
    chunks = [[] for _ in range(n_workers)]
    for idx, uid in enumerate(selected):
        chunks[idx % n_workers].append(uid)

    return chunks, len(selected)


# ---------------------------------------------------------------------------
# Report

def _compute_report(all_rows: list[dict], variant: str) -> dict:
    n = len(all_rows)
    if n == 0:
        return {"variant": variant, "n_users": 0}

    total_reviews = sum(r["n_reviews_input"] for r in all_rows)
    total_parse = sum(r["n_parse_success"] for r in all_rows)
    total_eligible = sum(r["n_eligible"] for r in all_rows)
    total_leakage = sum(r["n_leakage"] for r in all_rows)
    total_cache = sum(r["n_cache_hit"] for r in all_rows)
    k_dist = Counter(r["k_personal"] for r in all_rows)
    mean_k = statistics.mean(r["k_personal"] for r in all_rows)

    lats = [r["latency_total_s"] for r in all_rows if r["latency_total_s"] > 0]
    mean_lat_per_user = statistics.mean(lats) if lats else 0.0

    miss_reviews = [
        r["n_reviews_input"] - r["n_cache_hit"]
        for r in all_rows
        if (r["n_reviews_input"] - r["n_cache_hit"]) > 0
    ]
    lat_per_review = [
        r["latency_total_s"] / (r["n_reviews_input"] - r["n_cache_hit"])
        for r in all_rows
        if (r["n_reviews_input"] - r["n_cache_hit"]) > 0
    ]
    mean_lat_per_review = statistics.mean(lat_per_review) if lat_per_review else 0.0

    k0 = k_dist.get(0, 0)
    k1 = k_dist.get(1, 0)
    k2p = sum(v for k, v in k_dist.items() if k >= 2)

    return {
        "variant": variant,
        "n_users": n,
        "n_reviews_processed": total_reviews,
        "parse_success_rate": round(total_parse / max(total_reviews, 1), 4),
        "eligible_rate_per_review": round(total_eligible / max(total_reviews, 1), 4),
        "mean_eligible_per_user": round(total_eligible / max(n, 1), 2),
        "leakage_rate": round(total_leakage / max(total_reviews, 1), 4),
        "cache_hit_rate": round(total_cache / max(total_reviews, 1), 4),
        "k_personal_distribution": dict(sorted(k_dist.items())),
        "mean_k_personal": round(mean_k, 3),
        "k_strata": {
            "k0_n": k0, "k0_rate": round(k0 / max(n, 1), 4),
            "k1_n": k1, "k1_rate": round(k1 / max(n, 1), 4),
            "k2plus_n": k2p, "k2plus_rate": round(k2p / max(n, 1), 4),
        },
        "latency": {
            "mean_lat_per_review_s": round(mean_lat_per_review, 3),
            "mean_lat_per_user_s": round(mean_lat_per_user, 1),
        },
    }


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Books")
    parser.add_argument("--variant", default="B")
    parser.add_argument("--n_users", type=int, default=9807,
                        help="Max users to process (default: all eligible)")
    parser.add_argument("--eligible_min", type=int, default=2)
    parser.add_argument("--max_reviews_per_user", type=int, default=12)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--max_users_per_worker", type=int, default=None,
                        help="Cap per worker for smoke testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--llm_config_path", default="configs/llm/p1.yaml")
    parser.add_argument("--embed_model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/memory_full")
    parser.add_argument("--report_dir", default="reports")
    args = parser.parse_args()

    candidates_path = os.path.join(args.data_dir, args.category, "books_memory_candidates.jsonl")
    if not os.path.isfile(candidates_path):
        sys.exit(f"[parallel] Candidates not found: {candidates_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print(f"[parallel] Selecting users from {candidates_path} ...")
    chunks, total_selected = select_and_split(
        candidates_path, args.eligible_min, args.max_reviews_per_user,
        args.n_users, args.n_workers, args.seed,
    )

    # Apply per-worker cap for smoke testing
    if args.max_users_per_worker:
        chunks = [c[: args.max_users_per_worker] for c in chunks]
        total_selected = sum(len(c) for c in chunks)

    print(f"[parallel] {total_selected} users → {args.n_workers} workers "
          f"({[len(c) for c in chunks]} users each)")

    cfg = WorkerConfig(
        category=args.category,
        variant=args.variant,
        max_reviews_per_user=args.max_reviews_per_user,
        eligible_min=args.eligible_min,
        k_min=args.k_min,
        k_max=args.k_max,
        tau=args.tau,
        llm_config_path=args.llm_config_path,
        embed_model_name=args.embed_model_name,
        data_dir=args.data_dir,
        candidates_path=candidates_path,
    )

    chunk_paths = [
        os.path.join(args.output_dir, f"chunk_{i:04d}.jsonl")
        for i in range(args.n_workers)
    ]

    # Spawn (not fork): safe for CUDA — each worker initialises its own GPU context.
    ctx = mp.get_context("spawn")
    worker_args = [
        (i, chunks[i], cfg, chunk_paths[i])
        for i in range(args.n_workers)
        if chunks[i]  # skip empty chunks
    ]

    t0 = time.perf_counter()
    with ctx.Pool(processes=args.n_workers) as pool:
        worker_stats = pool.starmap(_worker, worker_args)
    elapsed = time.perf_counter() - t0

    # Verify all chunks exist and are non-empty
    print(f"\n[parallel] All workers done in {elapsed:.1f}s. Merging ...")
    missing = [s["chunk_path"] for s in worker_stats if not os.path.isfile(s["chunk_path"])]
    if missing:
        sys.exit(f"[parallel] ERROR: missing chunk files: {missing}")

    # Merge chunks → single output file
    output_path = os.path.join(
        args.output_dir,
        f"memory_{args.variant.lower()}_u{total_selected}_seed{args.seed}.jsonl",
    )
    all_rows = []
    with open(output_path, "w", encoding="utf-8") as out_f:
        for chunk_path in chunk_paths:
            if not os.path.isfile(chunk_path):
                continue
            with open(chunk_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_rows.append(json.loads(line))
                        out_f.write(line + "\n")

    print(f"[parallel] Merged {len(all_rows)} users → {output_path}")

    # Integrity check: no duplicate users
    user_ids_out = [r["user_id"] for r in all_rows]
    duplicates = len(user_ids_out) - len(set(user_ids_out))
    if duplicates:
        print(f"[parallel] WARNING: {duplicates} duplicate user_ids in output!")
    else:
        print(f"[parallel] Integrity OK: no duplicate user_ids")

    # Coverage check: all selected users accounted for
    selected_all = set(uid for chunk in chunks for uid in chunk)
    output_set = set(user_ids_out)
    missing_users = selected_all - output_set
    if missing_users:
        print(f"[parallel] WARNING: {len(missing_users)} users missing from output!")
    else:
        print(f"[parallel] Coverage OK: all {len(selected_all)} selected users present")

    # Report
    report = _compute_report(all_rows, args.variant)
    report["parallel"] = {
        "n_workers": args.n_workers,
        "wall_time_s": round(elapsed, 1),
        "worker_stats": worker_stats,
    }
    report_path = os.path.join(
        args.report_dir,
        f"parallel_report_{args.variant.lower()}_u{total_selected}_seed{args.seed}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[parallel] Report → {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"VARIANT {args.variant}  (n_users={report['n_users']})")
    print(f"{'='*60}")
    print(f"  wall_time               : {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print(f"  parse_success_rate      : {report['parse_success_rate']:.3f}")
    print(f"  eligible_rate_per_review: {report['eligible_rate_per_review']:.3f}")
    print(f"  mean_k_personal         : {report['mean_k_personal']}")
    print(f"  k_distribution          : {report['k_personal_distribution']}")
    s = report["k_strata"]
    print(f"  K=0 rate                : {s['k0_rate']:.3f} (n={s['k0_n']})")
    print(f"  K>=2 rate               : {s['k2plus_rate']:.3f} (n={s['k2plus_n']})")
    print(f"  cache_hit_rate          : {report['cache_hit_rate']:.3f}")
    print(f"  leakage_rate            : {report['leakage_rate']:.4f}")

    # Clean up chunk files after successful merge
    for path in chunk_paths:
        if os.path.isfile(path):
            os.remove(path)
    print(f"[parallel] Chunk files removed.")


if __name__ == "__main__":
    main()
