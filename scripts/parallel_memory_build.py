# -*- coding: utf-8 -*-
"""v0.4.6 — Parallel P1 extraction: deterministic selection, per-review canonical output.

Each worker:
  - Loads candidates.jsonl independently (no shared memory).
  - Selects reviews per user: timestamp DESC + item_id stable tiebreak → top cap.
  - Creates its own LLMClient (shared SQLite cache, WAL mode). No EmbeddingModel.
  - Writes per-review P1 records to worker_{id:04d}.jsonl shard.

Main process:
  - Verifies candidates.jsonl MD5 before anything else.
  - Merges shards → data/processed/Books/p1_extractions.jsonl (single canonical artifact).
  - Writes p1_extractions.manifest.json: row_count, user_count, md5, prompt_version, cap.

Usage (full run, 4 workers):
  python scripts/parallel_memory_build.py \\
      --category Books --variant B \\
      --n_workers 4 \\
      --output_dir data/p1_shards \\
      --report_dir reports

Usage (smoke test, 2 workers × 10 users):
  python scripts/parallel_memory_build.py \\
      --category Books --variant B \\
      --n_workers 2 --max_users_per_worker 10 \\
      --output_dir data/p1_shards_test \\
      --report_dir reports
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CANDIDATES_MD5 = "3a31f333b9e0b46e0ce5112a578d44a5"
CANONICAL_RELATIVE = "data/processed/Books/p1_extractions.jsonl"
MANIFEST_RELATIVE = "data/processed/Books/p1_extractions.manifest.json"


# ---------------------------------------------------------------------------
# Config

@dataclass
class WorkerConfig:
    category: str
    variant: str
    cap: int
    eligible_min: int
    k_min: int
    k_max: int
    tau: float
    llm_config_path: str
    data_dir: str
    candidates_path: str
    api_url: str | None = None  # overrides llm_config api_url if set


# ---------------------------------------------------------------------------
# MD5 check

def verify_md5(path: str, expected: str) -> None:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        sys.exit(
            f"[ABORT] MD5 mismatch for {path}\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}"
        )
    print(f"[md5] OK  {actual}  {path}")


# ---------------------------------------------------------------------------
# Deterministic review selection

def select_user_reviews(reviews: list[dict], cap: int) -> list[dict]:
    """Select reviews: timestamp DESC (most-recent first) + item_id stable tiebreak."""
    sorted_reviews = sorted(
        reviews,
        key=lambda r: (-int(r.get("timestamp", 0)), str(r.get("item_id", "")))
    )
    return sorted_reviews[:cap]


# ---------------------------------------------------------------------------
# Load user groups from candidates file

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


# ---------------------------------------------------------------------------
# Worker

def _worker(worker_id: int, user_ids: list, cfg: WorkerConfig, shard_path: str) -> dict:
    """Entry point for each worker process (spawned, no shared state)."""
    from src.llm.client import LLMClient, load_llm_config
    from src.memory.pipeline import extract_record

    uid_set = set(user_ids)
    user_groups = _load_user_groups(cfg.candidates_path, uid_set)

    llm_cfg = load_llm_config(cfg.llm_config_path)
    if cfg.api_url:
        llm_cfg.api_url = cfg.api_url
    client = LLMClient(llm_cfg)

    rows = []
    t_start = time.perf_counter()

    for i, uid in enumerate(user_ids):
        reviews = user_groups.get(uid, [])
        selected = select_user_reviews(reviews, cfg.cap)

        user_rows_before = len(rows)
        for item in selected:
            rec = extract_record(client, item, variant=cfg.variant)
            rows.append({
                "user_id": rec.user_id,
                "item_id": rec.item_id,
                "timestamp": item.get("timestamp"),
                "rating": item.get("rating"),
                "review_text": item.get("review_text", ""),
                "item_title": rec.item_title,
                "contextual_intent": rec.contextual_intent,
                "preference_summary": rec.preference_summary,
                "evidence_span": rec.evidence_span,
                "is_discriminative": rec.is_discriminative,
                "grounding_level": rec.grounding_level,
                "eligible": rec.eligible,
                "source_text": rec.source_text,
                "parse_failed": rec.parse_failed,
                "leakage_detected": rec.leakage_detected,
                "latency_s": rec.latency_s,
                "cache_hit": rec.cache_hit,
                "variant": rec.variant,
            })

        n_elig = sum(1 for r in rows[user_rows_before:] if r["eligible"])
        elapsed = time.perf_counter() - t_start
        print(
            f"  [W{worker_id:02d} {i+1:4d}/{len(user_ids)}] "
            f"uid={str(uid)[:12]:12s} in={len(selected):2d} "
            f"elig={n_elig:2d}  ({elapsed:.0f}s)",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(shard_path)), exist_ok=True)
    with open(shard_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    client.close()

    n_cache = sum(1 for r in rows if r["cache_hit"])
    n_elig_total = sum(1 for r in rows if r["eligible"])
    return {
        "worker_id": worker_id,
        "n_users": len(user_ids),
        "n_records": len(rows),
        "n_eligible": n_elig_total,
        "n_cache_hit": n_cache,
        "elapsed_s": time.perf_counter() - t_start,
        "shard_path": shard_path,
    }


# ---------------------------------------------------------------------------
# User selection + shard splitting

def select_and_split(
    candidates_path: str,
    eligible_min: int,
    cap: int,
    n_users: int,
    n_workers: int,
    seed: int,
    user_shard: tuple[int, int] | None = None,
) -> tuple[list[list], int, dict]:
    """Return (chunks, total_selected, selected_review_ids).

    selected_review_ids[user_id] = list of item_ids (deterministic, file-order-invariant).
    Users with >= eligible_min reviews in candidates are eligible.
    User ordering is by user_id sort (deterministic, no random).
    user_shard=(N, M): keep only users where sorted_index % M == N (disjoint sharding).
    """
    groups: dict = defaultdict(list)
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                groups[r["user_id"]].append(r)

    eligible_users = sorted(
        [u for u, recs in groups.items() if len(recs) >= eligible_min],
        key=str,
    )
    if user_shard is not None:
        shard_n, shard_m = user_shard
        eligible_users = [u for i, u in enumerate(eligible_users) if i % shard_m == shard_n]

    selected = eligible_users[:n_users]

    selected_review_ids: dict = {}
    for uid in selected:
        top_reviews = select_user_reviews(groups[uid], cap)
        selected_review_ids[uid] = [r["item_id"] for r in top_reviews]

    chunks = [[] for _ in range(n_workers)]
    for idx, uid in enumerate(selected):
        chunks[idx % n_workers].append(uid)

    return chunks, len(selected), selected_review_ids


# ---------------------------------------------------------------------------
# Manifest

def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    canonical_path: str,
    manifest_path: str,
    row_count: int,
    user_count: int,
    prompt_version: str,
    cap: int,
) -> dict:
    md5 = _file_md5(canonical_path)
    manifest = {
        "row_count": row_count,
        "user_count": user_count,
        "md5": md5,
        "prompt_version": prompt_version,
        "cap": cap,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "canonical_path": canonical_path,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Books")
    parser.add_argument("--variant", default="B")
    parser.add_argument("--n_users", type=int, default=99999,
                        help="Max users to process (default: all eligible)")
    parser.add_argument("--eligible_min", type=int, default=1)
    parser.add_argument("--cap", type=int, default=20,
                        help="Max reviews per user (deterministic top-cap by timestamp DESC)")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--max_users_per_worker", type=int, default=None,
                        help="Cap per worker for smoke testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Kept for CLI compat; user order is deterministic (no rng)")
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--llm_config_path", default="configs/llm/p1.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/p1_shards")
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--api_url", default=None,
                        help="Ollama API URL override (e.g. http://localhost:11435/api/chat)")
    parser.add_argument("--user_shard_idx", default=None,
                        help="Disjoint user shard as 'N_OF_M', e.g. '0_OF_2'. "
                             "Skips canonical merge — run merge_shards.py after both shards finish.")
    args = parser.parse_args()

    user_shard: tuple[int, int] | None = None
    if args.user_shard_idx:
        parts = args.user_shard_idx.split("_OF_")
        if len(parts) != 2:
            sys.exit(f"[parallel] --user_shard_idx must be 'N_OF_M', got: {args.user_shard_idx}")
        user_shard = (int(parts[0]), int(parts[1]))

    root = Path(__file__).resolve().parents[1]
    candidates_path = os.path.join(args.data_dir, args.category, "candidates.jsonl")
    canonical_path = str(root / CANONICAL_RELATIVE)
    manifest_path = str(root / MANIFEST_RELATIVE)

    if not os.path.isfile(candidates_path):
        sys.exit(f"[parallel] Candidates not found: {candidates_path}")

    print(f"[parallel] Verifying candidates.jsonl MD5 ...")
    verify_md5(candidates_path, CANDIDATES_MD5)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(canonical_path), exist_ok=True)

    print(f"[parallel] Selecting users from {candidates_path} ...")
    chunks, total_selected, selected_review_ids = select_and_split(
        candidates_path, args.eligible_min, args.cap,
        args.n_users, args.n_workers, args.seed,
        user_shard=user_shard,
    )
    if user_shard is not None:
        print(f"[parallel] Shard {user_shard[0]}_OF_{user_shard[1]}: "
              f"{total_selected} users assigned to this instance")

    # Apply per-worker cap for smoke testing
    if args.max_users_per_worker:
        chunks = [c[:args.max_users_per_worker] for c in chunks]
        total_selected = sum(len(c) for c in chunks)

    print(f"[parallel] {total_selected} users → {args.n_workers} workers "
          f"({[len(c) for c in chunks]} users each), cap={args.cap}")

    # Persist selected_review_ids for smoke test 2A
    sel_path = os.path.join(args.output_dir, "selected_review_ids.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in selected_review_ids.items()},
                  f, ensure_ascii=False)
    print(f"[parallel] selected_review_ids → {sel_path}")

    # Read prompt_version for manifest
    import yaml
    with open(args.llm_config_path, encoding="utf-8") as f:
        llm_cfg_raw = yaml.safe_load(f)
    prompt_version = llm_cfg_raw.get("prompt_version", "unknown")

    cfg = WorkerConfig(
        category=args.category,
        variant=args.variant,
        cap=args.cap,
        eligible_min=args.eligible_min,
        k_min=args.k_min,
        k_max=args.k_max,
        tau=args.tau,
        llm_config_path=args.llm_config_path,
        data_dir=args.data_dir,
        candidates_path=candidates_path,
        api_url=args.api_url,
    )

    shard_paths = [
        os.path.join(args.output_dir, f"worker_{i:04d}.jsonl")
        for i in range(args.n_workers)
    ]

    ctx = mp.get_context("spawn")
    worker_args = [
        (i, chunks[i], cfg, shard_paths[i])
        for i in range(args.n_workers)
        if chunks[i]
    ]

    t0 = time.perf_counter()
    with ctx.Pool(processes=args.n_workers) as pool:
        worker_stats = pool.starmap(_worker, worker_args)
    elapsed = time.perf_counter() - t0

    print(f"\n[parallel] All workers done in {elapsed:.1f}s.")
    missing = [s["shard_path"] for s in worker_stats if not os.path.isfile(s["shard_path"])]
    if missing:
        sys.exit(f"[parallel] ERROR: missing shard files: {missing}")

    if user_shard is not None:
        # Sharded run: skip canonical merge. Run scripts/merge_shards.py after both shards finish.
        total_records = sum(s["n_records"] for s in worker_stats)
        total_eligible = sum(s["n_eligible"] for s in worker_stats)
        total_cache = sum(s["n_cache_hit"] for s in worker_stats)
        print(f"[parallel] Shard {user_shard[0]}_OF_{user_shard[1]} complete: "
              f"{total_records} records, {total_eligible} eligible, "
              f"cache_hit={total_cache/max(total_records,1):.3f}")
        print(f"[parallel] Shards written to: {args.output_dir}")
        print(f"[parallel] Run scripts/merge_shards.py after both shards complete "
              f"to produce canonical p1_extractions.jsonl")
        return

    print(f"[parallel] Merging → {canonical_path} ...")
    # Merge shards → single canonical artifact
    all_rows = []
    with open(canonical_path, "w", encoding="utf-8") as out_f:
        for shard_path in shard_paths:
            if not os.path.isfile(shard_path):
                continue
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_rows.append(json.loads(line))
                        out_f.write(line + "\n")

    row_count = len(all_rows)
    user_count = len(set(r["user_id"] for r in all_rows))
    print(f"[parallel] Merged {row_count} records from {user_count} users → {canonical_path}")

    # Integrity checks
    selected_all = set(uid for chunk in chunks for uid in chunk)
    output_users = set(r["user_id"] for r in all_rows)
    missing_users = selected_all - output_users
    if missing_users:
        print(f"[parallel] WARNING: {len(missing_users)} users missing from output!")
    else:
        print(f"[parallel] Coverage OK: all {len(selected_all)} selected users present")

    # Duplicate user×item check
    uid_item_pairs = [(r["user_id"], r["item_id"]) for r in all_rows]
    n_dupes = len(uid_item_pairs) - len(set(uid_item_pairs))
    if n_dupes:
        print(f"[parallel] WARNING: {n_dupes} duplicate (user_id, item_id) pairs!")
    else:
        print(f"[parallel] Integrity OK: no duplicate (user_id, item_id) pairs")

    # Write manifest
    manifest = write_manifest(
        canonical_path, manifest_path,
        row_count=row_count,
        user_count=user_count,
        prompt_version=prompt_version,
        cap=args.cap,
    )
    print(f"[parallel] Manifest → {manifest_path}")
    print(f"  row_count={manifest['row_count']}, user_count={manifest['user_count']}, "
          f"md5={manifest['md5']}, prompt_version={manifest['prompt_version']}, cap={manifest['cap']}")

    # Summary report
    total_records = sum(s["n_records"] for s in worker_stats)
    total_eligible = sum(s["n_eligible"] for s in worker_stats)
    total_cache = sum(s["n_cache_hit"] for s in worker_stats)
    report = {
        "variant": args.variant,
        "n_users": user_count,
        "n_records": row_count,
        "n_eligible": total_eligible,
        "eligible_rate": round(total_eligible / max(row_count, 1), 4),
        "cache_hit_rate": round(total_cache / max(row_count, 1), 4),
        "cap": args.cap,
        "prompt_version": prompt_version,
        "canonical_path": canonical_path,
        "manifest": manifest,
        "parallel": {
            "n_workers": args.n_workers,
            "wall_time_s": round(elapsed, 1),
            "worker_stats": worker_stats,
        },
    }
    report_path = os.path.join(args.report_dir, f"p1_extract_report.json")
    try:
        os.makedirs(args.report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[parallel] Report → {report_path}")
    except OSError as exc:
        print(f"[parallel] WARNING: could not write report to {report_path}: {exc}")

    print(f"\n{'='*60}")
    print(f"P1 EXTRACTION COMPLETE  (cap={args.cap}, variant={args.variant})")
    print(f"{'='*60}")
    print(f"  wall_time             : {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print(f"  n_records             : {row_count}")
    print(f"  n_users               : {user_count}")
    print(f"  eligible_rate         : {report['eligible_rate']:.3f}")
    print(f"  cache_hit_rate        : {report['cache_hit_rate']:.3f}")
    print(f"  canonical             : {canonical_path}")
    print(f"  manifest_md5          : {manifest['md5']}")


if __name__ == "__main__":
    main()
