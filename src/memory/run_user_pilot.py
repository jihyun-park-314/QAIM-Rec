"""N-user memory construction pilot: P1 extract → filter → embed → cluster.

Runs the full pipeline for N users and reports K_personal distribution,
eligible rate, cluster quality, leakage counts.

Usage (single variant, after A/B smoke picks a winner):
  python -m src.memory.run_user_pilot \\
      --category Books \\
      --n_users 250 \\
      --eligible_min 2 \\
      --max_reviews_per_user 12 \\
      --variant A \\
      --seed 42 \\
      --output_dir data/memory_pilot \\
      --report_dir reports

Usage (50-user A/B subset when smoke was ambiguous):
  python -m src.memory.run_user_pilot \\
      --category Books \\
      --n_users 50 \\
      --eligible_min 2 \\
      --max_reviews_per_user 12 \\
      --variants A,B \\
      --same_users \\
      --seed 42 \\
      --output_dir data/memory_pilot_ab50 \\
      --report_dir reports
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.llm.client import LLMClient, load_llm_config
from src.memory.embed import EmbeddingModel
from src.memory.pipeline import (
    P1BooksRecord,
    extract_record,
    cluster_user_records,
)


# ---------------------------------------------------------------------------
# Config

@dataclass
class PilotConfig:
    category: str = "Books"
    n_users: int = 250
    eligible_min: int = 2
    max_reviews_per_user: int = 12
    variants: list = field(default_factory=lambda: ["A"])
    same_users: bool = True
    seed: int = 42
    tau: float = 0.3
    k_min: int = 1
    k_max: int = 5
    llm_config_path: str = "configs/llm/p1.yaml"
    embed_model_name: str = "BAAI/bge-base-en-v1.5"
    data_dir: str = "data/processed"
    output_dir: str = "data/memory_pilot"
    report_dir: str = "reports"


# ---------------------------------------------------------------------------
# Data loading and user grouping

def load_candidates(data_dir: str, category: str) -> list[dict]:
    path = os.path.join(data_dir, category, "books_memory_candidates.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Candidates not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_user(records: list[dict]) -> dict[Any, list[dict]]:
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["user_id"]].append(r)
    return dict(groups)


def select_users(
    user_groups: dict[Any, list[dict]],
    n_users: int,
    eligible_min: int,
    max_reviews: int,
    seed: int,
) -> list[Any]:
    """Select up to n_users users that have >= eligible_min reviews (capped at max_reviews)."""
    eligible_users = [u for u, recs in user_groups.items() if len(recs) >= eligible_min]
    rng = random.Random(seed)
    rng.shuffle(eligible_users)
    return eligible_users[:n_users]


# ---------------------------------------------------------------------------
# Per-user result

@dataclass
class UserResult:
    user_id: Any
    variant: str
    n_reviews_input: int = 0       # reviews fed to pipeline
    n_parse_success: int = 0
    n_eligible: int = 0
    k_personal: int = 0            # number of clusters
    n_leakage: int = 0
    latency_total_s: float = 0.0
    n_cache_hit: int = 0
    cluster_summaries: list = field(default_factory=list)
    eligible_records: list = field(default_factory=list)  # for export


# ---------------------------------------------------------------------------
# K_personal stratification: 0 / 1 / >=2

def _k_strata(user_results: list[UserResult]) -> dict:
    k0 = sum(1 for r in user_results if r.k_personal == 0)
    k1 = sum(1 for r in user_results if r.k_personal == 1)
    k2p = sum(1 for r in user_results if r.k_personal >= 2)
    n = len(user_results)
    return {
        "k0_n": k0, "k0_rate": k0 / max(n, 1),
        "k1_n": k1, "k1_rate": k1 / max(n, 1),
        "k2plus_n": k2p, "k2plus_rate": k2p / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Main pipeline for one user

def run_user_pipeline(
    client: LLMClient,
    emb_model: EmbeddingModel,
    user_id: Any,
    reviews: list[dict],
    variant: str,
    max_reviews: int,
    k_min: int,
    k_max: int,
    tau: float,
) -> UserResult:
    capped = reviews[:max_reviews]
    result = UserResult(user_id=user_id, variant=variant, n_reviews_input=len(capped))

    all_recs: list[P1BooksRecord] = []
    for item in capped:
        rec = extract_record(client, item, variant=variant)
        all_recs.append(rec)
        result.latency_total_s += rec.latency_s
        if rec.cache_hit:
            result.n_cache_hit += 1

    result.n_parse_success = sum(1 for r in all_recs if not r.parse_failed)
    eligible = [r for r in all_recs if r.eligible]
    result.n_eligible = len(eligible)
    result.n_leakage = sum(1 for r in all_recs if r.leakage_detected)

    if eligible:
        cluster_result = cluster_user_records(eligible, emb_model, k_min, k_max, tau)
        result.k_personal = cluster_result["k_personal"]
        result.cluster_summaries = cluster_result["cluster_summaries"]
        result.eligible_records = eligible

    return result


# ---------------------------------------------------------------------------
# Pilot report

def compute_pilot_report(
    user_results: list[UserResult],
    variant: str,
    full_scale_users: int = 9807,
) -> dict:
    n = len(user_results)
    if n == 0:
        return {"variant": variant, "n_users": 0}

    total_reviews = sum(r.n_reviews_input for r in user_results)
    total_parse = sum(r.n_parse_success for r in user_results)
    total_eligible = sum(r.n_eligible for r in user_results)
    total_leakage = sum(r.n_leakage for r in user_results)
    total_cache_hit = sum(r.n_cache_hit for r in user_results)

    k_dist = Counter(r.k_personal for r in user_results)
    k_strata = _k_strata(user_results)
    mean_k = statistics.mean(r.k_personal for r in user_results) if user_results else 0
    mean_eligible = total_eligible / max(n, 1)

    miss_latencies = [
        r.latency_total_s for r in user_results if r.latency_total_s > 0 and r.n_cache_hit < r.n_reviews_input
    ]
    mean_latency_per_user = statistics.mean(miss_latencies) if miss_latencies else 0.0

    # Estimated full-scale wall time (single process) at mean_miss seconds per user
    per_review_lat = [
        r.latency_total_s / max(r.n_reviews_input - r.n_cache_hit, 1)
        for r in user_results
        if (r.n_reviews_input - r.n_cache_hit) > 0
    ]
    mean_lat_per_review = statistics.mean(per_review_lat) if per_review_lat else 0.0

    # Export some user-level examples
    k0_examples = [
        {"user_id": r.user_id, "n_reviews": r.n_reviews_input, "n_eligible": r.n_eligible}
        for r in user_results if r.k_personal == 0
    ][:10]
    k2p_examples = [
        {
            "user_id": r.user_id, "k_personal": r.k_personal, "n_eligible": r.n_eligible,
            "cluster_sizes": [c["size"] for c in r.cluster_summaries],
            "cluster_intents": [[i[:80] for i in c["intents"]] for c in r.cluster_summaries],
        }
        for r in user_results if r.k_personal >= 2
    ][:5]

    return {
        "variant": variant,
        "n_users": n,
        "n_reviews_processed": total_reviews,
        "parse_success_rate": total_parse / max(total_reviews, 1),
        "eligible_rate_per_review": total_eligible / max(total_reviews, 1),
        "mean_eligible_per_user": round(mean_eligible, 2),
        "total_leakage": total_leakage,
        "leakage_rate": total_leakage / max(total_reviews, 1),
        "cache_hit_rate": total_cache_hit / max(total_reviews, 1),
        "k_personal_distribution": dict(sorted(k_dist.items())),
        "k_strata": k_strata,
        "mean_k_personal": round(mean_k, 3),
        "latency": {
            "mean_lat_per_review_s": round(mean_lat_per_review, 3),
            "mean_lat_per_user_s": round(mean_latency_per_user, 3),
        },
        "k0_examples": k0_examples,
        "k2plus_examples": k2p_examples,
    }


# ---------------------------------------------------------------------------
# Main

def run_pilot(cfg: PilotConfig) -> dict:
    all_records = load_candidates(cfg.data_dir, cfg.category)
    user_groups = group_by_user(all_records)
    selected_users = select_users(
        user_groups, cfg.n_users, cfg.eligible_min, cfg.max_reviews_per_user, cfg.seed
    )
    print(f"[pilot] {len(selected_users)} users selected (eligible_min={cfg.eligible_min}, "
          f"seed={cfg.seed}), max_reviews={cfg.max_reviews_per_user}")

    llm_cfg = load_llm_config(cfg.llm_config_path)
    client = LLMClient(llm_cfg)
    emb_model = EmbeddingModel(cfg.embed_model_name)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.report_dir, exist_ok=True)

    all_reports = {}

    for variant in cfg.variants:
        print(f"\n[pilot] === Variant {variant} ===")
        user_results: list[UserResult] = []

        t0 = time.perf_counter()
        for i, uid in enumerate(selected_users):
            reviews = user_groups[uid]
            ur = run_user_pipeline(
                client, emb_model, uid, reviews, variant,
                cfg.max_reviews_per_user, cfg.k_min, cfg.k_max, cfg.tau,
            )
            user_results.append(ur)
            k_str = f"K={ur.k_personal}" if ur.k_personal > 0 else "K=0(skip)"
            print(
                f"  [{i+1:4d}/{len(selected_users)}] uid={str(uid)[:12]:12s} "
                f"in={ur.n_reviews_input:2d} parse={ur.n_parse_success:2d} "
                f"elig={ur.n_eligible:2d} {k_str}"
            )

        elapsed = time.perf_counter() - t0
        print(f"[pilot] Variant {variant}: {elapsed:.1f}s total")

        # Save per-user records
        out_path = os.path.join(
            cfg.output_dir,
            f"pilot_{variant.lower()}_u{cfg.n_users}_seed{cfg.seed}.jsonl",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for ur in user_results:
                row = {
                    "user_id": ur.user_id,
                    "variant": ur.variant,
                    "n_reviews_input": ur.n_reviews_input,
                    "n_parse_success": ur.n_parse_success,
                    "n_eligible": ur.n_eligible,
                    "k_personal": ur.k_personal,
                    "n_leakage": ur.n_leakage,
                    "cluster_summaries": [
                        {
                            "label": c["label"],
                            "size": c["size"],
                            "intents": c["intents"],
                            "source_texts": c["source_texts"],
                        }
                        for c in ur.cluster_summaries
                    ],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[pilot] Saved user results → {out_path}")

        report = compute_pilot_report(user_results, variant)
        all_reports[variant] = report

    report_path = os.path.join(
        cfg.report_dir,
        f"pilot_report_{'_'.join(cfg.variants)}_u{cfg.n_users}_seed{cfg.seed}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"[pilot] Report → {report_path}")

    client.close()
    return all_reports


def _print_summary(reports: dict) -> None:
    for variant, r in reports.items():
        print(f"\n{'='*60}")
        print(f"VARIANT {variant}  (n_users={r['n_users']})")
        print(f"{'='*60}")
        print(f"  parse_success_rate          : {r['parse_success_rate']:.3f}")
        print(f"  eligible_rate_per_review    : {r['eligible_rate_per_review']:.3f}")
        print(f"  mean_eligible_per_user      : {r['mean_eligible_per_user']}")
        print(f"  mean_k_personal             : {r['mean_k_personal']}")
        print(f"  k_personal_distribution     : {r['k_personal_distribution']}")
        s = r["k_strata"]
        print(f"  K=0 rate                    : {s['k0_rate']:.3f} (n={s['k0_n']})")
        print(f"  K=1 rate                    : {s['k1_rate']:.3f} (n={s['k1_n']})")
        print(f"  K>=2 rate                   : {s['k2plus_rate']:.3f} (n={s['k2plus_n']})")
        print(f"  leakage_rate                : {r['leakage_rate']:.4f}")
        print(f"  cache_hit_rate              : {r['cache_hit_rate']:.3f}")
        lat = r["latency"]
        print(f"  mean_lat_per_review_s       : {lat['mean_lat_per_review_s']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Books")
    parser.add_argument("--n_users", type=int, default=250)
    parser.add_argument("--eligible_min", type=int, default=2)
    parser.add_argument("--max_reviews_per_user", type=int, default=12)
    # Accepts either --variant <single> or --variants <A,B>
    parser.add_argument("--variant", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--same_users", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--llm_config_path", default="configs/llm/p1.yaml")
    parser.add_argument("--embed_model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--output_dir", default="data/memory_pilot")
    parser.add_argument("--report_dir", default="reports")
    args = parser.parse_args()

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]
    elif args.variant:
        variants = [args.variant.strip()]
    else:
        variants = ["A"]

    cfg = PilotConfig(
        category=args.category,
        n_users=args.n_users,
        eligible_min=args.eligible_min,
        max_reviews_per_user=args.max_reviews_per_user,
        variants=variants,
        same_users=args.same_users,
        seed=args.seed,
        tau=args.tau,
        k_min=args.k_min,
        k_max=args.k_max,
        llm_config_path=args.llm_config_path,
        embed_model_name=args.embed_model_name,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )

    reports = run_pilot(cfg)
    _print_summary(reports)


if __name__ == "__main__":
    main()
