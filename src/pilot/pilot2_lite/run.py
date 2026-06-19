"""Pilot 2-Lite main runner.

Executes Tasks 1, 2-A, 2-B, Clustering-lite, and 2-C in sequence.
Saves results to results/pilot/pilot2_lite/.

Usage:
  python3 -m src.pilot.pilot2_lite.run --task audit
  python3 -m src.pilot.pilot2_lite.run --task funnel_sweep
  python3 -m src.pilot.pilot2_lite.run --task funnel_2b   # stream once → count5+ge4+ge10w IDs
  python3 -m src.pilot.pilot2_lite.run --task smoke       # LLM: 10 users × 3 domains × ≤8 interactions
  python3 -m src.pilot.pilot2_lite.run --task cluster     # LLM: embedding + agglomerative
  python3 -m src.pilot.pilot2_lite.run --task ablation    # LLM: ~20 short reviews × 3 domains
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set HF cache to project-local path BEFORE importing sentence-transformers
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
_HF_CACHE = os.path.join(_PROJECT_ROOT, ".cache", "huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _HF_CACHE)
os.makedirs(_HF_CACHE, exist_ok=True)

from src.pilot.pilot2_lite.audit import run_books_audit
from src.pilot.pilot2_lite.funnel import (
    collect_eligible_user_ids,
    collect_user_interactions,
    load_funnel_cache,
    pick_eligible_users,
    run_funnel,
    run_funnel_sweep,
    save_funnel_cache,
)
from src.pilot.pilot2_lite.ablation import (
    collect_short_reviews,
    estimate_ablation_calls,
    run_ablation_for_domain,
)
from src.pilot.pilot2_lite.smoke import estimate_calls, run_smoke_for_domain
from src.pilot.pilot2_lite.clustering import (
    build_representative_examples,
    compute_threshold_stats,
    run_clustering,
)


_DOMAINS = [
    ("Amazon_Fashion", "lifestyle"),
    ("Books", "content"),
    ("Beauty_and_Personal_Care", "lifestyle"),
]
_THRESHOLDS = [0.25, 0.35, 0.45]
_N_USERS_PER_DOMAIN = 10
_MAX_INTERACTIONS_PER_USER = 8
_SEED = 42
_OUTPUT_DIR = "results/pilot/pilot2_lite"
_DATA_DIR = "data/raw"


def _write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")


def _strip_internal(d: dict) -> dict:
    """Remove keys starting with _ before saving."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(args: argparse.Namespace) -> None:
    t_global_start = time.time()
    run_task = args.task  # None = all

    # ── Task 1: Books composition audit ──────────────────────────────────
    if run_task in (None, "audit"):
        print("\n=== Task 1: Books composition audit ===")
        pilot1_books_asins = [
            '1732156808', '0736427163', '0805087761', '0805087761',
            '0312509200', '0805087761', '0972842136', '1476746583',
            '059342297X', '1735311529', '1477824758', '1910884014',
            '0688098312', '0312509219', '1641849517', 'B002KE5SPG',
            '0312509529', 'B0BF1YNRGR', '1452152160', '0805210954',
        ]
        pilot1_books_asins = list(dict.fromkeys(pilot1_books_asins))  # deduplicate

        t0 = time.time()
        audit = run_books_audit(
            data_dir=_DATA_DIR,
            max_reviews=50_000,
            pilot1_asins=pilot1_books_asins,
        )
        audit["elapsed_seconds"] = round(time.time() - t0, 1)
        _write_json(f"{_OUTPUT_DIR}/books_composition_audit.json", audit)

        print(f"  scanned_reviews     : {audit['scanned_review_count']:,}")
        print(f"  unique_users        : {audit['unique_user_count']:,}")
        print(f"  unique_asins        : {audit['unique_parent_asin_count']:,}")
        print(f"  review length       : median={audit['review_length_tokens']['median']} "
              f"p25={audit['review_length_tokens']['p25']} p75={audit['review_length_tokens']['p75']}")
        cr = audit['children_related']
        print(f"  children_asin_ratio : {cr['children_asins_ratio_among_scanned']:.1%}  "
              f"({cr['children_asins_count']}/{audit['unique_parent_asin_count']})")
        print(f"  children_review_ratio: {cr['children_review_ratio']:.1%}  "
              f"({cr['children_review_count']}/{audit['scanned_review_count']})")
        p1c = audit['pilot1_aspect_asins_children_check']
        print(f"  pilot1_asins children: {p1c['children_ratio']:.1%} "
              f"({p1c['pilot1_asins_count']} asins)")

    # ── Task 2-A (sweep): Eligibility funnel 3-axis sweep ────────────────
    if run_task in (None, "funnel_sweep"):
        print("\n=== Task 2-A: Eligibility funnel 3-axis sweep ===")
        sweep_results: dict[str, dict] = {}
        for domain, _ in _DOMAINS:
            t0 = time.time()
            sweep = run_funnel_sweep(domain, data_dir=_DATA_DIR)
            sweep["elapsed_seconds"] = round(time.time() - t0, 1)
            sweep_results[domain] = sweep
            print(f"  {domain}: total_users={sweep['total_users']:,}  elapsed={sweep['elapsed_seconds']}s")
            for rg, tbl in sweep["rating_tables"].items():
                print(f"    rating={rg}:")
                for ck, cell in tbl["cells"].items():
                    print(f"      {ck}: eligible={cell['eligible_users']:,}  ratio={cell['eligibility_ratio']:.4%}"
                          f"  hist_median={cell['history_len_median']}  p25={cell['history_len_p25']}  p75={cell['history_len_p75']}")

        _write_json(f"{_OUTPUT_DIR}/eligibility_funnel_sweep.json", sweep_results)
        if run_task == "funnel_sweep":
            print("\n=== funnel_sweep done ===")
            return

    # ── Task 2-B pre-step: collect count5+ge4+ge10w user IDs ─────────────
    # Smoke uses this pool (not the old count≥8/no-rating funnel cache).
    # Cached to funnel_2b_cache_{domain}.json; re-streams only if missing.
    funnel_2b_user_ids: dict[str, list[str]] = {}

    if run_task in (None, "funnel_2b", "smoke", "cluster"):
        print("\n=== Task 2-B pre-step: count5+ge4+ge10w user IDs ===")
        for domain, _ in _DOMAINS:
            cache_path = f"{_OUTPUT_DIR}/funnel_2b_cache_{domain}.json"
            if Path(cache_path).is_file():
                with open(cache_path, encoding="utf-8") as f:
                    cached = json.load(f)
                funnel_2b_user_ids[domain] = cached["eligible_user_ids"]
                print(f"  {domain}: loaded {len(funnel_2b_user_ids[domain]):,} IDs from cache")
            else:
                t0 = time.time()
                ids = collect_eligible_user_ids(
                    domain, count_threshold=5, min_words=10, min_rating=4.0,
                    data_dir=_DATA_DIR,
                )
                elapsed = round(time.time() - t0, 1)
                funnel_2b_user_ids[domain] = ids
                os.makedirs(_OUTPUT_DIR, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"domain": domain, "count_threshold": 5,
                               "min_words": 10, "min_rating": 4.0,
                               "eligible_user_ids": ids}, f, ensure_ascii=False)
                print(f"  {domain}: {len(ids):,} eligible users  (elapsed={elapsed}s)  [saved {cache_path}]")
        if run_task == "funnel_2b":
            return

    # ── Task 2-A: Eligibility funnel ─────────────────────────────────────
    funnel_results: dict[str, dict] = {}

    if run_task in (None, "funnel", "smoke", "cluster"):
        print("\n=== Task 2-A: Eligibility funnel ===")
        for domain, _ in _DOMAINS:
            cache_path = f"{_OUTPUT_DIR}/funnel_cache_{domain}.json"
            if run_task != "funnel":
                cached = load_funnel_cache(cache_path)
                if cached is not None:
                    funnel_results[domain] = cached
                    r = cached
                    print(f"  {domain}: loaded from cache ({cache_path})")
                    print(f"    total_users       : {r['total_users']:,}")
                    print(f"    eligible_ge_8     : {r['users_with_eligible_ge_8']:,}")
                    print(f"    eligibility_ratio : {r['eligibility_ratio']:.2%}")
                    h = r['eligible_history_length_stats']
                    print(f"    history_len       : median={h['median']} p25={h['p25']} p75={h['p75']}")
                    continue

            t0 = time.time()
            result = run_funnel(domain, data_dir=_DATA_DIR)
            result["elapsed_seconds"] = round(time.time() - t0, 1)
            funnel_results[domain] = result
            save_funnel_cache(result, cache_path)
            r = result
            print(f"  {domain}:")
            print(f"    total_users       : {r['total_users']:,}")
            print(f"    eligible_ge_8     : {r['users_with_eligible_ge_8']:,}")
            print(f"    eligibility_ratio : {r['eligibility_ratio']:.2%}")
            h = r['eligible_history_length_stats']
            print(f"    history_len       : median={h['median']} p25={h['p25']} p75={h['p75']}")

    # ── Task 2-B: LLM smoke ───────────────────────────────────────────────
    smoke_results_per_domain: dict[str, list[dict]] = {}

    if run_task in (None, "smoke", "cluster") and not args.skip_llm:
        print("\n=== Task 2-B: LLM smoke ===")

        # Collect user data first (to estimate calls).
        # 2-B user pool: count≥5 + rating≥4 + ge10w (from funnel_2b_user_ids).
        # Interactions filtered to rating≥4 + ge10w (min_rating=4.0).
        import random as _random
        _rng = _random.Random(_SEED)
        user_data_per_domain: dict[str, list[dict]] = {}
        for domain, domain_type in _DOMAINS:
            pool = funnel_2b_user_ids.get(domain) or []
            selected_ids = _rng.sample(pool, min(_N_USERS_PER_DOMAIN, len(pool)))
            print(f"  [smoke] {domain}: {len(selected_ids)} users selected"
                  f" from pool={len(pool):,} (count5+ge4+ge10w)")
            user_data = collect_user_interactions(
                domain, selected_ids, {},
                max_per_user=_MAX_INTERACTIONS_PER_USER,
                min_rating=4.0,
                data_dir=_DATA_DIR,
            )
            user_data_per_domain[domain] = user_data

        # Estimate
        est = estimate_calls(user_data_per_domain)
        print(f"\n  Estimated LLM calls: {est['total_estimated_calls']} "
              f"(~{est['estimated_minutes']} min)")
        for d, n in est['per_domain'].items():
            print(f"    {d}: {n} calls")

        if est['total_estimated_calls'] > 300:
            print(f"  WARNING: {est['total_estimated_calls']} calls exceeds safety limit 300. "
                  "Truncating to 240 max.")

        # Run LLM
        for domain, domain_type in _DOMAINS:
            users = user_data_per_domain.get(domain, [])
            print(f"\n  Running p1_aspect for {domain} ({len(users)} users) ...")
            t0 = time.time()
            domain_smoke = run_smoke_for_domain(domain, users, domain_type=domain_type)
            elapsed = round(time.time() - t0, 1)
            smoke_results_per_domain[domain] = domain_smoke

            n_calls = sum(u["selected_interaction_count"] for u in users)
            n_parsed = sum(
                sum(1 for it in u["interactions"] if it["parse_success"])
                for u in domain_smoke
            )
            print(f"    {domain}: {n_calls} calls, {n_parsed} parsed, {elapsed}s")

    # ── Clustering-lite ───────────────────────────────────────────────────
    if run_task in (None, "cluster") and not args.skip_llm:
        print("\n=== Clustering-lite ===")

        for domain, domain_type in _DOMAINS:
            if domain not in smoke_results_per_domain:
                print(f"  [cluster] No smoke results for {domain}, skipping")
                continue

            smoke = smoke_results_per_domain[domain]
            eligibility_ratio = funnel_results.get(domain, {}).get("eligibility_ratio", 0.0)

            cluster_result = run_clustering(smoke, thresholds=_THRESHOLDS)
            threshold_stats = compute_threshold_stats(
                cluster_result, eligibility_ratio, thresholds=_THRESHOLDS
            )
            rep_examples = build_representative_examples(
                cluster_result, n_users=3, threshold=0.35
            )

            # K stability note
            k_vals_by_threshold = {}
            for t in _THRESHOLDS:
                k_vals_by_threshold[str(t)] = [
                    u["cluster_results"][str(t)]["k_personal"]
                    for u in cluster_result["per_user"]
                ]
            k_stability = "K_personal across thresholds: " + " | ".join(
                f"t={t}: avg={threshold_stats[str(t)]['avg_k_personal']}"
                for t in _THRESHOLDS if str(t) in threshold_stats
            )

            # Per-user summary (save without raw texts to keep JSON small)
            per_user_summary = []
            for u in cluster_result["per_user"]:
                per_user_summary.append({
                    "user_id": u["user_id"],
                    "n_discriminative": u["n_discriminative"],
                    "n_aspect_valid": u["n_aspect_valid"],
                    "k_personal_by_threshold": {
                        str(t): u["cluster_results"][str(t)]["k_personal"]
                        for t in _THRESHOLDS
                    },
                })

            domain_report = {
                "domain": domain,
                "eligibility_ratio": eligibility_ratio,
                "embedding_method": cluster_result["embedding_method"],
                "n_sampled_users": len(smoke),
                "threshold_stats": threshold_stats,
                "k_stability_note": k_stability,
                "per_user_summary": per_user_summary,
                "representative_examples": rep_examples,
                "funnel_stats": _strip_internal(funnel_results.get(domain, {})),
                "smoke_call_estimate": estimate_calls({domain: smoke}),
            }
            _write_json(f"{_OUTPUT_DIR}/pilot2_lite_{domain}.json", domain_report)

            # Print stats
            print(f"\n  {domain} | eligibility_ratio={eligibility_ratio:.2%}")
            print(f"    embedding_method: {cluster_result['embedding_method']}")
            for t in _THRESHOLDS:
                s = threshold_stats.get(str(t), {})
                print(f"    t={t}: avg_K={s.get('avg_k_personal')} "
                      f"K>=2_ratio={s.get('k_personal_ge2_ratio'):.1%} "
                      f"effective_cov={s.get('effective_coverage'):.4f}")

    # ── Task 2-C: Short-review ablation ──────────────────────────────────
    if run_task in (None, "ablation") and not args.skip_llm:
        print("\n=== Task 2-C: Short-review ablation ===")
        ablation_reviews: dict[str, list] = {}
        for domain, _ in _DOMAINS:
            reviews = collect_short_reviews(domain, n_target=20, seed=_SEED, data_dir=_DATA_DIR)
            ablation_reviews[domain] = reviews
            print(f"  {domain}: {len(reviews)} short reviews collected")

        est_abl = estimate_ablation_calls(ablation_reviews)
        print(f"\n  Estimated LLM calls: {est_abl['total_estimated_calls']}"
              f" (~{est_abl['estimated_minutes']} min)")

        for domain, domain_type in _DOMAINS:
            reviews = ablation_reviews.get(domain, [])
            if not reviews:
                continue
            print(f"\n  Running p1_aspect ablation for {domain} ({len(reviews)} reviews) ...")
            t0 = time.time()
            abl_results = run_ablation_for_domain(domain, reviews, domain_type=domain_type)
            elapsed = round(time.time() - t0, 1)
            n_parsed = sum(1 for r in abl_results if r["parse_success"])
            print(f"    {domain}: {len(reviews)} calls, {n_parsed} parsed, {elapsed}s")
            _write_json(f"{_OUTPUT_DIR}/short_review_ablation_{domain}.json", {
                "domain": domain,
                "n_reviews": len(abl_results),
                "n_parsed": n_parsed,
                "short_word_range": [3, 9],
                "min_rating": 4.0,
                "results": abl_results,
            })

        if run_task == "ablation":
            return

    # ── Summary ───────────────────────────────────────────────────────────
    if run_task in (None,):
        print("\n=== Writing summary ===")
        summary = {
            "pilot": "pilot2_lite",
            "thresholds": _THRESHOLDS,
            "n_users_per_domain": _N_USERS_PER_DOMAIN,
            "max_interactions_per_user": _MAX_INTERACTIONS_PER_USER,
            "seed": _SEED,
        }

        for domain, _ in _DOMAINS:
            funnel = funnel_results.get(domain, {})
            summary[domain] = {
                "total_users": funnel.get("total_users"),
                "eligibility_ratio": funnel.get("eligibility_ratio"),
                "eligible_history_stats": funnel.get("eligible_history_length_stats"),
            }

        total_elapsed = round(time.time() - t_global_start, 1)
        summary["total_elapsed_seconds"] = total_elapsed
        _write_json(f"{_OUTPUT_DIR}/pilot2_lite_summary.json", summary)
        print(f"\nTotal elapsed: {total_elapsed}s")

    print("\n=== Pilot 2-Lite complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot 2-Lite runner")
    parser.add_argument("--task", default=None,
                        choices=["audit", "funnel", "funnel_sweep", "funnel_2b",
                                 "smoke", "cluster", "ablation"],
                        help="Run only one task (default: all)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM calls (Task 2-B + clustering)")
    parser.add_argument("--data_dir", default="data/raw")
    args = parser.parse_args()
    if args.data_dir != "data/raw":
        _DATA_DIR = args.data_dir
    main(args)
