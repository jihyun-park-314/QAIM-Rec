"""Prompt-level smoke: run P1 compact Books on N random eligible reviews.

Reports parse rate, discriminative rate, grounding distribution, leakage, latency.
Supports A/B comparison on the same sample set.

Usage:
  python -m src.memory.run_prompt_smoke \\
      --category Books \\
      --n 100 \\
      --variants A,B \\
      --same_samples \\
      --seed 42 \\
      --output_dir data/p1_smoke \\
      --report_dir reports
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from src.llm.client import LLMClient, load_llm_config
from src.memory.pipeline import (
    P1BooksRecord,
    extract_record,
    is_eligible,
    check_leakage,
)


# ---------------------------------------------------------------------------
# Config

@dataclass
class SmokeConfig:
    category: str = "Books"
    n: int = 100
    variants: list = field(default_factory=lambda: ["A"])
    same_samples: bool = True
    seed: int = 42
    llm_config_path: str = "configs/llm/p1.yaml"
    data_dir: str = "data/processed"
    output_dir: str = "data/p1_smoke"
    report_dir: str = "reports"


# ---------------------------------------------------------------------------
# Data loading

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


def sample_records(all_records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(all_records):
        return list(all_records)
    return rng.sample(all_records, n)


# ---------------------------------------------------------------------------
# Per-variant report

def compute_report(
    records: list[P1BooksRecord],
    variant: str,
    full_scale: int = 62607,
) -> dict:
    n = len(records)
    if n == 0:
        return {"variant": variant, "n": 0}

    parsed = [r for r in records if not r.parse_failed]
    failed = [r for r in records if r.parse_failed]
    discriminative = [r for r in parsed if r.is_discriminative]
    eligible = [r for r in records if r.eligible]
    grounding_counts = Counter(r.grounding_level for r in parsed)
    metadata_dom = [r for r in parsed if r.grounding_level == "metadata_dominant"]

    miss_latencies = [r.latency_s for r in records if not r.cache_hit and r.latency_s > 0]
    hit_latencies = [r.latency_s for r in records if r.cache_hit]

    mean_miss = statistics.mean(miss_latencies) if miss_latencies else 0.0
    estimated_full_s = mean_miss * full_scale
    estimated_full_h = estimated_full_s / 3600

    # Evidence span validity: non-empty list with at least 1 non-empty string
    valid_evidence = [
        r for r in parsed
        if r.evidence_span and any((s or "").strip() for s in r.evidence_span)
    ]

    leakage_cases = [r for r in parsed if r.leakage_detected]

    # Good examples: eligible, diverse grounding
    good_ex = [r for r in eligible if not r.leakage_detected][:5]
    bad_ex = [r for r in records if r.parse_failed or not r.is_discriminative][:5]

    # Approximate token counts (chars/4 for input, output estimate)
    input_tokens = [r.input_tokens_approx for r in records if r.input_tokens_approx > 0]
    mean_input_tokens = statistics.mean(input_tokens) if input_tokens else 0

    def _rec_summary(r: P1BooksRecord) -> dict:
        return {
            "user_id": r.user_id,
            "item_title": r.item_title,
            "contextual_intent": r.contextual_intent,
            "preference_summary": r.preference_summary,
            "evidence_span": r.evidence_span,
            "is_discriminative": r.is_discriminative,
            "grounding_level": r.grounding_level,
            "eligible": r.eligible,
            "leakage": r.leakage_detected,
            "parse_failed": r.parse_failed,
            "review_excerpt": r.review_text_original[:200],
        }

    return {
        "variant": variant,
        "n": n,
        "parse_success_rate": len(parsed) / n,
        "parse_failed_n": len(failed),
        "discriminative_rate": len(discriminative) / max(len(parsed), 1),
        "eligible_rate": len(eligible) / n,
        "grounding_level_distribution": dict(grounding_counts),
        "metadata_dominant_rate": len(metadata_dom) / max(len(parsed), 1),
        "metadata_dominant_n": len(metadata_dom),
        "evidence_span_valid_rate": len(valid_evidence) / max(len(parsed), 1),
        "leakage_n": len(leakage_cases),
        "leakage_rate": len(leakage_cases) / max(len(parsed), 1),
        "latency": {
            "n_cache_miss": len(miss_latencies),
            "n_cache_hit": len(hit_latencies),
            "mean_miss_s": round(mean_miss, 3),
            "estimated_full_calls_s": round(estimated_full_s, 1),
            "estimated_full_calls_h": round(estimated_full_h, 2),
        },
        "avg_input_tokens_approx": round(mean_input_tokens, 1),
        "good_examples": [_rec_summary(r) for r in good_ex],
        "bad_examples": [_rec_summary(r) for r in bad_ex],
        "leakage_examples": [_rec_summary(r) for r in leakage_cases[:5]],
        "metadata_dominant_examples": [_rec_summary(r) for r in metadata_dom[:5]],
    }


# ---------------------------------------------------------------------------
# Main

def run_smoke(cfg: SmokeConfig) -> dict:
    all_records = load_candidates(cfg.data_dir, cfg.category)
    sample = sample_records(all_records, cfg.n, cfg.seed)
    print(f"[smoke] {len(sample)} samples from {len(all_records)} candidates, seed={cfg.seed}")

    llm_cfg = load_llm_config(cfg.llm_config_path)
    client = LLMClient(llm_cfg)

    results: dict[str, list[P1BooksRecord]] = {}

    for variant in cfg.variants:
        print(f"\n[smoke] === Variant {variant} ===")
        recs = []
        t0 = time.perf_counter()
        for i, item in enumerate(sample):
            rec = extract_record(client, item, variant=variant)
            recs.append(rec)
            status = "OK" if not rec.parse_failed else "FAIL"
            disc = "disc" if rec.is_discriminative else "generic"
            gl = rec.grounding_level[:12] if rec.grounding_level else "?"
            leak = " LEAK!" if rec.leakage_detected else ""
            print(
                f"  [{i+1:3d}/{len(sample)}] {status} {disc:7s} {gl:20s} "
                f"{'cache' if rec.cache_hit else f'{rec.latency_s:.1f}s':>8}{leak}"
            )
        elapsed = time.perf_counter() - t0
        print(f"[smoke] Variant {variant}: {elapsed:.1f}s total, "
              f"{sum(1 for r in recs if not r.parse_failed)}/{len(recs)} parsed")
        results[variant] = recs

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.report_dir, exist_ok=True)

    reports = {}
    for variant, recs in results.items():
        report = compute_report(recs, variant)
        reports[variant] = report
        out_path = os.path.join(cfg.output_dir, f"smoke_{variant.lower()}_n{cfg.n}_seed{cfg.seed}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                row = {
                    "user_id": r.user_id, "item_id": r.item_id,
                    "item_title": r.item_title, "variant": r.variant,
                    "contextual_intent": r.contextual_intent,
                    "preference_summary": r.preference_summary,
                    "evidence_span": r.evidence_span,
                    "is_discriminative": r.is_discriminative,
                    "grounding_level": r.grounding_level,
                    "eligible": r.eligible,
                    "leakage_detected": r.leakage_detected,
                    "parse_failed": r.parse_failed,
                    "latency_s": r.latency_s,
                    "cache_hit": r.cache_hit,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[smoke] Saved records → {out_path}")

    report_path = os.path.join(
        cfg.report_dir, f"smoke_report_{'_'.join(cfg.variants)}_n{cfg.n}_seed{cfg.seed}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"[smoke] Report → {report_path}")

    client.close()
    return reports


def _print_summary(reports: dict) -> None:
    for variant, r in reports.items():
        print(f"\n{'='*60}")
        print(f"VARIANT {variant}  (n={r['n']})")
        print(f"{'='*60}")
        print(f"  parse_success_rate     : {r['parse_success_rate']:.3f}")
        print(f"  discriminative_rate    : {r['discriminative_rate']:.3f}")
        print(f"  eligible_rate          : {r['eligible_rate']:.3f}")
        print(f"  metadata_dominant_rate : {r['metadata_dominant_rate']:.3f}")
        print(f"  grounding distribution : {r['grounding_level_distribution']}")
        print(f"  evidence_valid_rate    : {r['evidence_span_valid_rate']:.3f}")
        print(f"  leakage_n              : {r['leakage_n']}")
        print(f"  mean_miss_latency_s    : {r['latency']['mean_miss_s']}")
        print(f"  est. full 62607 calls  : {r['latency']['estimated_full_calls_h']}h")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Books")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--variants", default="A,B")
    parser.add_argument("--same_samples", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm_config_path", default="configs/llm/p1.yaml")
    parser.add_argument("--output_dir", default="data/p1_smoke")
    parser.add_argument("--report_dir", default="reports")
    args = parser.parse_args()

    cfg = SmokeConfig(
        category=args.category,
        n=args.n,
        variants=[v.strip() for v in args.variants.split(",")],
        same_samples=args.same_samples,
        seed=args.seed,
        llm_config_path=args.llm_config_path,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )

    reports = run_smoke(cfg)
    _print_summary(reports)


if __name__ == "__main__":
    main()
