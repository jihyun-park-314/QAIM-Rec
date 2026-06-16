"""Pilot 1 — Intent extraction A/B comparison (plan.md §4 Stage P1, updated).

Two prompt versions run on identical samples from identical domains:
  p1_base   — purpose-centric narrow baseline (plan.md §5 P1 schema, unchanged)
  p1_aspect — domain-general contextual preference schema (new)

Run one (category, prompt_version) pair at a time; collate results manually.

Usage:
  python -m src.pilot.pilot1_intent_extraction --config configs/pilot/pilot1_base.yaml
  python -m src.pilot.pilot1_intent_extraction --config configs/pilot/pilot1_aspect.yaml

N=200 is forbidden without explicit approval.  Default smoke N=3, comparison N=20.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.data.sample import sample_review_pairs
from src.llm.client import LLMClient, load_llm_config
from src.llm.prompts import p1_intent, p1_aspect
from src.pilot.common import load_config, write_report

_N_HARD_LIMIT = 50


@dataclass
class Pilot1Config:
    """Config for Pilot 1 A/B (plan.md §4 Stage P1).

    prompt_version: "p1_base" | "p1_aspect"
    category: must exist under data/raw/
    n_samples: smoke=3, comparison=20; N=200 requires explicit approval.
    domain_type: informational label ("functional" | "lifestyle" | "content")
    """

    category: str = "Office_Products"
    prompt_version: str = "p1_base"
    domain_type: str = "functional"
    n_samples: int = 3
    seed: int = 42

    llm_config_path: str = "configs/llm/p1.yaml"
    retry_max: int = 2
    data_dir: str = "data/raw"

    output_path: str = "results/pilot/pilot1_report.json"


@dataclass
class LatencyStats:
    n_cache_miss: int = 0
    n_cache_hit: int = 0
    mean_miss_latency_s: float = 0.0
    median_miss_latency_s: float = 0.0
    mean_hit_latency_s: float = 0.0
    estimated_n200_time_s: float = 0.0


@dataclass
class Pilot1Report:
    """Pilot 1 results — all metrics per (category, prompt_version) cell."""

    category: str
    domain_type: str
    prompt_version: str
    n_samples_requested: int
    n_total: int
    n_parsed: int
    parse_success_rate: float

    # discriminative
    n_discriminative: int
    discriminative_ratio: float

    # aspect coverage (p1_aspect only; -1.0 = not applicable)
    aspect_coverage_ratio: float
    discriminative_and_aspect_valid_ratio: float

    # disposition note
    disposition_note_nonnull_ratio: float

    # intent length (p1_base: "purpose"; p1_aspect: "contextual_intent[0]")
    intent_length_stats: dict[str, float]

    # latency
    latency: LatencyStats

    # failed sample indices (0-based)
    failed_samples: list[int]

    # raw LLM outputs for manual inspection (2-3 samples)
    raw_outputs: list[dict[str, Any]]

    # go/no-go (N=200 gate — directional at N=20, confirmed at N=200)
    go_nogo: bool
    notes: str


def _length_stats(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    lengths = [len(t.split()) for t in texts]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 2),
        "median": round(statistics.median(lengths), 2),
    }


def run_pilot1(config: Pilot1Config) -> Pilot1Report:
    if config.n_samples > _N_HARD_LIMIT:
        raise ValueError(
            f"n_samples={config.n_samples} exceeds hard limit {_N_HARD_LIMIT}. "
            "N=200 requires explicit user approval — set n_samples <= 50 for pilots."
        )

    samples = sample_review_pairs(
        config.category,
        config.n_samples,
        config.seed,
        data_dir=config.data_dir,
    )
    n_total = len(samples)

    llm_cfg = load_llm_config(config.llm_config_path)
    llm_cfg.retry_max = config.retry_max
    client = LLMClient(llm_cfg)

    use_aspect = config.prompt_version == "p1_aspect"

    results: list[dict | None] = []
    miss_latencies: list[float] = []
    hit_count = 0
    failed: list[int] = []
    raw_outputs: list[dict] = []

    for i, item in enumerate(samples):
        if use_aspect:
            parsed, lat, hit = p1_aspect.run_p1_aspect(client, item, config.retry_max)
        else:
            parsed, lat, hit = p1_intent.run_p1(client, item, config.retry_max)

        results.append(parsed)

        if hit:
            hit_count += 1
        else:
            miss_latencies.append(lat)

        if parsed is None:
            failed.append(i)

        if len(raw_outputs) < 3 and parsed is not None:
            raw_outputs.append({"sample_index": i, "item_title": item.get("title"), "result": parsed})

    client.close()

    n_parsed = sum(1 for r in results if r is not None)
    parsed_results = [r for r in results if r is not None]

    # discriminative
    n_disc = sum(1 for r in parsed_results if r.get("is_discriminative"))
    disc_ratio = n_disc / n_parsed if n_parsed else 0.0

    # aspect coverage (p1_aspect only)
    if use_aspect:
        n_ac = sum(1 for r in parsed_results if r.get("_aspect_coverage_valid"))
        ac_ratio = n_ac / n_parsed if n_parsed else 0.0
        n_disc_and_ac = sum(
            1 for r in parsed_results
            if r.get("is_discriminative") and r.get("_aspect_coverage_valid")
        )
        disc_and_ac_ratio = n_disc_and_ac / n_parsed if n_parsed else 0.0
    else:
        ac_ratio = -1.0
        disc_and_ac_ratio = -1.0

    # disposition note
    dn_nonnull = sum(1 for r in parsed_results if r.get("disposition_note") is not None)
    dn_ratio = dn_nonnull / n_parsed if n_parsed else 0.0

    # intent text length
    if use_aspect:
        intent_texts = [
            r["contextual_intent"][0]
            for r in parsed_results
            if r.get("contextual_intent")
        ]
    else:
        intent_texts = [
            r["purpose"]
            for r in parsed_results
            if r.get("purpose")
        ]
    length_stats = _length_stats(intent_texts)

    # latency
    mean_miss = statistics.mean(miss_latencies) if miss_latencies else 0.0
    median_miss = statistics.median(miss_latencies) if miss_latencies else 0.0
    mean_hit = 0.0
    lat_stats = LatencyStats(
        n_cache_miss=len(miss_latencies),
        n_cache_hit=hit_count,
        mean_miss_latency_s=round(mean_miss, 3),
        median_miss_latency_s=round(median_miss, 3),
        mean_hit_latency_s=round(mean_hit, 3),
        estimated_n200_time_s=round(mean_miss * 200, 1),
    )

    # go/no-go (directional at N=20; confirmed at N=200)
    parse_rate = n_parsed / n_total if n_total else 0.0
    go = parse_rate >= 0.95 and disc_ratio >= 0.40
    note_parts = [
        f"category={config.category}",
        f"prompt={config.prompt_version}",
        f"N={n_total}",
        f"parse={parse_rate:.0%}",
        f"disc={disc_ratio:.0%}",
    ]
    if use_aspect:
        note_parts.append(f"aspect_valid={ac_ratio:.0%}")
    note_parts.append("NOTE: N=20 is directional only; go/no-go at N=200 (needs approval).")
    notes = " | ".join(note_parts)

    return Pilot1Report(
        category=config.category,
        domain_type=config.domain_type,
        prompt_version=config.prompt_version,
        n_samples_requested=config.n_samples,
        n_total=n_total,
        n_parsed=n_parsed,
        parse_success_rate=round(parse_rate, 4),
        n_discriminative=n_disc,
        discriminative_ratio=round(disc_ratio, 4),
        aspect_coverage_ratio=round(ac_ratio, 4),
        discriminative_and_aspect_valid_ratio=round(disc_and_ac_ratio, 4),
        disposition_note_nonnull_ratio=round(dn_ratio, 4),
        intent_length_stats=length_stats,
        latency=lat_stats,
        failed_samples=failed,
        raw_outputs=raw_outputs,
        go_nogo=go,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot 1 A/B: intent extraction (plan.md §4 Stage P1)"
    )
    parser.add_argument("--config", required=True, help="Path to Pilot 1 YAML config")
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Override n_samples (e.g. 3 for smoke test)"
    )
    args = parser.parse_args()

    config = load_config(args.config, Pilot1Config)
    if args.n_samples is not None:
        config.n_samples = args.n_samples

    report = run_pilot1(config)
    write_report(report, config.output_path)
    print(f"Report written to {config.output_path}")
    print(f"  parse_success_rate: {report.parse_success_rate:.1%}")
    print(f"  discriminative_ratio: {report.discriminative_ratio:.1%}")
    if report.prompt_version == "p1_aspect":
        print(f"  aspect_coverage_ratio: {report.aspect_coverage_ratio:.1%}")
        print(f"  disc_and_aspect_valid: {report.discriminative_and_aspect_valid_ratio:.1%}")
    print(f"  cache_miss latency: mean={report.latency.mean_miss_latency_s:.2f}s "
          f"median={report.latency.median_miss_latency_s:.2f}s")
    print(f"  estimated N=200 time: {report.latency.estimated_n200_time_s:.0f}s "
          f"(cache-miss mean × 200; do NOT run N=200 without approval)")
    print(f"  go_nogo (directional): {report.go_nogo}")


if __name__ == "__main__":
    main()
