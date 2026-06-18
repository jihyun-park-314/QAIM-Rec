"""Pilot 1 — Intent extraction A/B comparison (plan.md §4 Stage P1).

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
import collections
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.llm.client import LLMClient, load_llm_config
from src.llm.prompts import p1_intent, p1_aspect
from src.pilot.common import load_config, write_report
from src.pilot.sample_cache import get_pilot1_sample

_N_HARD_LIMIT = 50


@dataclass
class Pilot1Config:
    """Config for Pilot 1 A/B (plan.md §4 Stage P1).

    prompt_version: "p1_base" | "p1_aspect"
    category: must exist under data/raw/
    n_samples: smoke=3, comparison=20; N=200 requires explicit approval.
    domain_type: informational label ("functional" | "lifestyle" | "content")
    """

    category: str = "Amazon_Fashion"
    prompt_version: str = "p1_base"
    domain_type: str = "lifestyle"
    n_samples: int = 3
    seed: int = 42

    llm_config_path: str = "configs/llm/p1.yaml"
    retry_max: int = 2
    data_dir: str = "data/raw"
    cache_dir: str = "data/processed"

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
class RetryStats:
    """Aggregate retry and failure tracking across all samples (plan.md stabilisation spec)."""
    retry_count: int = 0                  # total number of retries across all samples
    retry_rate: float = 0.0               # fraction of samples that used >= 1 retry
    empty_response_count: int = 0         # total empty-string responses from Ollama
    done_reason_length_count: int = 0     # total done_reason=="length" events
    parse_failure_count: int = 0          # total json.loads() failures
    schema_failure_count: int = 0         # total schema/content validation failures
    input_truncation_count: int = 0       # samples where review_text > 6000 chars was truncated
    no_signal_empty_intent_count: int = 0 # samples where contextual_intent == [] (no-signal)
    null_string_normalized_count: int = 0 # samples where ["null"] etc. were normalized to []
    # distribution of final max_new_tokens used (e.g. {"1200": 3, "4096": 0})
    final_max_new_tokens_distribution: dict = field(default_factory=dict)
    # per-sample latency list (cache-miss only)
    per_sample_latencies: list = field(default_factory=list)


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
    intent_length_stats: dict

    # latency
    latency: LatencyStats

    # retry / failure stats (plan.md stabilisation spec)
    retry_stats: RetryStats

    # per-sample detail (latency, retry, final token budget, done_reason)
    per_sample_detail: list

    # failed sample indices (0-based)
    failed_samples: list

    # parent_asins used — verify base==aspect for A/B validity
    sample_parent_asins: list

    # raw LLM outputs for manual inspection (up to 15 samples)
    raw_outputs: list

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

    samples = get_pilot1_sample(
        config.category,
        config.seed,
        config.n_samples,
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
    )
    n_total = len(samples)
    sample_asins = [s["parent_asin"] for s in samples]
    print(f"[A/B] category={config.category} n={n_total} seed={config.seed} "
          f"prompt={config.prompt_version}")
    print(f"[A/B] parent_asins: {sample_asins}")

    llm_cfg = load_llm_config(config.llm_config_path)
    llm_cfg.retry_max = config.retry_max

    use_aspect = config.prompt_version == "p1_aspect"
    llm_cfg.prompt_version = (p1_aspect if use_aspect else p1_intent).PROMPT_VERSION

    client = LLMClient(llm_cfg)

    miss_latencies: list[float] = []
    hit_count = 0
    failed: list[int] = []
    raw_outputs: list[dict] = []
    per_sample_detail: list[dict] = []

    # retry stats aggregation
    total_retry_count = 0
    n_samples_with_retry = 0
    total_empty = 0
    total_length = 0
    total_parse_fail = 0
    total_schema_fail = 0
    total_truncated = 0
    total_no_signal = 0
    total_null_normalized = 0
    token_dist: collections.Counter = collections.Counter()

    parsed_results: list[dict] = []
    all_results: list[dict | None] = []

    for i, item in enumerate(samples):
        if use_aspect:
            res = p1_aspect.run_p1_aspect(client, item, config.retry_max)
        else:
            res = p1_intent.run_p1(client, item, config.retry_max)

        all_results.append(res.parsed)

        # latency
        if res.cache_hit:
            hit_count += 1
        else:
            miss_latencies.append(res.latency_s)

        if res.parsed is None:
            failed.append(i)
        else:
            parsed_results.append(res.parsed)

        # retry aggregation
        total_retry_count += res.retry_count
        if res.retry_count > 0:
            n_samples_with_retry += 1
        total_empty += res.empty_response_count
        total_length += res.done_reason_length_count
        total_parse_fail += res.parse_failure_count
        total_schema_fail += res.schema_failure_count
        if res.truncated_input:
            total_truncated += 1
        if res.parsed is not None:
            if res.parsed.get("contextual_intent") == []:
                total_no_signal += 1
            if res.parsed.get("_null_string_normalized"):
                total_null_normalized += 1
        token_dist[str(res.final_max_new_tokens)] += 1

        per_sample_detail.append({
            "index": i,
            "parent_asin": item.get("parent_asin"),
            "latency_s": round(res.latency_s, 3),
            "cache_hit": res.cache_hit,
            "retry_count": res.retry_count,
            "final_max_new_tokens": res.final_max_new_tokens,
            "done_reason": res.done_reason,
            "parse_success": res.parsed is not None,
        })

        if len(raw_outputs) < 15 and res.parsed is not None:
            raw_outputs.append({
                "sample_index": i,
                "item_title": item.get("title"),
                "review_text": item.get("review_text", "")[:200],
                "result": res.parsed,
                "final_max_new_tokens": res.final_max_new_tokens,
                "done_reason": res.done_reason,
                "retry_count": res.retry_count,
            })

    client.close()

    n_parsed = len(parsed_results)

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
        intent_texts = [r["purpose"] for r in parsed_results if r.get("purpose")]
    length_stats = _length_stats(intent_texts)

    # latency stats
    mean_miss = statistics.mean(miss_latencies) if miss_latencies else 0.0
    median_miss = statistics.median(miss_latencies) if miss_latencies else 0.0
    lat_stats = LatencyStats(
        n_cache_miss=len(miss_latencies),
        n_cache_hit=hit_count,
        mean_miss_latency_s=round(mean_miss, 3),
        median_miss_latency_s=round(median_miss, 3),
        mean_hit_latency_s=0.0,
        estimated_n200_time_s=round(mean_miss * 200, 1),
    )

    # retry stats
    retry_rate = n_samples_with_retry / n_total if n_total else 0.0
    rstat = RetryStats(
        retry_count=total_retry_count,
        retry_rate=round(retry_rate, 4),
        empty_response_count=total_empty,
        done_reason_length_count=total_length,
        parse_failure_count=total_parse_fail,
        schema_failure_count=total_schema_fail,
        input_truncation_count=total_truncated,
        no_signal_empty_intent_count=total_no_signal,
        null_string_normalized_count=total_null_normalized,
        final_max_new_tokens_distribution=dict(token_dist),
        per_sample_latencies=[
            round(d["latency_s"], 3) for d in per_sample_detail if not d["cache_hit"]
        ],
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
        f"retry_rate={retry_rate:.0%}",
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
        retry_stats=rstat,
        per_sample_detail=per_sample_detail,
        failed_samples=failed,
        sample_parent_asins=sample_asins,
        raw_outputs=raw_outputs,
        go_nogo=go,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot 1 A/B: intent extraction (plan.md §4 Stage P1)"
    )
    parser.add_argument("--config", required=True, help="Path to Pilot 1 YAML config")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Override n_samples (e.g. 3 for smoke test)")
    parser.add_argument("--category", default=None,
                        help="Override category (e.g. Beauty_and_Personal_Care, Books)")
    parser.add_argument("--domain_type", default=None,
                        help="Override domain_type (functional|lifestyle|content)")
    parser.add_argument("--output", default=None,
                        help="Override output_path (e.g. results/pilot/smoke3.json)")
    args = parser.parse_args()

    config = load_config(args.config, Pilot1Config)
    if args.n_samples is not None:
        config.n_samples = args.n_samples
    if args.category is not None:
        config.category = args.category
        stem = config.output_path.rsplit("_", 1)[0]
        config.output_path = f"{stem}_{args.category}.json"
    if args.domain_type is not None:
        config.domain_type = args.domain_type
    if args.output is not None:
        config.output_path = args.output

    report = run_pilot1(config)
    write_report(report, config.output_path)
    print(f"Report written to {config.output_path}")
    print(f"  parse_success_rate  : {report.parse_success_rate:.1%}")
    print(f"  discriminative_ratio: {report.discriminative_ratio:.1%}")
    if report.prompt_version == "p1_aspect":
        print(f"  aspect_coverage_ratio          : {report.aspect_coverage_ratio:.1%}")
        print(f"  disc_and_aspect_valid          : {report.discriminative_and_aspect_valid_ratio:.1%}")
    r = report.retry_stats
    print(f"  retry_rate          : {r.retry_rate:.1%}  (total_retries={r.retry_count})")
    print(f"  empty_response_count: {r.empty_response_count}")
    print(f"  done_reason_length  : {r.done_reason_length_count}")
    print(f"  parse_failure_count : {r.parse_failure_count}")
    print(f"  schema_failure_count: {r.schema_failure_count}")
    print(f"  input_truncation    : {r.input_truncation_count}")
    if report.prompt_version == "p1_aspect":
        print(f"  no_signal_empty_intent: {r.no_signal_empty_intent_count}")
        print(f"  null_string_normalized: {r.null_string_normalized_count}")
    print(f"  token_distribution  : {r.final_max_new_tokens_distribution}")
    print(f"  cache_miss latency  : mean={report.latency.mean_miss_latency_s:.2f}s "
          f"median={report.latency.median_miss_latency_s:.2f}s")
    print(f"  estimated N=200 time: {report.latency.estimated_n200_time_s:.0f}s "
          f"(cache-miss mean × 200; do NOT run N=200 without approval)")
    print(f"  go_nogo (directional): {report.go_nogo}")


if __name__ == "__main__":
    main()
