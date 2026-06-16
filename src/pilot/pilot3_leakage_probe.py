"""Pilot 3 — Leakage probe / leakage floor (plan.md §4 Stage P3, v0.2 redefinition).

Per plan.md §1 design principle, this pilot is a pure
`config -> input path -> output path` CLI that reuses M1/`prompts/p2_pseudo_query.py`
and full-catalog retrieval with a small config (no separate implementation).
Skeleton only: signatures + dataclass schemas, no implementation logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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
    recall_ks: list[int] = None  # e.g. [10, 50]

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

    TODO: M1 integration — src/llm/client.py + src/llm/prompts/p2_pseudo_query.py
    for pseudo-query generation and identifying-information masking validation.
    TODO: M2/M5 integration — src/memory/embed.py (config.embedding_model_id)
    for query embedding, and full-catalog retrieval/recall computation
    (reuse src/eval/ once available).
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 3: leakage probe (plan.md §4 Stage P3)")
    parser.add_argument("--config", required=True, help="Path to Pilot 3 YAML config")
    args = parser.parse_args()

    config = load_config(args.config, Pilot3Config)
    report = run_pilot3(config)
    write_report(report, config.output_path)


if __name__ == "__main__":
    main()
