"""Pilot 2 — User facet distribution (plan.md §4 Stage P2).

Per plan.md §1 design principle, this pilot is a pure
`config -> input path -> output path` CLI that reuses M1/`prompts/p1_intent.py`
and M2/`cluster.py` + `prototypes.py` with a small config (no separate
implementation). Skeleton only: signatures + dataclass schemas, no
implementation logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.pilot.common import load_config, write_report


@dataclass
class Pilot2Config:
    """Config for Pilot 2 (plan.md §4 Stage P2).

    `tau_personal_candidates`, `tau_global`, `k_min`/`k_max`, and
    `prototype_count_range` mirror `configs/memory/bank.yaml` (plan.md §2.2,
    M2) and are swept/applied during clustering.
    """

    category: str = "Office_Products"
    min_history_length: int = 8  # L, plan.md §6.2
    n_users: int = 400

    tau_personal_candidates: list[float] = None  # e.g. [0.3, 0.4, 0.5]
    k_min: int = 2
    k_max: int = 5

    tau_global: float = 0.4
    prototype_count_range: list[int] = None  # e.g. [8, 15]

    embedding_model_id: str = "BAAI/bge-base-en-v1.5"  # plan.md §7 #8

    seed: int = 42
    output_path: str = "results/pilot/pilot2_report.json"


@dataclass
class Pilot2Report:
    """Pilot 2 results (plan.md §4 Stage P2).

    go_nogo is True iff, for some `tau_personal`, the fraction of users with
    K_personal >= 2 OR (K_personal < 2 AND a suitable prototype exists) is
    >= 0.30 (plan.md §4 Stage P2 success criteria).
    """

    tau_sweep_results: list[dict]  # per tau_personal: K_personal distribution, >=2 ratio, etc.
    prototype_cluster_count: int
    fallback_ratios: dict  # per tau_personal: ratio of users with K_personal < k_min

    go_nogo: bool
    notes: str


def run_pilot2(config: Pilot2Config) -> Pilot2Report:
    """Run Pilot 2: user facet distribution over `config.category` users with
    history length >= `config.min_history_length` (plan.md §4 Stage P2).

    Procedure:
      1. M0+M1: extend Pilot 1's P1 extraction to `config.n_users` users with
         history length >= `config.min_history_length`.
      2. M2/`cluster.py`: for each `tau` in `config.tau_personal_candidates`,
         agglomerative cluster each user's discriminative `purpose` embeddings
         (bounded by `config.k_min`/`config.k_max`) -> K_personal per user.
      3. M2/`prototypes.py`: run population-level prototype clustering once
         over all candidate users' discriminative purposes using
         `config.tau_global` -> prototype cluster count + size distribution.
      4. For each `tau_personal`, compute the K_personal distribution, the
         ratio of users with K_personal >= 2, and the fallback ratio (users
         with K_personal < `config.k_min`, who would use prototype fallback
         via `bank.py.assemble`).
      5. go/no-go: for some tau_personal, ratio of users with
         (K_personal >= 2) OR (K_personal < 2 AND suitable prototype exists)
         >= 0.30 (plan.md §4 Stage P2 success criteria).

    TODO: M1 integration — reuse src/llm/prompts/p1_intent.py results from
    Pilot 1 (or re-run for the extended user set).
    TODO: M2 integration — src/memory/cluster.py (per-user agglomerative
    clustering, tau_personal sweep) and src/memory/prototypes.py
    (population-level prototype clustering, tau_global) and
    src/memory/embed.py (config.embedding_model_id) for embeddings.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 2: user facet distribution (plan.md §4 Stage P2)")
    parser.add_argument("--config", required=True, help="Path to Pilot 2 YAML config")
    args = parser.parse_args()

    config = load_config(args.config, Pilot2Config)
    report = run_pilot2(config)
    write_report(report, config.output_path)


if __name__ == "__main__":
    main()
