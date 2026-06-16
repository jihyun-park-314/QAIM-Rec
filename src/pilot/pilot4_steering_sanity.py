"""Pilot 4 — Minimal steering sanity check (plan.md §4 Stage P4, v0.2: minimal
trained projector).

Per plan.md §1 design principle, this pilot is a pure
`config -> input path -> output path` CLI that reuses M2 (memory bank /
wrong-intent synthesis), M3 (SASRec + text_encoder + projector), and M5
(retrieval/recall evaluation) with a small config (no separate
implementation). Skeleton only: signatures + dataclass schemas, no
implementation logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.pilot.common import load_config, write_report


@dataclass
class Pilot4Config:
    """Config for Pilot 4 (plan.md §4 Stage P4).

    Users are drawn from Pilot 2/3 candidates with K_personal >= 2 (so a
    "wrong intent" memory can be synthesized by swapping in another cluster's
    intent_description/preference_signal/persona, plan.md §7 #4).
    """

    category: str = "Office_Products"
    n_users: int = 30  # plan.md §4 P4: N≈20-50
    top_k: int = 10
    n_seeds_for_controllability: int = 2

    train_epochs: int = 5
    sasrec_config: dict = None  # small SASRec hyperparams (M3)
    projector_config: dict = None  # MLP(h_query, h_memory) -> prefix_embeds (M3, §7 #5)

    feature_concat_baseline: bool = True

    seed: int = 42
    output_path: str = "results/pilot/pilot4_report.json"


@dataclass
class Pilot4Report:
    """Pilot 4 results (plan.md §4 Stage P4).

    go_nogo is True iff `controllability.holds` is True AND
    `directionality.fraction_users_holds` >= 0.50 (plan.md §4 Stage P4
    success criteria). If `directionality.measured` is False, go_nogo is
    based on `controllability.holds` alone and `limitations` MUST explain
    that (b) is deferred to F7/F8 (plan.md §4 P4 "명시적 한계",
    requirement [필수]#2) — an untrained/minimally-trained prefix must not be
    used to judge (b).
    """

    controllability: dict  # {jaccard_correct_vs_wrong, jaccard_correct_vs_seed, holds}
    directionality: dict  # {recall_correct, recall_concat_baseline, fraction_users_holds, measured}

    go_nogo: bool
    notes: str
    limitations: str


def run_pilot4(config: Pilot4Config) -> Pilot4Report:
    """Run Pilot 4: minimal steering sanity check over `config.n_users` users
    from `config.category` (plan.md §4 Stage P4).

    Procedure:
      1. Select `config.n_users` users with K_personal >= 2 (from Pilot 2/3
         candidates). For each user, designate one memory as "correct intent"
         and synthesize a "wrong intent" memory by swapping
         intent_description/preference_signal/persona from another
         user's/cluster's memory (plan.md §7 #4 Pilot 4 wrong-intent synthesis).
      2. M3: pretrain a small SASRec from scratch on `config.category` data
         using `config.sasrec_config`, and minimally train a small
         text_encoder + projector (`h_intent = MLP(h_query, h_memory)`,
         `config.projector_config`) for `config.train_epochs` epochs on a
         small subset of P3 align pairs (L_align only or L_align + small
         L_retrieval, plan.md §4 P4).
      3. Inject `h_intent` as a single prefix token (P=1, plan.md §7 #5) and
         compute top-`config.top_k` recommendations under:
         (a) correct intent, (b) wrong intent, (c) correct intent with a
         different seed (`config.n_seeds_for_controllability`).
      4. controllability: compare Jaccard(top-k correct, top-k wrong) vs
         Jaccard(top-k correct, top-k correct/other-seed) ->
         `controllability.holds` = True if the former is significantly lower.
      5. directionality: compare Recall@`config.top_k` (held-out interactions)
         under correct-intent prefix vs `config.feature_concat_baseline`
         (concat h_query/h_memory into SASRec input instead of prefix) ->
         `directionality.fraction_users_holds` = fraction of users where
         prefix Recall > concat-baseline Recall. If unmeasurable at this
         scale, set `directionality.measured = False` and document the
         deferral to F7/F8 in `limitations`.
      6. go/no-go: controllability.holds AND
         directionality.fraction_users_holds >= 0.50 (or controllability.holds
         alone if directionality.measured is False, per `limitations`).

    TODO: M2 integration — src/memory/{bank,synth}.py for memory bank
    loading and wrong-intent synthesis (intent_description/preference_signal/
    persona swap, plan.md §7 #4).
    TODO: M3 integration — src/model/ SASRec (config.sasrec_config),
    text_encoder, and projector.py (config.projector_config,
    h_intent = MLP(h_query, h_memory), single prefix token P=1).
    TODO: M5 integration — src/eval/ for Recall@k computation under both
    prefix-injection and feature-concat-baseline configurations.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 4: minimal steering sanity check (plan.md §4 Stage P4)")
    parser.add_argument("--config", required=True, help="Path to Pilot 4 YAML config")
    args = parser.parse_args()

    config = load_config(args.config, Pilot4Config)
    report = run_pilot4(config)
    write_report(report, config.output_path)


if __name__ == "__main__":
    main()
