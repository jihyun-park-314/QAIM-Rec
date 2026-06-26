"""F7 eval conditions: unified prefix generation for 4 eval settings.

Conditions (plan.md v0.4.5 §4 F7):
  vanilla  — no prefix (baseline SASRec)
  correct  — top-1 routed memory for this user (steering signal)
  wrong    — swapped memory from another user / cluster (ablation)
  default  — model.default_intent learnable parameter (cold-start)

All conditions except vanilla return prefix_embeds: [B, 1, d_sasrec].
Headline comparison in the paper: correct vs vanilla (lift).
"""
from __future__ import annotations

from typing import Literal

import torch

from src.models.projector import IntentProjector

Condition = Literal["vanilla", "correct", "wrong", "default"]
ALL_CONDITIONS: tuple[Condition, ...] = ("vanilla", "correct", "wrong", "default")


def make_prefix(
    condition: Condition,
    projector: IntentProjector,
    default_intent: torch.nn.Parameter,  # model.default_intent [1, d_sasrec]
    h_query: torch.Tensor | None = None,         # [B, d_text]
    h_memory_correct: torch.Tensor | None = None, # [B, d_text]
    h_memory_wrong: torch.Tensor | None = None,   # [B, d_text]
    device: str = "cpu",
) -> torch.Tensor | None:
    """Return prefix_embeds [B, 1, d_sasrec] or None for vanilla.

    Caller is responsible for routing (finding correct / wrong memories).
    """
    if condition == "vanilla":
        return None

    if condition == "correct":
        assert h_query is not None and h_memory_correct is not None, \
            "'correct' condition requires h_query and h_memory_correct"
        return projector(
            h_query.to(device),
            h_memory_correct.to(device),
        )  # [B, 1, d_sasrec]

    if condition == "wrong":
        assert h_query is not None and h_memory_wrong is not None, \
            "'wrong' condition requires h_query and h_memory_wrong"
        return projector(
            h_query.to(device),
            h_memory_wrong.to(device),
        )  # [B, 1, d_sasrec]

    if condition == "default":
        # Expand to batch size using h_query shape if available
        B = h_query.shape[0] if h_query is not None else 1
        # default_intent: [1, d_sasrec] → [B, 1, d_sasrec]
        return default_intent.unsqueeze(0).expand(B, -1, -1).to(device)

    raise ValueError(f"Unknown condition: {condition!r}. Use one of {ALL_CONDITIONS}")
