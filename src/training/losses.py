"""M4 losses: InfoNCE (Stage1) and BPR retrieval (Stage2).

InfoNCE convention (plan.md §3 M4):
  L_align = -log[ exp(sim(hq,h_pos)/τ) / (exp(sim(hq,h_pos)/τ) + Σ_neg exp(sim(hq,h_neg)/τ)) ]

Implemented via cross-entropy on [B, B+K] logit matrix:
  - First B cols: in-batch logits (sim(hq_i, h_pos_j)/τ), label = i (diagonal = positive)
  - Last K cols: per-row hard-negative logits (padded to max_K)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    h_q: torch.Tensor,          # [B, d] L2-normalized query embeddings
    h_pos: torch.Tensor,        # [B, d] L2-normalized positive memory embeddings
    h_hard_neg: torch.Tensor,   # [B, K, d] L2-normalized hard negative embeddings
    tau: float = 0.07,
) -> torch.Tensor:
    """InfoNCE with in-batch + hard negatives.

    In-batch negatives: sim(hq_i, h_pos_j) for j≠i (cross-batch positives).
    Hard negatives: same-user other-cluster memories.

    Returns scalar loss.
    """
    B = h_q.size(0)

    # In-batch similarity matrix [B, B]: row i = sim(hq_i, h_pos_j) for all j
    inbatch_logits = (h_q @ h_pos.T) / tau   # [B, B]

    # Hard-neg similarity [B, K]: sim(hq_i, h_neg_ik)
    # h_hard_neg: [B, K, d]; h_q: [B, d] -> [B, 1, d]
    hard_logits = (h_q.unsqueeze(1) * h_hard_neg).sum(-1) / tau   # [B, K]

    # Full logit matrix [B, B+K]; positive index = diagonal = label i
    all_logits = torch.cat([inbatch_logits, hard_logits], dim=1)   # [B, B+K]
    labels = torch.arange(B, device=h_q.device)

    return F.cross_entropy(all_logits, labels)


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """BPR loss over non-padding positions."""
    mask = (pos_logits != 0).float()
    loss = -F.logsigmoid(pos_logits - neg_logits)
    return (loss * mask).sum() / mask.sum().clamp(min=1)
