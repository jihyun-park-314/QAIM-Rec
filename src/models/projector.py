"""M3 Projector: (h_query, h_memory) → prefix_embeds [B, 1, d_sasrec].

Architecture (plan.md v0.4.5 §3 M3):
  concat(h_query, h_memory) [B, 2·d_text]
    → Linear(2·d_text, hidden_dim) → LayerNorm → GELU → Dropout
    → Linear(hidden_dim, d_sasrec)
    → reshape [B, 1, d_sasrec]   (P=1 prefix, §7 #5)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IntentProjector(nn.Module):
    def __init__(
        self,
        d_text: int = 768,
        d_sasrec: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_text, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_sasrec),
        )
        self.d_sasrec = d_sasrec

    def forward(
        self,
        h_query: torch.Tensor,   # [B, d_text]
        h_memory: torch.Tensor,  # [B, d_text]
    ) -> torch.Tensor:           # [B, 1, d_sasrec]
        x = torch.cat([h_query, h_memory], dim=-1)  # [B, 2·d_text]
        out = self.net(x)                            # [B, d_sasrec]
        return out.unsqueeze(1)                      # [B, 1, d_sasrec]
