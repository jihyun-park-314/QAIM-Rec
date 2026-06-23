"""M2/M5 embedding module: BAAI/bge-base-en-v1.5 wrapper (plan.md §7 #8).

Loads from local HF cache only (no auto-download — offline environment).
Uses mean-pooling + L2 normalisation, returning float32 numpy arrays.

BGE retrieval convention (asymmetric search):
  - Queries: prepend "Represent this sentence for searching relevant passages: "
  - Passages/titles: no prefix

Use encode_queries() for user pseudo-queries and encode_corpus() for item titles.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class EmbeddingModel:
    DEFAULT_MODEL_ID = "BAAI/bge-base-en-v1.5"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        batch_size: int = 256,
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tok = AutoTokenizer.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self._model = AutoModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).to(self.device).eval()

    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._model.config.hidden_size

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        """Encode a list of pre-formatted strings, return [N, d] float32."""
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self._model(**enc)
                # Mean-pool over non-padding tokens
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vecs = F.normalize(vecs, p=2, dim=-1)
            all_vecs.append(vecs.cpu().float().numpy())
        return np.concatenate(all_vecs, axis=0)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode search queries with BGE retrieval prefix."""
        prefix = "Represent this sentence for searching relevant passages: "
        return self._encode_raw([prefix + t for t in texts])

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        """Encode item titles / passages (no prefix)."""
        return self._encode_raw(texts)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Convenience wrapper — dispatches to encode_queries or encode_corpus."""
        if is_query:
            return self.encode_queries(texts)
        return self.encode_corpus(texts)
