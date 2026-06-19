# SASRec.pytorch Integration Notes

## Source
- Repo: https://github.com/pmixer/SASRec.pytorch
- Commit: fde8a9c (Remove urldate from citation in README.md)
- Cloned: 2026-06-19
- Branch: main (LayerNorm-improved version with `norm_first` flag)

## Dependencies (install manually)
```bash
pip install torch torchvision torchaudio  # PyTorch ≥ 2.0, CUDA 12.x
pip install numpy scipy tqdm
```

## What we took from this repo

Only `python/model.py` was adapted into `src/models/sasrec.py`.
The files `python/utils.py`, `python/main.py`, and `python/data/` are **not used** — replaced by our own implementations.

## Changes made to model.py (4 additions, 0 core modifications)

| Change | Location | Description |
|--------|----------|-------------|
| `default_intent` parameter | `SASRec.__init__` | `nn.Parameter(torch.zeros(1, hidden_units))` — defined for F7+ use, unused in F6a |
| `prefix_embeds=None` argument | `log2feats()` | PPR-style prefix injection; when None → identical code path |
| `prefix_embeds=None` argument | `forward()` | Passes through to `log2feats`, BPR loss uses `log_feats[:, P:, :]` |
| `P` return value | `log2feats()` | Returns (log_feats, P) instead of log_feats alone |

## Core UNCHANGED (identity verified)
- `PointWiseFeedForward` — exact copy
- `item_emb`, `pos_emb`, `emb_dropout` — exact copy
- `attention_layernorms`, `attention_layers` (MultiheadAttention) — exact copy
- `forward_layernorms`, `forward_layers` — exact copy
- `last_layernorm` — exact copy
- BPR loss formula in `forward()` — exact copy (only slice added for prefix skip)
- `predict()` — exact copy (uses `log_feats[:, -1, :]` unchanged)
- Causal attention mask generation — exact copy (size extended when P > 0)

## Prefix injection design (PPR-style)
```
item_embs = item_emb(log_seqs) * sqrt(d) + pos_emb(positions)   ← UNCHANGED
item_embs = emb_dropout(item_embs)                                ← UNCHANGED

if prefix_embeds is not None:                                     ← NEW
    seqs = cat([prefix_embeds, item_embs], dim=1)  # [B, P+L, d]
    attention_mask: [P+L, P+L] causal (same formula, larger size)

for each block:  ← UNCHANGED (just operates on larger seqs)
    ...

BPR loss: log_feats[:, P:, :]   ← P=0 when no prefix → identical
predict:  log_feats[:, -1, :]   ← unchanged (last position = last item)
```
