"""SASRec backbone — adapted from pmixer/SASRec.pytorch with minimal changes.

Changes vs. third_party/SASRec.pytorch/python/model.py:
  1. dataloader: replaced by src/models/dataloader.py (external change)
  2. full-ranking eval: 100-neg sampling removed (external change in src/eval/full_ranking.py)
  3. prefix hook: log2feats(log_seqs, prefix_embeds=None) — None = identical code path
  4. default_intent: nn.Parameter defined in __init__ (unused in F6a)

Core UNCHANGED (byte-for-byte equivalent when prefix_embeds=None):
  - PointWiseFeedForward (FFN)
  - item_emb, pos_emb, emb_dropout
  - attention_layernorms, attention_layers (MultiheadAttention)
  - forward_layernorms, forward_layers
  - last_layernorm
  - BPR loss formula (pos_logits - neg_logits)
  - predict() — unchanged
"""
import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # [ADDED] Learnable default intent vector (d_sasrec dim).
        # Defined here for F7+ use; intentionally unused in F6a baseline.
        self.default_intent = torch.nn.Parameter(torch.zeros(1, args.hidden_units))

    def log2feats(self, log_seqs, prefix_embeds=None):
        # ── Item embedding + positional embedding (UNCHANGED from pmixer) ──
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)
        # seqs: [B, L, d]

        # [ADDED] Prefix injection — PPR-style.
        # prefix_embeds: [B, P, d] (already scaled/encoded by caller, no pos_emb applied).
        # When None, this entire block is skipped → identical code path to pmixer.
        if prefix_embeds is not None:
            P = prefix_embeds.shape[1]
            seqs = torch.cat([prefix_embeds.to(self.dev), seqs], dim=1)  # [B, P+L, d]
        else:
            P = 0

        tl = seqs.shape[1]  # P+L (or just L when no prefix)
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                          attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                          attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)  # [B, P+L, d]
        return log_feats, P

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, prefix_embeds=None):
        log_feats, P = self.log2feats(log_seqs, prefix_embeds)

        # BPR loss targets — skip prefix positions
        item_feats = log_feats[:, P:, :]  # [B, L, d]; identical to log_feats when P=0

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (item_feats * pos_embs).sum(dim=-1)
        neg_logits = (item_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """Inference — unchanged from pmixer. Uses last position of item sequence."""
        log_feats, _ = self.log2feats(log_seqs)  # prefix_embeds=None in inference
        final_feat = log_feats[:, -1, :]  # [B, d]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # [B, I, d]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
