"""Pilot 4b — Mechanism de-risk (plan.md v0.4.13 Pilot4 NO-GO 후속).

STEP 0: P=1/3/5 prefix token count sweep → "양으로 안 풀린다" 확증
STEP 1A: Final-feature additive injection, α ∈ {0.3, 0.5, 1.0, 2.0}
STEP 1B: Cross-attention injection into each SASRec transformer block
STEP 2: Mechanism decision (pass criteria + query-only collapse check)

Pass criteria (same for 1A and 1B):
  - over_smoothing_rate < 0.50   (J(C vs vanilla) < 0.80 for majority)
  - ctrl_rate ≥ 0.50             (J_CW < J_CC for majority)
  - J(C vs vanilla) > 0.30       (history signal preserved, not collapsed)

Usage:
    python scripts/run_pilot4b_mechanism.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.sasrec import SASRec
from src.models.dataloader import load_data

_F3_BANK = "data/memory/Books/f3_bank.jsonl"
_ALIGN_PAIRS_FILES = [
    "data/processed/Books/align_pairs_gpu01.jsonl",
    "data/processed/Books/align_pairs_gpu23.jsonl",
]
_CHECKPOINT = "checkpoints/Books/sasrec_pretrain.pt"
_DATA_DIR = "data/processed"
_CATEGORY = "Books"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─── Data loading (mirrors run_pilot4_mini) ───────────────────────────────────

def load_f3_bank():
    mem_index, user_to_mems = {}, defaultdict(list)
    with open(_F3_BANK) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mid, uid = rec["memory_id"], rec["user_id"]
            emb = rec.get("embedding", {})
            mem_index[mid] = {
                "user_id": uid,
                "text": emb.get("source_text") or rec.get("intent_description", ""),
                "vector": emb.get("vector"),
            }
            user_to_mems[uid].append(mid)
    return mem_index, dict(user_to_mems)


def load_align_pairs(mem_index):
    pairs_by_user = defaultdict(list)
    for fpath in _ALIGN_PAIRS_FILES:
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                uid = mem_index.get(pair["positive_memory_id"], {}).get("user_id")
                if uid:
                    pairs_by_user[uid].append(pair)
    return dict(pairs_by_user)


def sample_slice_users(user_to_mems, pairs_by_user, n, seed=42):
    eligible = [u for u, mems in user_to_mems.items() if len(mems) >= 2 and u in pairs_by_user]
    rng = random.Random(seed)
    return rng.sample(eligible, min(n, len(eligible)))


def build_seq_arrays(uid, splits, maxlen, itemnum):
    train_items = splits["users"].get(uid, {}).get("train", [])
    if len(train_items) <= 1:
        return None, None, None
    seq = np.zeros(maxlen, dtype=np.int32)
    pos = np.zeros(maxlen, dtype=np.int32)
    neg = np.zeros(maxlen, dtype=np.int32)
    nxt = train_items[-1]
    idx = maxlen - 1
    ts = set(train_items)
    for item in reversed(train_items[:-1]):
        seq[idx] = item
        pos[idx] = nxt
        n_item = np.random.randint(1, itemnum + 1)
        while n_item in ts:
            n_item = np.random.randint(1, itemnum + 1)
        neg[idx] = n_item
        nxt = item
        idx -= 1
        if idx == -1:
            break
    return seq, pos, neg


# ─── Text encoder ─────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        mid = "BAAI/bge-base-en-v1.5"
        self._tok = AutoTokenizer.from_pretrained(mid)
        self._model = AutoModel.from_pretrained(mid)
        self.device = device
        self._model.to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self):
        return self._model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, is_query=False, max_length=128):
        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(self.device)
        out = self._model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vecs, p=2, dim=-1)


# ─── Projectors ───────────────────────────────────────────────────────────────

class MultiPrefixProjector(nn.Module):
    """IntentProjector generalized to P output tokens."""
    def __init__(self, d_text=768, d_sasrec=256, P=1, hidden_dim=256):
        super().__init__()
        self.P = P
        self.d_sasrec = d_sasrec
        self.net = nn.Sequential(
            nn.Linear(2 * d_text, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, P * d_sasrec),
        )

    def forward(self, h_q, h_m):  # → [B, P, d_sasrec]
        x = torch.cat([h_q, h_m], dim=-1)
        return self.net(x).view(-1, self.P, self.d_sasrec)


# ─── 1B: Cross-attention injector ─────────────────────────────────────────────

class CrossAttnInjector(nn.Module):
    """One cross-attention module per SASRec transformer block (num_blocks=2)."""
    def __init__(self, d_sasrec=256, n_heads=4, n_blocks=2, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_sasrec, n_heads, dropout=dropout, batch_first=False)
            for _ in range(n_blocks)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_sasrec) for _ in range(n_blocks)])

    def forward_cross_attn(self, seqs, intent_vec, layer_idx):
        """seqs: [B, L, d]; intent_vec: [B, d] → [B, L, d]"""
        seqs_T = seqs.transpose(0, 1)                  # [L, B, d]
        kv = intent_vec.unsqueeze(0)                   # [1, B, d]
        out, _ = self.cross_attn[layer_idx](seqs_T, kv, kv)
        return seqs + self.norms[layer_idx](out.transpose(0, 1))


def log2feats_with_xattn(sasrec, injector, log_seqs, intent_vec, device):
    """Custom SASRec forward with cross-attention intent injection after each block.

    Mirrors sasrec.log2feats (no prefix) but injects intent_vec via cross-attn.
    intent_vec: [B, d_sasrec]
    Returns: [B, L, d_sasrec]
    """
    seqs = sasrec.item_emb(torch.LongTensor(log_seqs).to(device))
    seqs *= sasrec.item_emb.embedding_dim ** 0.5
    poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
    poss *= (log_seqs != 0)
    seqs += sasrec.pos_emb(torch.LongTensor(poss).to(device))
    seqs = sasrec.emb_dropout(seqs)

    tl = seqs.shape[1]
    attn_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))

    for i in range(len(sasrec.attention_layers)):
        seqs = seqs.transpose(0, 1)  # [L, B, d]
        if sasrec.norm_first:
            x = sasrec.attention_layernorms[i](seqs)
            mha_out, _ = sasrec.attention_layers[i](x, x, x, attn_mask=attn_mask)
            seqs = seqs + mha_out
            seqs = seqs.transpose(0, 1)
            seqs = seqs + sasrec.forward_layers[i](sasrec.forward_layernorms[i](seqs))
        else:
            mha_out, _ = sasrec.attention_layers[i](seqs, seqs, seqs, attn_mask=attn_mask)
            seqs = sasrec.attention_layernorms[i](seqs + mha_out)
            seqs = seqs.transpose(0, 1)
            seqs = sasrec.forward_layernorms[i](seqs + sasrec.forward_layers[i](seqs))

        seqs = injector.forward_cross_attn(seqs, intent_vec, i)  # intent injection

    return sasrec.last_layernorm(seqs)  # [B, L, d]


# ─── Training ─────────────────────────────────────────────────────────────────

def bpr_loss(pos_logits, neg_logits):
    mask = (pos_logits != 0).float()
    return ((-F.logsigmoid(pos_logits - neg_logits)) * mask).sum() / mask.sum().clamp(min=1)


def train_projector(
    tag, sasrec, encoder, projector, pairs_by_user, mem_index,
    slice_users, splits, maxlen, itemnum,
    epochs=3, lr=1e-3, alpha=0.1, device="cpu",
):
    """Generic Stage2 training for MultiPrefixProjector (prefix injection mode)."""
    opt = torch.optim.Adam(projector.parameters(), lr=lr)
    projector.train()
    np.random.seed(42)
    for epoch in range(epochs):
        total, n = 0.0, 0
        for uid in slice_users:
            uid_pairs = pairs_by_user.get(uid, [])
            if not uid_pairs:
                continue
            seq, pos, neg = build_seq_arrays(uid, splits, maxlen, itemnum)
            if seq is None:
                continue
            pair = uid_pairs[epoch % len(uid_pairs)]
            pos_mem_id = pair["positive_memory_id"]

            with torch.no_grad():
                h_q = encoder.encode([pair["query"]], is_query=True)
                vec = mem_index[pos_mem_id]["vector"]
                h_m = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

            opt.zero_grad()
            pfx = projector(h_q.to(device), h_m.to(device))  # [1, P, d]
            pos_l, neg_l = sasrec(
                np.zeros(1, dtype=np.int32),
                seq[np.newaxis, :], pos[np.newaxis, :], neg[np.newaxis, :],
                prefix_embeds=pfx,
            )
            loss = bpr_loss(pos_l, neg_l)

            last_pos = int(pos[-1])
            if last_pos > 0:
                pfx_vec = pfx[:, 0, :]
                target = sasrec.item_emb(torch.tensor([last_pos], device=device)).detach()
                loss = loss + alpha * (1 - F.cosine_similarity(pfx_vec, target, dim=-1)).mean()

            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"  [{tag}] epoch {epoch+1}/{epochs}  loss={total/max(n,1):.4f}")
    projector.eval()
    return projector


def train_1a(
    sasrec, encoder, projector, pairs_by_user, mem_index,
    slice_users, splits, maxlen, itemnum,
    alpha=1.0, epochs=3, lr=1e-3, device="cpu",
):
    """Train projector for final-feature additive mode (no prefix)."""
    opt = torch.optim.Adam(projector.parameters(), lr=lr)
    projector.train()
    np.random.seed(42)
    for epoch in range(epochs):
        total, n = 0.0, 0
        for uid in slice_users:
            uid_pairs = pairs_by_user.get(uid, [])
            if not uid_pairs:
                continue
            seq, pos, neg = build_seq_arrays(uid, splits, maxlen, itemnum)
            if seq is None:
                continue
            pair = uid_pairs[epoch % len(uid_pairs)]
            pos_mem_id = pair["positive_memory_id"]
            with torch.no_grad():
                h_q = encoder.encode([pair["query"]], is_query=True).to(device)
                vec = mem_index[pos_mem_id]["vector"]
                h_m = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
                log_feats, _ = sasrec.log2feats(seq[np.newaxis, :], prefix_embeds=None)
                final_feat_base = log_feats[0, -1, :]  # [d]

            opt.zero_grad()
            intent = projector(h_q, h_m)[:, 0, :]  # [1, d]
            combined = final_feat_base.unsqueeze(0) + alpha * intent  # [1, d]

            pos_t = torch.LongTensor(pos[np.newaxis, -1:]).to(device)  # last pos item
            neg_t = torch.LongTensor(neg[np.newaxis, -1:]).to(device)
            pos_emb = sasrec.item_emb(pos_t)   # [1, 1, d]
            neg_emb = sasrec.item_emb(neg_t)
            pos_logit = (combined.unsqueeze(1) * pos_emb).sum(-1)  # [1, 1]
            neg_logit = (combined.unsqueeze(1) * neg_emb).sum(-1)
            loss = bpr_loss(pos_logit, neg_logit)

            last_pos_id = int(pos[-1])
            if last_pos_id > 0:
                target = sasrec.item_emb(torch.tensor([last_pos_id], device=device)).detach()
                loss = loss + 0.1 * (1 - F.cosine_similarity(intent, target, dim=-1)).mean()

            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"  [1A α={alpha}] epoch {epoch+1}/{epochs}  loss={total/max(n,1):.4f}")
    projector.eval()
    return projector


def train_1b(
    sasrec, encoder, projector_1b, injector, pairs_by_user, mem_index,
    slice_users, splits, maxlen, itemnum,
    epochs=3, lr=1e-3, alpha=0.1, device="cpu",
):
    """Train CrossAttnInjector + projector_1b jointly."""
    trainable = list(projector_1b.parameters()) + list(injector.parameters())
    opt = torch.optim.Adam(trainable, lr=lr)
    projector_1b.train()
    injector.train()
    np.random.seed(42)
    for epoch in range(epochs):
        total, n = 0.0, 0
        for uid in slice_users:
            uid_pairs = pairs_by_user.get(uid, [])
            if not uid_pairs:
                continue
            seq, pos, neg = build_seq_arrays(uid, splits, maxlen, itemnum)
            if seq is None:
                continue
            pair = uid_pairs[epoch % len(uid_pairs)]
            pos_mem_id = pair["positive_memory_id"]
            with torch.no_grad():
                h_q = encoder.encode([pair["query"]], is_query=True).to(device)
                vec = mem_index[pos_mem_id]["vector"]
                h_m = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

            opt.zero_grad()
            intent = projector_1b(h_q, h_m)[:, 0, :]  # [1, d_sasrec]
            log_feats = log2feats_with_xattn(
                sasrec, injector, seq[np.newaxis, :], intent, device
            )  # [1, L, d]
            pos_embs = sasrec.item_emb(torch.LongTensor(pos[np.newaxis, :]).to(device))
            neg_embs = sasrec.item_emb(torch.LongTensor(neg[np.newaxis, :]).to(device))
            pos_l = (log_feats * pos_embs).sum(-1)
            neg_l = (log_feats * neg_embs).sum(-1)
            loss = bpr_loss(pos_l, neg_l)

            final_feat = log_feats[0, -1, :]
            last_pos_id = int(pos[-1])
            if last_pos_id > 0:
                target = sasrec.item_emb(torch.tensor([last_pos_id], device=device)).detach()
                loss = loss + alpha * (1 - F.cosine_similarity(final_feat.unsqueeze(0), target, dim=-1)).mean()

            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"  [1B] epoch {epoch+1}/{epochs}  loss={total/max(n,1):.4f}")
    projector_1b.eval()
    injector.eval()
    return projector_1b, injector


# ─── Measurement ──────────────────────────────────────────────────────────────

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def topk_prefix(sasrec, seq, prefix_embeds, all_item_ids, k, device):
    with torch.no_grad():
        log_feats, _ = sasrec.log2feats(seq[np.newaxis, :], prefix_embeds=prefix_embeds)
        final = log_feats[0, -1, :]
        scores = sasrec.item_emb(all_item_ids) @ final
        idx = scores.topk(k).indices
    return {int(all_item_ids[i]) for i in idx}


def topk_1a(sasrec, seq, projector, h_q, h_m, alpha, all_item_ids, k, device):
    with torch.no_grad():
        log_feats, _ = sasrec.log2feats(seq[np.newaxis, :], prefix_embeds=None)
        final = log_feats[0, -1, :]
        intent = projector(h_q, h_m)[0, 0, :]
        combined = final + alpha * intent
        scores = sasrec.item_emb(all_item_ids) @ combined
        idx = scores.topk(k).indices
    return {int(all_item_ids[i]) for i in idx}


def topk_1b(sasrec, injector, projector, seq, h_q, h_m, all_item_ids, k, device):
    with torch.no_grad():
        intent = projector(h_q, h_m)[0, 0, :]  # [d]
        log_feats = log2feats_with_xattn(sasrec, injector, seq[np.newaxis, :], intent.unsqueeze(0), device)
        final = log_feats[0, -1, :]
        scores = sasrec.item_emb(all_item_ids) @ final
        idx = scores.topk(k).indices
    return {int(all_item_ids[i]) for i in idx}


def measure_users(
    slice_users, pairs_by_user, mem_index, user_to_mems, splits,
    sasrec, encoder, all_item_ids, maxlen, k, device, mode, **kwargs,
):
    """Unified measurement for prefix / 1A / 1B modes.

    mode: "prefix" | "1a" | "1b"
    kwargs for prefix: projector (MultiPrefixProjector)
    kwargs for 1a: projector, alpha
    kwargs for 1b: projector, injector
    """
    rng = random.Random(42)
    results = []

    for uid in slice_users:
        uid_pairs = pairs_by_user.get(uid, [])
        if len(uid_pairs) < 2:
            continue
        uid_mems = user_to_mems.get(uid, [])
        if len(uid_mems) < 2:
            continue
        user_split = splits["users"].get(uid)
        if not user_split:
            continue
        train_items = user_split.get("train", [])
        val_item = user_split.get("val")
        if len(train_items) < 1 or not val_item:
            continue

        seq = np.zeros(maxlen, dtype=np.int32)
        for i, item in enumerate(train_items[-maxlen:]):
            seq[maxlen - len(train_items[-maxlen:]) + i] = item

        pair1 = uid_pairs[0]
        pair2 = uid_pairs[min(1, len(uid_pairs) - 1)]
        correct_mid = pair1["positive_memory_id"]
        correct_vec = mem_index.get(correct_mid, {}).get("vector")
        if correct_vec is None:
            continue

        other_users = [u for u in slice_users if u != uid and user_to_mems.get(u)]
        if not other_users:
            continue
        wrong_mid = rng.choice(user_to_mems[rng.choice(other_users)])
        wrong_vec = mem_index.get(wrong_mid, {}).get("vector")
        if wrong_vec is None:
            continue

        h_correct = torch.tensor(correct_vec, dtype=torch.float32).unsqueeze(0).to(device)
        h_wrong = torch.tensor(wrong_vec, dtype=torch.float32).unsqueeze(0).to(device)
        h_q1 = encoder.encode([pair1["query"]], is_query=True)
        h_q2 = encoder.encode([pair2["query"]], is_query=True)

        if mode == "prefix":
            proj = kwargs["projector"]
            pfx_c1 = proj(h_q1, h_correct)
            pfx_w1 = proj(h_q1, h_wrong)
            pfx_c2 = proj(h_q2, h_correct)
            top_c1 = topk_prefix(sasrec, seq, pfx_c1, all_item_ids, k, device)
            top_w1 = topk_prefix(sasrec, seq, pfx_w1, all_item_ids, k, device)
            top_c2 = topk_prefix(sasrec, seq, pfx_c2, all_item_ids, k, device)
            top_van = topk_prefix(sasrec, seq, None, all_item_ids, k, device)
            recall = 1.0 if val_item in top_c1 else 0.0

        elif mode == "1a":
            proj = kwargs["projector"]
            alpha = kwargs["alpha"]
            top_c1 = topk_1a(sasrec, seq, proj, h_q1, h_correct, alpha, all_item_ids, k, device)
            top_w1 = topk_1a(sasrec, seq, proj, h_q1, h_wrong,   alpha, all_item_ids, k, device)
            top_c2 = topk_1a(sasrec, seq, proj, h_q2, h_correct, alpha, all_item_ids, k, device)
            top_van = topk_prefix(sasrec, seq, None, all_item_ids, k, device)
            recall = 1.0 if val_item in top_c1 else 0.0

        elif mode == "1b":
            proj = kwargs["projector"]
            inj = kwargs["injector"]
            top_c1 = topk_1b(sasrec, inj, proj, seq, h_q1, h_correct, all_item_ids, k, device)
            top_w1 = topk_1b(sasrec, inj, proj, seq, h_q1, h_wrong,   all_item_ids, k, device)
            top_c2 = topk_1b(sasrec, inj, proj, seq, h_q2, h_correct, all_item_ids, k, device)
            top_van = topk_prefix(sasrec, seq, None, all_item_ids, k, device)
            recall = 1.0 if val_item in top_c1 else 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        j_cw = jaccard(top_c1, top_w1)
        j_cc = jaccard(top_c1, top_c2)
        j_cv = jaccard(top_c1, top_van)

        results.append({
            "user_id": uid,
            "j_cw": j_cw,
            "j_cc": j_cc,
            "j_cv": j_cv,
            "ctrl": j_cw < j_cc,
            "over_smooth": j_cv > 0.8,
            "history_collapse": j_cv < 0.3,
            "recall": recall,
        })

    return results


def summarize(results, label):
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}
    ctrl_rate = sum(r["ctrl"] for r in results) / n
    smooth_rate = sum(r["over_smooth"] for r in results) / n
    collapse_rate = sum(r["history_collapse"] for r in results) / n
    avg_jcw = sum(r["j_cw"] for r in results) / n
    avg_jcc = sum(r["j_cc"] for r in results) / n
    avg_jcv = sum(r["j_cv"] for r in results) / n
    recall = sum(r["recall"] for r in results) / n
    passed = ctrl_rate >= 0.5 and smooth_rate < 0.5 and avg_jcv > 0.3
    return {
        "label": label, "n": n,
        "ctrl_rate": ctrl_rate,
        "over_smooth_rate": smooth_rate,
        "history_collapse_rate": collapse_rate,
        "avg_j_cw": avg_jcw,
        "avg_j_cc": avg_jcc,
        "avg_j_cv": avg_jcv,
        "recall_at_k": recall,
        "passed": passed,
    }


def print_summary(s):
    label = s["label"]
    n = s.get("n", 0)
    if n == 0:
        print(f"  [{label}] no data")
        return
    sym = "✓ PASS" if s["passed"] else "✗ FAIL"
    ctrl = s["ctrl_rate"]
    smooth = s["over_smooth_rate"]
    collapse = s["history_collapse_rate"]
    jcw, jcc, jcv = s["avg_j_cw"], s["avg_j_cc"], s["avg_j_cv"]
    rc = s["recall_at_k"]
    print(f"  [{label:20s}] {sym} | ctrl={ctrl:.2%} smooth={smooth:.2%} collapse={collapse:.2%} | "
          f"J_CW={jcw:.3f} J_CC={jcc:.3f} J_CV={jcv:.3f} | R@k={rc:.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="results/pilot4b/report.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    print("[Init] Loading data indices...")
    mem_index, user_to_mems = load_f3_bank()
    pairs_by_user = load_align_pairs(mem_index)
    slice_users = sample_slice_users(user_to_mems, pairs_by_user, args.n_users, args.seed)
    print(f"       {len(slice_users)} K≥2 users sampled")

    print("[Init] Loading SASRec F6a...")
    ckpt = torch.load(_CHECKPOINT, map_location=device)
    saved_args = ckpt["args"]
    if isinstance(saved_args, dict):
        saved_args = SimpleNamespace(**saved_args)
    sasrec_args = SimpleNamespace(
        maxlen=saved_args.maxlen, hidden_units=saved_args.hidden_units,
        num_blocks=saved_args.num_blocks, num_heads=saved_args.num_heads,
        dropout_rate=saved_args.dropout_rate, norm_first=saved_args.norm_first,
        device=device,
    )
    dataset = load_data(_CATEGORY, _DATA_DIR)
    sasrec = SASRec(dataset["usernum"], dataset["itemnum"], sasrec_args)
    sasrec.load_state_dict(ckpt["model_state_dict"])
    for p in sasrec.parameters():
        p.requires_grad_(False)
    sasrec.to(device).eval()
    maxlen = saved_args.maxlen
    itemnum = dataset["itemnum"]
    d_sasrec = saved_args.hidden_units
    num_blocks = saved_args.num_blocks
    print(f"       num_blocks={num_blocks}, d_sasrec={d_sasrec}")

    with open(f"{_DATA_DIR}/{_CATEGORY}/splits.json") as f:
        splits = json.load(f)

    print("[Init] Loading text encoder...")
    encoder = TextEncoder(device=device)
    d_text = encoder.dim
    all_item_ids = torch.arange(1, itemnum + 1, dtype=torch.long, device=device)

    measure_kwargs = dict(
        slice_users=slice_users,
        pairs_by_user=pairs_by_user,
        mem_index=mem_index,
        user_to_mems=user_to_mems,
        splits=splits,
        sasrec=sasrec,
        encoder=encoder,
        all_item_ids=all_item_ids,
        maxlen=maxlen,
        k=args.top_k,
        device=device,
    )

    all_summaries = []

    # ── STEP 0: P=1/3/5 prefix sweep ─────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  STEP 0: P=1/3/5 prefix sweep")
    print("══════════════════════════════════════════")
    step0_results = {}
    for P in [1, 3, 5]:
        proj = MultiPrefixProjector(d_text=d_text, d_sasrec=d_sasrec, P=P).to(device)
        print(f"\n  Training MultiPrefixProjector(P={P})...")
        proj = train_projector(
            f"P={P}", sasrec, encoder, proj,
            pairs_by_user, mem_index, slice_users, splits, maxlen, itemnum,
            epochs=args.epochs, lr=args.lr, device=device,
        )
        res = measure_users(mode="prefix", projector=proj, **measure_kwargs)
        s = summarize(res, f"prefix P={P}")
        step0_results[P] = s
        all_summaries.append(s)
        print_summary(s)

    p5_ctrl = step0_results[5]["ctrl_rate"]
    p_tuning_dead = p5_ctrl < 0.5
    print(f"\n  P-tuning verdict: {'DEAD — P=5 still fails (ctrl={:.1%}). Proceed to mechanism.'.format(p5_ctrl) if p_tuning_dead else 'ALIVE — P=5 passes, prefix depth sufficient.'}")

    # ── STEP 1A: Final-feature additive, α grid ───────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  STEP 1A: Final-feature additive (α grid)")
    print("══════════════════════════════════════════")
    # Train one shared projector for 1A (trained at α=1.0 to balance)
    proj_1a = MultiPrefixProjector(d_text=d_text, d_sasrec=d_sasrec, P=1).to(device)
    print(f"\n  Training projector for 1A (α=1.0 training)...")
    proj_1a = train_1a(
        sasrec, encoder, proj_1a,
        pairs_by_user, mem_index, slice_users, splits, maxlen, itemnum,
        alpha=1.0, epochs=args.epochs, lr=args.lr, device=device,
    )
    step1a_results = {}
    alpha_grid = [0.3, 0.5, 1.0, 2.0]
    print("\n  Measuring α grid...")
    for alpha in alpha_grid:
        res = measure_users(mode="1a", projector=proj_1a, alpha=alpha, **measure_kwargs)
        s = summarize(res, f"1A α={alpha}")
        step1a_results[alpha] = s
        all_summaries.append(s)
        print_summary(s)

    best_1a_alpha = max(step1a_results.keys(), key=lambda a: (
        step1a_results[a]["ctrl_rate"] - step1a_results[a]["over_smooth_rate"]
    ))
    best_1a = step1a_results[best_1a_alpha]

    # ── STEP 1B: Cross-attention injection ────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  STEP 1B: Cross-attention injection")
    print("══════════════════════════════════════════")
    proj_1b = MultiPrefixProjector(d_text=d_text, d_sasrec=d_sasrec, P=1).to(device)
    injector = CrossAttnInjector(d_sasrec=d_sasrec, n_heads=4, n_blocks=num_blocks).to(device)
    print(f"\n  Training projector + CrossAttnInjector...")
    proj_1b, injector = train_1b(
        sasrec, encoder, proj_1b, injector,
        pairs_by_user, mem_index, slice_users, splits, maxlen, itemnum,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    res_1b = measure_users(mode="1b", projector=proj_1b, injector=injector, **measure_kwargs)
    s_1b = summarize(res_1b, "1B cross-attn")
    step1b_results = s_1b
    all_summaries.append(s_1b)
    print_summary(s_1b)

    # ── STEP 2: Decision ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 2: Mechanism Decision")
    print("═" * 60)
    print("\n  Summary table:")
    for s in all_summaries:
        print_summary(s)

    candidates = []
    if best_1a["passed"]:
        candidates.append(("1A", best_1a, f"α={best_1a_alpha}"))
    if s_1b["passed"]:
        candidates.append(("1B", s_1b, "cross-attn"))

    print()
    if not candidates:
        decision = "NO-GO"
        reason = "Neither 1A nor 1B passed all criteria."
        chosen = None
    elif len(candidates) == 1:
        decision = "GO"
        chosen_name, chosen_s, chosen_cfg = candidates[0]
        reason = f"{chosen_name} ({chosen_cfg}) passes. Adopt as prefix mechanism."
        chosen = {"mechanism": chosen_name, "config": chosen_cfg}
    else:
        # Both pass: prefer simpler (1A)
        # Compare quality: lower ctrl_rate gap and lower over_smooth wins
        decision = "GO"
        chosen_name, chosen_s, chosen_cfg = candidates[0]  # 1A is simpler
        reason = f"Both 1A and 1B pass. Adopting simpler 1A (α={best_1a_alpha}) — lower architecture cost."
        chosen = {"mechanism": "1A", "config": f"α={best_1a_alpha}"}

    print(f"  ★ Decision: {decision}")
    print(f"    {reason}")
    if chosen:
        # Check query-only collapse sanity
        if chosen["mechanism"] == "1A":
            avg_jcv = best_1a["avg_j_cv"]
        else:
            avg_jcv = s_1b["avg_j_cv"]
        if avg_jcv < 0.3:
            print(f"    ⚠ WARNING: avg J(C vs vanilla)={avg_jcv:.3f} < 0.30 — history signal may be suppressed.")
        else:
            print(f"    ✓ No query-only collapse (J_CV={avg_jcv:.3f} > 0.30)")

    print("═" * 60)

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "step0_p_sweep": {str(P): step0_results[P] for P in [1, 3, 5]},
        "step1a_alpha_grid": {str(a): step1a_results[a] for a in alpha_grid},
        "step1b_cross_attn": step1b_results,
        "step2_decision": {"verdict": decision, "reason": reason, "adopted": chosen},
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Done] Report saved to {out_path}")


if __name__ == "__main__":
    main()
