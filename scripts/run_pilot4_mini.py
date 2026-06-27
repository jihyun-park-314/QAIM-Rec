"""Pilot 4 Mini — controllability de-risk check (plan.md v0.4.13 §4 P4).

Implements three steps on a K≥2 user slice (N≈20-50):

  STEP 1 — mini Stage1: text_encoder InfoNCE fine-tuning
           (skipped by default on CPU; enable with --train_stage1)
  STEP 2 — mini Stage2: IntentProjector trained with L_align + L_retrieval,
           frozen F6a SASRec + frozen or Stage1 text_encoder
  STEP 3 — controllability + directionality measurement:
           (a) Jaccard(correct vs wrong) vs Jaccard(correct vs correct-seed2)
           (b) Recall@10: correct-prefix vs concat-additive baseline
           (c) frozen-bge vs Stage1 Jaccard gap (requires --train_stage1)

Usage (CPU-friendly, ~10 min):
    python scripts/run_pilot4_mini.py

Usage (with Stage1, needs GPU for reasonable speed):
    python scripts/run_pilot4_mini.py --train_stage1 --device cuda:0
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
from src.models.projector import IntentProjector
from src.models.dataloader import load_data

# ─── Constants ────────────────────────────────────────────────────────────────

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_F3_BANK = "data/memory/Books/f3_bank.jsonl"
_ALIGN_PAIRS_FILES = [
    "data/processed/Books/align_pairs_gpu01.jsonl",
    "data/processed/Books/align_pairs_gpu23.jsonl",
]
_CHECKPOINT = "checkpoints/Books/sasrec_pretrain.pt"
_DATA_DIR = "data/processed"
_CATEGORY = "Books"


# ─── Data loading helpers ─────────────────────────────────────────────────────

def load_f3_bank() -> tuple[dict, dict, dict]:
    """Load f3_bank.jsonl.

    Returns:
        mem_index:   mem_id → {user_id, text, vector}
        user_to_mems: user_id → [mem_id, ...]
        mem_records: mem_id → full record
    """
    mem_index: dict[str, dict] = {}
    user_to_mems: dict[str, list[str]] = defaultdict(list)
    with open(_F3_BANK) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mid = rec["memory_id"]
            uid = rec["user_id"]
            emb_field = rec.get("embedding", {})
            vec = emb_field.get("vector")  # pre-computed 768-dim
            text = emb_field.get("source_text") or rec.get("intent_description", "")
            mem_index[mid] = {"user_id": uid, "text": text, "vector": vec}
            user_to_mems[uid].append(mid)
    return mem_index, dict(user_to_mems), {}


def load_align_pairs(mem_index: dict) -> dict[str, list[dict]]:
    """Group align pairs by user_id (resolved through mem_index)."""
    pairs_by_user: dict[str, list[dict]] = defaultdict(list)
    for fpath in _ALIGN_PAIRS_FILES:
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                uid = mem_index.get(pair["positive_memory_id"], {}).get("user_id")
                if uid is not None:
                    pairs_by_user[uid].append(pair)
    return dict(pairs_by_user)


def sample_slice_users(
    user_to_mems: dict,
    pairs_by_user: dict,
    n_users: int,
    seed: int = 42,
) -> list[str]:
    """Sample N K≥2 users that have align pairs."""
    k2_eligible = [
        uid for uid, mems in user_to_mems.items()
        if len(mems) >= 2 and uid in pairs_by_user
    ]
    rng = random.Random(seed)
    return rng.sample(k2_eligible, min(n_users, len(k2_eligible)))


# ─── Text encoder ─────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """bge-base wrapper for Stage1/2 training."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        model_id = "BAAI/bge-base-en-v1.5"
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self.device = device
        self._model.to(device)

    @property
    def dim(self) -> int:
        return self._model.config.hidden_size

    def encode(self, texts: list[str], is_query: bool = False, max_length: int = 128) -> torch.Tensor:
        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        enc = self._tok(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        out = self._model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vecs, p=2, dim=-1)


# ─── Stage1: InfoNCE ──────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Small MLP for InfoNCE contrastive space."""

    def __init__(self, d_in: int = 768, d_out: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


def infonce_loss(
    q: torch.Tensor,     # [B, d]
    pos: torch.Tensor,   # [B, d]
    negs: torch.Tensor,  # [B, K, d]
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE: query vs positive + negatives (explicit hard-negs)."""
    pos_sim = (q * pos).sum(-1, keepdim=True) / temperature         # [B, 1]
    neg_sim = torch.bmm(negs, q.unsqueeze(-1)).squeeze(-1) / temperature  # [B, K]
    logits = torch.cat([pos_sim, neg_sim], dim=1)                   # [B, 1+K]
    labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


def train_stage1(
    encoder: TextEncoder,
    pairs_by_user: dict,
    mem_index: dict,
    slice_users: list[str],
    epochs: int = 3,
    lr: float = 2e-5,
    max_hard_negs: int = 3,
    temperature: float = 0.07,
    device: str = "cpu",
) -> TextEncoder:
    """Fine-tune bge-base backbone + projection head via InfoNCE.

    Returns the fine-tuned encoder (projection head discarded).
    """
    print(f"\n[Stage1] InfoNCE training: {epochs} epochs, lr={lr}, device={device}")
    proj_head = ProjectionHead(d_in=encoder.dim, d_out=128).to(device)
    # Only update projection head + last 2 transformer layers for speed
    trainable_params = list(proj_head.parameters())
    for name, param in encoder._model.named_parameters():
        if any(f"layer.{i}" in name for i in [10, 11]):
            trainable_params.append(param)
        else:
            param.requires_grad_(False)
    opt = torch.optim.Adam(trainable_params, lr=lr)

    # Build flat list of (query, pos_mem_id, [hard_neg_mem_ids]) for slice users
    all_triplets = []
    for uid in slice_users:
        for pair in pairs_by_user.get(uid, []):
            pos_id = pair["positive_memory_id"]
            hard_ids = pair["hard_negative_memory_ids"][:max_hard_negs]
            # Filter hard negs not in mem_index
            hard_ids = [h for h in hard_ids if h in mem_index]
            if not hard_ids:
                continue
            all_triplets.append((pair["query"], pos_id, hard_ids))

    if not all_triplets:
        print("[Stage1] No valid triplets found, skipping.")
        return encoder

    encoder._model.train()
    proj_head.train()
    total_loss = 0.0
    steps = 0

    for epoch in range(epochs):
        random.shuffle(all_triplets)
        epoch_loss = 0.0
        for query_text, pos_id, neg_ids in all_triplets:
            pos_text = mem_index[pos_id]["text"]
            neg_texts = [mem_index[nid]["text"] for nid in neg_ids]

            opt.zero_grad()
            q = proj_head(encoder.encode([query_text], is_query=True))          # [1, 128]
            p = proj_head(encoder.encode([pos_text], is_query=False))            # [1, 128]
            n = proj_head(encoder.encode(neg_texts, is_query=False)).unsqueeze(0)  # [1, K, 128]

            loss = infonce_loss(q, p, n, temperature=temperature)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            steps += 1

        avg = epoch_loss / max(len(all_triplets), 1)
        print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}  steps={steps}")

    encoder._model.eval()
    return encoder


# ─── Stage2: Projector training ───────────────────────────────────────────────

def build_seq_arrays(
    uid: str,
    splits: dict,
    maxlen: int,
    itemnum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (seq, pos, neg) arrays for one user from splits.json."""
    uid_int = int(uid)
    train_items = splits["users"].get(str(uid_int), {}).get("train", [])
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
        neg_item = np.random.randint(1, itemnum + 1)
        while neg_item in ts:
            neg_item = np.random.randint(1, itemnum + 1)
        neg[idx] = neg_item
        nxt = item
        idx -= 1
        if idx == -1:
            break

    return seq, pos, neg


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    mask = (pos_logits != 0).float()
    loss = -F.logsigmoid(pos_logits - neg_logits)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def train_stage2(
    sasrec: SASRec,
    encoder: TextEncoder,
    projector: IntentProjector,
    pairs_by_user: dict,
    mem_index: dict,
    slice_users: list[str],
    splits: dict,
    maxlen: int,
    itemnum: int,
    epochs: int = 3,
    lr: float = 1e-3,
    alpha: float = 0.1,
    device: str = "cpu",
    use_precomputed_emb: bool = True,
) -> IntentProjector:
    """Train IntentProjector (Stage2).

    When use_precomputed_emb=True, h_memory comes from pre-computed vectors
    (no encoder forward pass for memory) — fast on CPU.
    h_query is always encoded on-the-fly.
    """
    print(f"\n[Stage2] Projector training: {epochs} epochs, lr={lr}, alpha={alpha}")
    opt = torch.optim.Adam(projector.parameters(), lr=lr)
    projector.train()
    sasrec.eval()
    # Freeze encoder — only the projector is trained here
    if encoder is not None:
        encoder._model.eval()
        for p in encoder._model.parameters():
            p.requires_grad_(False)

    np.random.seed(42)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = 0

        for uid in slice_users:
            uid_pairs = pairs_by_user.get(uid, [])
            if not uid_pairs:
                continue

            seq, pos, neg = build_seq_arrays(uid, splits, maxlen, itemnum)
            if seq is None:
                continue

            # Pick one pair per user per step
            pair = uid_pairs[epoch % len(uid_pairs)]
            query_text = pair["query"]
            pos_mem_id = pair["positive_memory_id"]

            opt.zero_grad()

            # Encode without grad — projector params still get gradients through MLP
            with torch.no_grad():
                if encoder is not None:
                    h_query = encoder.encode([query_text], is_query=True)  # [1, 768]
                else:
                    h_query = torch.zeros(1, 768)

                if use_precomputed_emb and pos_mem_id in mem_index:
                    vec = mem_index[pos_mem_id]["vector"]
                    h_memory = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                elif encoder is not None:
                    mem_text = mem_index.get(pos_mem_id, {}).get("text", "")
                    h_memory = encoder.encode([mem_text], is_query=False)
                else:
                    h_memory = torch.zeros(1, 768)

            h_query = h_query.to(device)
            h_memory = h_memory.to(device)

            prefix_embeds = projector(h_query, h_memory)  # [1, 1, d_sasrec]

            seq_b = seq[np.newaxis, :]
            pos_b = pos[np.newaxis, :]
            neg_b = neg[np.newaxis, :]

            pos_logits, neg_logits = sasrec(
                np.zeros(1, dtype=np.int32), seq_b, pos_b, neg_b,
                prefix_embeds=prefix_embeds,
            )

            l_retrieval = bpr_loss(pos_logits, neg_logits)

            # Align prefix to last positive item embedding
            last_pos_id = int(pos_b[0, -1])
            if last_pos_id > 0:
                prefix_vec = prefix_embeds[:, 0, :]
                target_vec = sasrec.item_emb(
                    torch.tensor([last_pos_id], device=device)
                ).detach()
                l_align = (1 - F.cosine_similarity(prefix_vec, target_vec, dim=-1)).mean()
            else:
                l_align = torch.tensor(0.0)

            loss = l_retrieval + alpha * l_align
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_steps += 1

        avg = epoch_loss / max(n_steps, 1)
        print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}  steps={n_steps}")

    projector.eval()
    return projector


# ─── Step3: Measurement ───────────────────────────────────────────────────────

def get_topk(
    sasrec: SASRec,
    seq: np.ndarray,
    prefix_embeds: torch.Tensor | None,
    all_item_ids: torch.Tensor,
    k: int,
    device: str,
) -> set[int]:
    """Return top-k item IDs for one user under given prefix."""
    seq_b = seq[np.newaxis, :]
    with torch.no_grad():
        log_feats, _ = sasrec.log2feats(seq_b, prefix_embeds=prefix_embeds)
        final_feat = log_feats[0, -1, :]  # [d_sasrec]
        item_embs = sasrec.item_emb(all_item_ids)  # [N_items, d]
        scores = item_embs @ final_feat   # [N_items]
        topk_ids = scores.topk(k).indices
    return {int(all_item_ids[i]) for i in topk_ids}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def make_prefix_from_vecs(
    h_query: torch.Tensor,    # [1, 768]
    h_memory: torch.Tensor,   # [1, 768]
    projector: IntentProjector,
    device: str,
) -> torch.Tensor:
    return projector(h_query.to(device), h_memory.to(device))  # [1, 1, d_sasrec]


def measure_step3(
    sasrec: SASRec,
    encoder: TextEncoder,
    projector_frozen: IntentProjector,   # trained with frozen-bge h_memory
    projector_stage1: IntentProjector | None,  # trained with Stage1 encoder (or None)
    pairs_by_user: dict,
    mem_index: dict,
    user_to_mems: dict,
    slice_users: list[str],
    splits: dict,
    maxlen: int,
    itemnum: int,
    top_k: int = 10,
    device: str = "cpu",
) -> dict:
    """Compute all Step3 metrics across slice users."""
    all_item_ids = torch.arange(1, itemnum + 1, dtype=torch.long, device=device)

    results = []
    rng = random.Random(42)

    for uid in slice_users:
        uid_pairs = pairs_by_user.get(uid, [])
        if len(uid_pairs) < 2:
            continue

        uid_mems = user_to_mems.get(uid, [])
        if len(uid_mems) < 2:
            continue

        uid_int = int(uid)
        user_split = splits["users"].get(str(uid_int))
        if not user_split:
            continue
        train_items = user_split.get("train", [])
        val_item = user_split.get("val")
        if len(train_items) < 1 or not val_item:
            continue

        # Build sequence
        seq = np.zeros(maxlen, dtype=np.int32)
        for i, item in enumerate(train_items[-maxlen:]):
            seq[maxlen - len(train_items[-maxlen:]) + i] = item

        # Queries: q1 (primary), q2 (different pair = seed2)
        pair1 = uid_pairs[0]
        pair2 = uid_pairs[min(1, len(uid_pairs) - 1)]
        q1_text = pair1["query"]
        q2_text = pair2["query"]

        # Correct memory: positive from pair1
        correct_mid = pair1["positive_memory_id"]
        correct_mem_text = mem_index.get(correct_mid, {}).get("text", "")
        correct_vec = mem_index.get(correct_mid, {}).get("vector")

        # Wrong memory: random mem from a different user
        other_users = [u for u in slice_users if u != uid and user_to_mems.get(u)]
        if not other_users:
            continue
        wrong_uid = rng.choice(other_users)
        wrong_mid = rng.choice(user_to_mems[wrong_uid])
        wrong_vec = mem_index.get(wrong_mid, {}).get("vector")
        wrong_mem_text = mem_index.get(wrong_mid, {}).get("text", "")

        if correct_vec is None or wrong_vec is None:
            continue

        # Encode queries on-the-fly
        with torch.no_grad():
            h_q1 = encoder.encode([q1_text], is_query=True)   # [1, 768]
            h_q2 = encoder.encode([q2_text], is_query=True)   # [1, 768]

        h_correct = torch.tensor(correct_vec, dtype=torch.float32).unsqueeze(0)  # [1, 768]
        h_wrong = torch.tensor(wrong_vec, dtype=torch.float32).unsqueeze(0)

        # ── (a) Controllability with frozen-bge projector ───────────────────
        with torch.no_grad():
            pfx_correct_q1 = make_prefix_from_vecs(h_q1, h_correct, projector_frozen, device)
            pfx_wrong_q1   = make_prefix_from_vecs(h_q1, h_wrong,   projector_frozen, device)
            pfx_correct_q2 = make_prefix_from_vecs(h_q2, h_correct, projector_frozen, device)

        top_correct_q1 = get_topk(sasrec, seq, pfx_correct_q1, all_item_ids, top_k, device)
        top_wrong_q1   = get_topk(sasrec, seq, pfx_wrong_q1,   all_item_ids, top_k, device)
        top_correct_q2 = get_topk(sasrec, seq, pfx_correct_q2, all_item_ids, top_k, device)
        top_no_prefix  = get_topk(sasrec, seq, None,            all_item_ids, top_k, device)

        j_cw = jaccard(top_correct_q1, top_wrong_q1)
        j_cc = jaccard(top_correct_q1, top_correct_q2)
        j_vanilla = jaccard(top_correct_q1, top_no_prefix)

        # controllability holds if correct-wrong overlap < correct-correct overlap
        ctrl_holds = j_cw < j_cc
        # over-smoothing: prefix barely changes output vs no-prefix
        over_smooth = j_vanilla > 0.8

        # ── (b) Directionality: Recall@k correct vs concat-additive ─────────
        recall_correct = 1.0 if val_item in top_correct_q1 else 0.0

        # concat baseline: add projector output to final feature (no prefix injection)
        with torch.no_grad():
            log_feats_base, _ = sasrec.log2feats(seq[np.newaxis, :], prefix_embeds=None)
            final_feat_base = log_feats_base[0, -1, :]    # [d]
            intent_vec = pfx_correct_q1[0, 0, :]           # [d]
            # additive combination
            combined = F.normalize(final_feat_base + intent_vec, p=2, dim=-1)
            item_embs = sasrec.item_emb(all_item_ids)       # [N_items, d]
            scores_concat = item_embs @ combined
            topk_concat = {int(all_item_ids[i]) for i in scores_concat.topk(top_k).indices}
        recall_concat = 1.0 if val_item in topk_concat else 0.0

        # ── (c) frozen-bge vs Stage1 Jaccard gap ────────────────────────────
        j_cw_stage1 = None
        if projector_stage1 is not None:
            # Re-encode memory with Stage1 encoder (on-the-fly, not pre-computed)
            with torch.no_grad():
                h_correct_s1 = encoder.encode([correct_mem_text], is_query=False)
                h_wrong_s1   = encoder.encode([wrong_mem_text],   is_query=False)
                pfx_c_s1 = make_prefix_from_vecs(h_q1, h_correct_s1, projector_stage1, device)
                pfx_w_s1 = make_prefix_from_vecs(h_q1, h_wrong_s1,   projector_stage1, device)
            top_c_s1 = get_topk(sasrec, seq, pfx_c_s1, all_item_ids, top_k, device)
            top_w_s1 = get_topk(sasrec, seq, pfx_w_s1, all_item_ids, top_k, device)
            j_cw_stage1 = jaccard(top_c_s1, top_w_s1)

        results.append({
            "user_id": uid,
            "j_correct_vs_wrong": j_cw,
            "j_correct_vs_seed2": j_cc,
            "j_correct_vs_vanilla": j_vanilla,
            "controllability_holds": ctrl_holds,
            "over_smoothing": over_smooth,
            "recall_correct": recall_correct,
            "recall_concat_baseline": recall_concat,
            "j_cw_stage1": j_cw_stage1,
        })

    return results


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        print("\n[Report] No results to report.")
        return {}

    ctrl_rate = sum(r["controllability_holds"] for r in results) / n
    smooth_rate = sum(r["over_smoothing"] for r in results) / n
    avg_j_cw = sum(r["j_correct_vs_wrong"] for r in results) / n
    avg_j_cc = sum(r["j_correct_vs_seed2"] for r in results) / n
    avg_j_vanilla = sum(r["j_correct_vs_vanilla"] for r in results) / n
    recall_correct = sum(r["recall_correct"] for r in results) / n
    recall_concat  = sum(r["recall_concat_baseline"] for r in results) / n

    stage1_results = [r for r in results if r["j_cw_stage1"] is not None]
    if stage1_results:
        avg_j_cw_s1 = sum(r["j_cw_stage1"] for r in stage1_results) / len(stage1_results)
        gap_s1 = avg_j_cw - avg_j_cw_s1
    else:
        avg_j_cw_s1 = None
        gap_s1 = None

    # ── Go/No-go decision ───────────────────────────────────────────────────
    # Green: controllability_rate >= 0.5 AND over_smoothing_rate < 0.5
    # Red: over_smoothing_rate >= 0.5 OR controllability_rate < 0.3
    go_nogo = "GO" if (ctrl_rate >= 0.5 and smooth_rate < 0.5) else "NO-GO"

    print("\n" + "═" * 60)
    print("  Pilot 4 Mini — Step 3 Report")
    print("═" * 60)
    print(f"  Users measured:         {n}")
    print()
    print("  (a) Controllability")
    print(f"    avg Jaccard(C vs W):  {avg_j_cw:.4f}")
    print(f"    avg Jaccard(C vs C2): {avg_j_cc:.4f}")
    print(f"    avg Jaccard(C vs ∅):  {avg_j_vanilla:.4f}  (over-smoothing check)")
    print(f"    ctrl holds rate:      {ctrl_rate:.2%}  (J_CW < J_CC)")
    print(f"    over-smoothing rate:  {smooth_rate:.2%}  (J_vanilla > 0.8)")
    print()
    print("  (b) Directionality (Recall@k)")
    print(f"    correct-prefix:       {recall_correct:.4f}")
    print(f"    concat-additive:      {recall_concat:.4f}")
    diff = recall_correct - recall_concat
    print(f"    delta:                {diff:+.4f}")
    print()
    if avg_j_cw_s1 is not None:
        print("  (c) frozen-bge vs Stage1 Jaccard gap")
        print(f"    frozen-bge J_CW:      {avg_j_cw:.4f}")
        print(f"    Stage1 J_CW:          {avg_j_cw_s1:.4f}")
        print(f"    gap (frozen-stage1):  {gap_s1:+.4f}  (>0 = Stage1 improves discrimination)")
    else:
        print("  (c) Stage1 comparison: skipped (run with --train_stage1)")
    print()
    print(f"  ★ Decision: {go_nogo}")
    if go_nogo == "NO-GO":
        if smooth_rate >= 0.5:
            print("    → OVER-SMOOTHING: P=1 prefix washed out by 20-item sequence.")
            print("      Consider P>1, prefix scaling, or cross-attention injection.")
        else:
            print("    → LOW CONTROLLABILITY: prefix not reliably changing output.")
    else:
        print("    → Prefix mechanism works. Proceed to full Stage1 → Stage2.")
    print("═" * 60)

    summary = {
        "n_users": n,
        "controllability_rate": ctrl_rate,
        "over_smoothing_rate": smooth_rate,
        "avg_j_correct_vs_wrong": avg_j_cw,
        "avg_j_correct_vs_seed2": avg_j_cc,
        "avg_j_correct_vs_vanilla": avg_j_vanilla,
        "recall_correct": recall_correct,
        "recall_concat_baseline": recall_concat,
        "recall_delta": diff,
        "avg_j_cw_stage1": avg_j_cw_s1,
        "stage1_jaccard_gap": gap_s1,
        "go_nogo": go_nogo,
        "per_user": results,
    }
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--stage1_epochs", type=int, default=3)
    parser.add_argument("--stage2_epochs", type=int, default=3)
    parser.add_argument("--stage2_lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--train_stage1", action="store_true",
                        help="Fine-tune bge-base with InfoNCE (slow on CPU, prefer GPU)")
    parser.add_argument("--out", type=str, default="results/pilot4_mini/report.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

    # ── Load indices ──────────────────────────────────────────────────────────
    print("[Init] Loading f3_bank index...")
    mem_index, user_to_mems, _ = load_f3_bank()
    print(f"       {len(mem_index)} memories, {len(user_to_mems)} users")

    print("[Init] Loading align pairs...")
    pairs_by_user = load_align_pairs(mem_index)
    print(f"       {sum(len(v) for v in pairs_by_user.values())} pairs across {len(pairs_by_user)} users")

    print(f"[Init] Sampling {args.n_users} K≥2 slice users (seed={args.seed})...")
    slice_users = sample_slice_users(user_to_mems, pairs_by_user, args.n_users, args.seed)
    print(f"       Sampled {len(slice_users)} users: {slice_users[:5]}...")

    # ── Load SASRec ───────────────────────────────────────────────────────────
    print(f"\n[Init] Loading F6a SASRec from {_CHECKPOINT}...")
    ckpt = torch.load(_CHECKPOINT, map_location=device)
    saved_args = ckpt["args"]
    if isinstance(saved_args, dict):
        saved_args = SimpleNamespace(**saved_args)
    sasrec_args = SimpleNamespace(
        maxlen=saved_args.maxlen,
        hidden_units=saved_args.hidden_units,
        num_blocks=saved_args.num_blocks,
        num_heads=saved_args.num_heads,
        dropout_rate=saved_args.dropout_rate,
        norm_first=saved_args.norm_first,
        device=device,
    )
    dataset = load_data(_CATEGORY, _DATA_DIR)
    sasrec = SASRec(dataset["usernum"], dataset["itemnum"], sasrec_args)
    sasrec.load_state_dict(ckpt["model_state_dict"])
    for p in sasrec.parameters():
        p.requires_grad_(False)
    sasrec.to(device)
    sasrec.eval()
    maxlen = saved_args.maxlen
    itemnum = dataset["itemnum"]
    print(f"       usernum={dataset['usernum']}, itemnum={itemnum}, maxlen={maxlen}")

    # Load splits
    with open(f"{_DATA_DIR}/{_CATEGORY}/splits.json") as f:
        splits = json.load(f)

    # ── Load text encoder ─────────────────────────────────────────────────────
    print("\n[Init] Loading text encoder (bge-base)...")
    encoder = TextEncoder(device=device)
    d_text = encoder.dim
    d_sasrec = saved_args.hidden_units
    print(f"       d_text={d_text}, d_sasrec={d_sasrec}")

    # ── Stage1 (optional) ─────────────────────────────────────────────────────
    encoder_frozen = encoder  # default: always use frozen bge
    encoder_stage1 = None
    if args.train_stage1:
        print("\n[Stage1] Fine-tuning bge-base backbone with InfoNCE...")
        # Deep-copy encoder for Stage1 to keep frozen baseline intact
        import copy
        encoder_stage1 = copy.deepcopy(encoder)
        encoder_stage1 = train_stage1(
            encoder=encoder_stage1,
            pairs_by_user=pairs_by_user,
            mem_index=mem_index,
            slice_users=slice_users,
            epochs=args.stage1_epochs,
            device=device,
        )
    else:
        print("\n[Stage1] Skipped (--train_stage1 not set). Using pre-computed embeddings.")

    # ── Stage2: frozen-bge projector ──────────────────────────────────────────
    projector_frozen = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
    projector_frozen = train_stage2(
        sasrec=sasrec,
        encoder=encoder_frozen,
        projector=projector_frozen,
        pairs_by_user=pairs_by_user,
        mem_index=mem_index,
        slice_users=slice_users,
        splits=splits,
        maxlen=maxlen,
        itemnum=itemnum,
        epochs=args.stage2_epochs,
        lr=args.stage2_lr,
        alpha=args.alpha,
        device=device,
        use_precomputed_emb=True,
    )

    # ── Stage2: Stage1 projector (if Stage1 ran) ──────────────────────────────
    projector_stage1 = None
    if encoder_stage1 is not None:
        print("\n[Stage2/Stage1-enc] Training projector with Stage1 encoder...")
        projector_stage1 = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
        projector_stage1 = train_stage2(
            sasrec=sasrec,
            encoder=encoder_stage1,
            projector=projector_stage1,
            pairs_by_user=pairs_by_user,
            mem_index=mem_index,
            slice_users=slice_users,
            splits=splits,
            maxlen=maxlen,
            itemnum=itemnum,
            epochs=args.stage2_epochs,
            lr=args.stage2_lr,
            alpha=args.alpha,
            device=device,
            use_precomputed_emb=False,  # re-encode with Stage1 encoder
        )

    # ── Step3: Measurement ────────────────────────────────────────────────────
    print("\n[Step3] Measuring controllability + directionality...")
    results = measure_step3(
        sasrec=sasrec,
        encoder=encoder_frozen,
        projector_frozen=projector_frozen,
        projector_stage1=projector_stage1,
        pairs_by_user=pairs_by_user,
        mem_index=mem_index,
        user_to_mems=user_to_mems,
        slice_users=slice_users,
        splits=splits,
        maxlen=maxlen,
        itemnum=itemnum,
        top_k=args.top_k,
        device=device,
    )

    summary = print_report(results)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Done] Report saved to {out_path}")

    # Save checkpoints
    ckpt_dir = out_path.parent
    torch.save(projector_frozen.state_dict(),
               ckpt_dir / "projector_frozen_bge.pt")
    if projector_stage1 is not None:
        torch.save(projector_stage1.state_dict(),
                   ckpt_dir / "projector_stage1.pt")
        torch.save(encoder_stage1._model.state_dict(),
                   ckpt_dir / "encoder_stage1_backbone.pt")
    print(f"[Done] Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
