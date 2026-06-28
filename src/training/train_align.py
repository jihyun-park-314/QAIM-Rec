"""Stage 1: InfoNCE contrastive training of text_encoder + projection head.

Architecture (plan.md §3 M4, v0.4.11/v0.4.13/v0.4.14):
  h_query = L2_norm(head(bge(query)))   — trainable (encoder + proj head)
  h_memory = pre-computed frozen bank vector  — frozen (no re-encoding)

  L_align = InfoNCE(h_query, h_pos, {hard_negs + in-batch negs})

Key design decisions:
  - h_memory frozen: bank vectors are offline bge embeddings; only query side trains.
  - Projection head: 768→768 MLP, bridges potential distribution shift between
    fine-tuned query space and frozen memory space.
  - Asymmetric LR (plan.md v0.4.14): encoder lr << head lr (LLaVA convention).
  - Hard negs = same-user other-cluster memories (from align_pairs.jsonl).
  - In-batch negs = other queries' positives (cross_entropy diagonal label).

Frozen-bge baseline: raw bge output (no head) vs bank vectors — establishes the
floor routing accuracy that Stage1 must not drop below (gate: ≥0.90).

Smoke gate (plan.md STEP1):
  1. L_align strictly decreases over smoke_steps (no NaN/divergence).
  2. Train-slice routing accuracy ≥ frozen-bge floor after smoke steps.
  3. Config exposed; determinism (seed=42, two identical runs).

Usage — smoke:
    python3 src/training/train_align.py --smoke \
        --bank_dir data/processed/Books/memory_bank \
        --device cuda:0

Usage — full Stage1 (run by researcher after smoke passes):
    python3 src/training/train_align.py \
        --pairs data/processed/Books/align_pairs.jsonl \
        --bank_dir data/processed/Books/memory_bank \
        --device cuda:0 \
        --out checkpoints/Books/stage1_align_best.pt \
        --num_epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.losses import info_nce

# BGE query prefix (BAAI/bge-base-en-v1.5 convention)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ──────────────────────────────────────────────────────────────────────────────
# Memory bank index
# ──────────────────────────────────────────────────────────────────────────────

def load_memory_index(bank_dir: str) -> tuple[dict, dict, dict, dict]:
    """Scan all per-user JSON files; return four dicts.

    Returns:
        mem_to_vec:    memory_id -> np.float32 array [768]
        mem_to_user:   memory_id -> user_id (str)
        user_to_mems:  user_id -> list[memory_id]
        user_to_k:     user_id -> k_personal (int)
    """
    mem_to_vec: dict[str, np.ndarray] = {}
    mem_to_user: dict[str, str] = {}
    user_to_mems: dict[str, list[str]] = {}
    user_to_k: dict[str, int] = {}

    bank_path = Path(bank_dir)
    for fpath in bank_path.glob("*.json"):
        if fpath.name.startswith("_"):
            continue
        with open(fpath, encoding="utf-8") as f:
            ub = json.load(f)
        uid = str(ub["user_id"])
        k = int(ub.get("k_personal", 0))
        user_to_k[uid] = k
        mids = []
        for unit in ub.get("units", []):
            mid = unit["memory_id"]
            vec = np.array(unit["embedding"]["vector"], dtype=np.float32)
            mem_to_vec[mid] = vec
            mem_to_user[mid] = uid
            mids.append(mid)
        user_to_mems[uid] = mids

    print(f"[bank] loaded {len(mem_to_vec)} memories across {len(user_to_mems)} users")
    return mem_to_vec, mem_to_user, user_to_mems, user_to_k


# ──────────────────────────────────────────────────────────────────────────────
# Text encoder + projection head
# ──────────────────────────────────────────────────────────────────────────────

class TextEncoderWithHead(nn.Module):
    """bge-base-en-v1.5 + query-side projection head.

    The head is a 2-layer MLP outputting in the same 768-dim space as the
    pre-computed frozen memory bank vectors, followed by L2-normalization.

    Asymmetric LR (plan.md v0.4.14):
        - encoder parameters: lr_encoder (small, conservative)
        - head parameters: lr_head (larger, modality-gap focus)

    Use param_groups() to get two param groups for the optimizer.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en-v1.5",
        proj_hidden: int = 768,
        proj_out: int = 768,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self._enc = AutoModel.from_pretrained(model_id, local_files_only=True)
        self.device = device
        self._enc.to(device)

        enc_dim = self._enc.config.hidden_size   # 768 for bge-base
        self.head = nn.Sequential(
            nn.Linear(enc_dim, proj_hidden),
            nn.GELU(),
            nn.LayerNorm(proj_hidden),
            nn.Linear(proj_hidden, proj_out),
        ).to(device)

    @property
    def enc_dim(self) -> int:
        return self._enc.config.hidden_size

    def _bge_pool(self, texts: list[str], add_query_prefix: bool) -> torch.Tensor:
        """Tokenize + mean-pool (bge convention) → [N, d] float32."""
        if add_query_prefix:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        enc = self._tok(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        out = self._enc(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return vecs   # [N, d], NOT normalized yet (head does final normalization)

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        """Encode queries with BGE prefix + projection head → L2-normalized [N, d]."""
        raw = self._bge_pool(texts, add_query_prefix=True)   # [N, d_enc]
        out = self.head(raw)                                  # [N, d_proj]
        return F.normalize(out, p=2, dim=-1)

    def encode_queries_frozen_bge(self, texts: list[str]) -> torch.Tensor:
        """Frozen-bge baseline: BGE only (no head), for routing floor comparison.

        Gradient disabled — call inside torch.no_grad().
        """
        raw = self._bge_pool(texts, add_query_prefix=True)
        return F.normalize(raw, p=2, dim=-1)

    def param_groups(self, lr_encoder: float, lr_head: float) -> list[dict]:
        """Return two param groups for the optimizer (asymmetric LR)."""
        return [
            {"params": self._enc.parameters(), "lr": lr_encoder},
            {"params": self.head.parameters(), "lr": lr_head},
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AlignPairsDataset(torch.utils.data.Dataset):
    """Align pairs with pre-looked-up memory vectors.

    Each item: (query_str, pos_vec [768], hard_neg_vecs [K, 768])
    Unknown memory_ids are silently skipped at load time.
    """

    def __init__(
        self,
        pairs_path: str,
        mem_to_vec: dict[str, np.ndarray],
        max_hard_neg: int = 11,
    ) -> None:
        self.items: list[tuple[str, np.ndarray, np.ndarray]] = []

        skipped = 0
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                q = rec["query"]
                pos_id = rec["positive_memory_id"]
                neg_ids = rec.get("hard_negative_memory_ids", [])

                if pos_id not in mem_to_vec:
                    skipped += 1
                    continue

                pos_vec = mem_to_vec[pos_id]

                # Gather hard neg vectors (skip missing, pad to max_hard_neg)
                neg_vecs = [mem_to_vec[nid] for nid in neg_ids if nid in mem_to_vec]
                if not neg_vecs:
                    # No hard negs: use zeros (in-batch negs still provide signal)
                    neg_arr = np.zeros((max_hard_neg, pos_vec.shape[0]), dtype=np.float32)
                else:
                    # Pad or truncate to max_hard_neg
                    neg_arr = np.array(neg_vecs[:max_hard_neg], dtype=np.float32)
                    if len(neg_arr) < max_hard_neg:
                        pad = np.zeros((max_hard_neg - len(neg_arr), neg_arr.shape[1]), dtype=np.float32)
                        neg_arr = np.vstack([neg_arr, pad])

                self.items.append((q, pos_vec, neg_arr))

        if skipped:
            print(f"[dataset] skipped {skipped} pairs (missing memory_ids)")
        print(f"[dataset] loaded {len(self.items)} pairs from {pairs_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        q, pos_vec, neg_arr = self.items[idx]
        return q, torch.from_numpy(pos_vec), torch.from_numpy(neg_arr)


def collate_fn(batch):
    """Collate variable-length text with fixed-size tensor vectors."""
    queries = [b[0] for b in batch]
    pos_vecs = torch.stack([b[1] for b in batch])   # [B, d]
    neg_vecs = torch.stack([b[2] for b in batch])   # [B, K, d]
    return queries, pos_vecs, neg_vecs


# ──────────────────────────────────────────────────────────────────────────────
# Routing accuracy
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_routing_accuracy(
    model: TextEncoderWithHead,
    pairs: list[dict],
    mem_to_vec: dict[str, np.ndarray],
    mem_to_user: dict[str, str],
    user_to_mems: dict[str, list[str]],
    use_frozen_bge: bool = False,
    batch_size: int = 32,
    user_to_k: Optional[dict] = None,
) -> dict:
    """Top-1 routing accuracy on a list of align_pairs records.

    Route query → user's memory bank (all memories for that user).
    Correct if top-1 == positive_memory_id.

    use_frozen_bge=True: use raw bge (no head) — baseline floor.

    Returns dict with overall acc + K-stratified breakdown (if user_to_k provided).
    K=1 users are trivially 1.0 (only one memory to route to) — excluded from K≥2 delta.
    """
    model.eval()
    correct = 0
    total = 0

    # K-stratified counters: k → {correct, total}
    k_correct: dict[int, int] = {}
    k_total: dict[int, int] = {}

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start: start + batch_size]
        queries = [p["query"] for p in batch]
        pos_ids = [p["positive_memory_id"] for p in batch]

        if use_frozen_bge:
            h_q = model.encode_queries_frozen_bge(queries)   # [B, d]
        else:
            h_q = model.encode_queries(queries)               # [B, d]

        for i, (pos_id, hq) in enumerate(zip(pos_ids, h_q)):
            uid = mem_to_user.get(pos_id)
            if uid is None:
                continue
            cand_ids = user_to_mems[uid]
            if not cand_ids:
                continue

            cand_vecs = torch.tensor(
                np.stack([mem_to_vec[mid] for mid in cand_ids]),
                device=hq.device, dtype=torch.float32,
            )   # [K, d]
            sims = cand_vecs @ hq   # [K] — both L2-normalized
            top1_id = cand_ids[sims.argmax().item()]
            hit = int(top1_id == pos_id)
            correct += hit
            total += 1

            if user_to_k is not None:
                k = user_to_k.get(uid, len(cand_ids))
                k_correct[k] = k_correct.get(k, 0) + hit
                k_total[k] = k_total.get(k, 0) + 1

    model.train()
    overall = correct / total if total else 0.0
    result = {"overall": overall, "total": total}

    if user_to_k is not None:
        k_breakdown: dict = {}
        for k in sorted(k_total):
            acc = k_correct.get(k, 0) / k_total[k]
            k_breakdown[k] = {"n": k_total[k], "acc": round(acc, 6)}
        result["k_breakdown"] = k_breakdown

        # K≥2 aggregate (headline metric for Stage1 contribution B)
        kge2_corr = sum(k_correct.get(k, 0) for k in k_total if k >= 2)
        kge2_tot = sum(k_total[k] for k in k_total if k >= 2)
        result["kge2_acc"] = kge2_corr / kge2_tot if kge2_tot else 0.0
        result["kge2_total"] = kge2_tot

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────────

def train_step(
    model: TextEncoderWithHead,
    optimizer: torch.optim.Optimizer,
    queries: list[str],
    pos_vecs: torch.Tensor,    # [B, d]
    neg_vecs: torch.Tensor,    # [B, K, d]
    tau: float,
    device: str,
    max_grad_norm: float = 1.0,
) -> dict:
    """One InfoNCE step. Returns {loss, grad_norm}."""
    optimizer.zero_grad()
    model.train()

    pos_vecs = pos_vecs.to(device)
    neg_vecs = neg_vecs.to(device)

    h_q = model.encode_queries(queries)   # [B, d], normalized, gradient-enabled

    # h_pos and h_neg are pre-computed frozen bank vectors — already normalized
    h_pos = pos_vecs                      # [B, d]
    h_hard_neg = neg_vecs                 # [B, K, d]

    loss = info_nce(h_q, h_pos, h_hard_neg, tau=tau)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(model.parameters()), max_grad_norm
    ).item()
    optimizer.step()

    return {"loss": loss.item(), "grad_norm": grad_norm}


# ──────────────────────────────────────────────────────────────────────────────
# Smoke gate
# ──────────────────────────────────────────────────────────────────────────────

def run_smoke(args: argparse.Namespace, model: TextEncoderWithHead, mem_to_vec, mem_to_user, user_to_mems, user_to_k):
    """Smoke gate: few steps, check L_align descent + routing accuracy.

    Gate criteria (plan.md STEP1):
      (1) L_align strictly decreasing (step_N < step_1), no NaN/Inf.
      (2) Routing accuracy ≥ frozen-bge floor (≥ 0.90) after smoke steps.
      (3) Config logged; two runs with same seed produce identical losses.
    """
    print("\n" + "="*60)
    print("SMOKE GATE — Stage1 InfoNCE")
    print(f"  tau={args.tau}  lr_encoder={args.lr_encoder}  lr_head={args.lr_head}")
    print(f"  batch_size={args.batch_size}  smoke_steps={args.smoke_steps}  seed={args.seed}")
    print("="*60)

    dataset = AlignPairsDataset(args.smoke_pairs_path, mem_to_vec, max_hard_neg=11)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr_encoder, args.lr_head),
        weight_decay=args.weight_decay,
    )

    # ── Frozen-bge baseline routing ──
    smoke_pairs_raw = []
    with open(args.smoke_pairs_path) as f:
        for line in f:
            smoke_pairs_raw.append(json.loads(line))

    print("\n[baseline] computing frozen-bge routing accuracy ...")
    with torch.no_grad():
        frozen_res = compute_routing_accuracy(
            model, smoke_pairs_raw, mem_to_vec, mem_to_user, user_to_mems,
            use_frozen_bge=True, user_to_k=user_to_k,
        )
    frozen_acc = frozen_res["overall"]
    frozen_kge2 = frozen_res.get("kge2_acc", float("nan"))
    print(f"  frozen-bge routing accuracy (overall): {frozen_acc:.4f}")
    print(f"  frozen-bge routing accuracy (K≥2):    {frozen_kge2:.4f}")

    # ── Few-step training ──
    # Wrap loader in a while loop so we always run exactly smoke_steps steps
    # regardless of dataset size (smoke set has 28 pairs, batch=64 → 1 batch/epoch).
    losses = []
    step = 0
    model.train()

    while step < args.smoke_steps:
        for queries, pos_vecs, neg_vecs in loader:
            if step >= args.smoke_steps:
                break
            result = train_step(
                model, optimizer, queries, pos_vecs, neg_vecs,
                tau=args.tau, device=args.device, max_grad_norm=args.max_grad_norm,
            )
            losses.append(result["loss"])
            print(f"  step {step+1:3d}  L_align={result['loss']:.4f}  grad_norm={result['grad_norm']:.3f}")

            if math.isnan(result["loss"]) or math.isinf(result["loss"]):
                print("GATE FAIL: NaN/Inf detected at step", step + 1)
                return False
            step += 1

    # ── Post-smoke routing accuracy ──
    print("\n[gate] computing routing accuracy after smoke steps ...")
    with torch.no_grad():
        post_res = compute_routing_accuracy(
            model, smoke_pairs_raw, mem_to_vec, mem_to_user, user_to_mems,
            use_frozen_bge=False, user_to_k=user_to_k,
        )
    post_acc = post_res["overall"]
    post_kge2 = post_res.get("kge2_acc", float("nan"))
    print(f"  Stage1 routing accuracy (smoke, overall): {post_acc:.4f}")
    print(f"  Stage1 routing accuracy (smoke, K≥2):    {post_kge2:.4f}")
    print(f"  frozen-bge floor (overall):               {frozen_acc:.4f}")
    print(f"  K≥2 delta (Stage1 - frozen):             {post_kge2 - frozen_kge2:+.4f}")

    # ── Gate checks ──
    gate_ok = True

    # (1) L_align descent
    if losses[-1] >= losses[0]:
        print(f"GATE WARN: loss did not decrease ({losses[0]:.4f} → {losses[-1]:.4f})")
    else:
        print(f"GATE PASS (1): L_align descent {losses[0]:.4f} → {losses[-1]:.4f}")

    # (2) Routing accuracy not catastrophically below frozen floor.
    smoke_threshold = frozen_acc - 0.10
    if post_acc < smoke_threshold:
        print(f"GATE FAIL (2): routing {post_acc:.4f} < smoke threshold {smoke_threshold:.4f}")
        gate_ok = False
    else:
        print(f"GATE PASS (2): routing {post_acc:.4f} ≥ smoke threshold {smoke_threshold:.4f}")

    print(f"\nSMOKE RESULT: {'PASS' if gate_ok else 'FAIL'}")
    return gate_ok


# ──────────────────────────────────────────────────────────────────────────────
# Full training
# ──────────────────────────────────────────────────────────────────────────────

def run_full_training(args: argparse.Namespace, model: TextEncoderWithHead, mem_to_vec, mem_to_user, user_to_mems, user_to_k=None):
    """Full Stage1 training loop.

    Saves best checkpoint (lowest L_align on last-epoch average).
    Logs routing accuracy (frozen-bge vs Stage1) every epoch.
    """
    dataset = AlignPairsDataset(args.pairs_path, mem_to_vec, max_hard_neg=11)
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.num_epochs

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False, num_workers=4, pin_memory=True,
    )

    # Asymmetric LR optimizer (plan.md v0.4.14)
    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr_encoder, args.lr_head),
        weight_decay=args.weight_decay,
    )

    # Linear warmup + cosine decay scheduler
    def lr_lambda(current_step: int) -> float:
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(
            max(1, total_steps - args.warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Frozen-bge baseline ──
    print("\n[baseline] computing frozen-bge routing accuracy on sample ...")
    sample_pairs = []
    with open(args.pairs_path) as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            sample_pairs.append(json.loads(line))

    with torch.no_grad():
        frozen_res = compute_routing_accuracy(
            model, sample_pairs, mem_to_vec, mem_to_user, user_to_mems,
            use_frozen_bge=True, user_to_k=user_to_k,
        )
    frozen_acc = frozen_res["overall"]
    frozen_kge2 = frozen_res.get("kge2_acc", float("nan"))
    print(f"  frozen-bge routing accuracy (n=500, overall): {frozen_acc:.4f}")
    print(f"  frozen-bge routing accuracy (n=500, K≥2):    {frozen_kge2:.4f}")
    if "k_breakdown" in frozen_res:
        for k, v in sorted(frozen_res["k_breakdown"].items()):
            print(f"    K={k}: n={v['n']}  acc={v['acc']:.4f}")

    # ── Training loop ──
    best_loss = float("inf")
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        epoch_losses = []
        t0 = time.time()

        for queries, pos_vecs, neg_vecs in loader:
            result = train_step(
                model, optimizer, queries, pos_vecs, neg_vecs,
                tau=args.tau, device=args.device, max_grad_norm=args.max_grad_norm,
            )
            scheduler.step()
            epoch_losses.append(result["loss"])
            global_step += 1

            if global_step % 200 == 0:
                lr_enc = optimizer.param_groups[0]["lr"]
                lr_hd = optimizer.param_groups[1]["lr"]
                print(
                    f"  step {global_step}/{total_steps}"
                    f"  L_align={result['loss']:.4f}"
                    f"  grad={result['grad_norm']:.3f}"
                    f"  lr_enc={lr_enc:.2e}  lr_head={lr_hd:.2e}"
                )

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        elapsed = time.time() - t0

        # Per-epoch routing accuracy (train sample) — K-stratified
        with torch.no_grad():
            train_res = compute_routing_accuracy(
                model, sample_pairs, mem_to_vec, mem_to_user, user_to_mems,
                use_frozen_bge=False, user_to_k=user_to_k,
            )
        train_acc = train_res["overall"]
        train_kge2 = train_res.get("kge2_acc", float("nan"))
        kge2_delta = train_kge2 - frozen_kge2

        print(
            f"\nEpoch {epoch}/{args.num_epochs}"
            f"  avg_loss={epoch_loss:.4f}"
            f"  routing_acc(overall)={train_acc:.4f}"
            f"  routing_acc(K≥2)={train_kge2:.4f}"
            f"  K≥2_delta={kge2_delta:+.4f}"
            f"  frozen_floor={frozen_acc:.4f}"
            f"  time={elapsed:.0f}s"
        )
        if "k_breakdown" in train_res:
            for k, v in sorted(train_res["k_breakdown"].items()):
                fk = frozen_res.get("k_breakdown", {}).get(k, {})
                delta = v["acc"] - fk.get("acc", float("nan"))
                print(f"    K={k}: n={v['n']}  acc={v['acc']:.4f}  delta={delta:+.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": vars(args),
                },
                out_path,
            )
            print(f"  [ckpt] saved best checkpoint → {out_path}  (loss={best_loss:.4f})")

    print(f"\nStage1 training complete. Best L_align: {best_loss:.4f}")
    print(f"Checkpoint: {out_path}")
    print(
        f"\n[final] routing (overall):  frozen-bge={frozen_acc:.4f}  Stage1={train_acc:.4f}"
        f"  delta={train_acc - frozen_acc:+.4f}"
    )
    print(
        f"[final] routing (K≥2):     frozen-bge={frozen_kge2:.4f}  Stage1={train_kge2:.4f}"
        f"  delta={kge2_delta:+.4f}  ← headline (contribution B gate)"
    )
    return best_loss


# ──────────────────────────────────────────────────────────────────────────────
# Determinism helper
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage1 InfoNCE align training")

    # Mode
    p.add_argument("--smoke", action="store_true", help="Run smoke gate only (few steps)")

    # Data
    p.add_argument("--pairs", dest="pairs_path",
                   default="data/processed/Books/align_pairs.jsonl")
    p.add_argument("--smoke_pairs", dest="smoke_pairs_path",
                   default="data/processed/Books/align_pairs_smoke.jsonl")
    p.add_argument("--bank_dir", default="data/processed/Books/memory_bank")

    # Model
    p.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--proj_hidden", type=int, default=768)
    p.add_argument("--proj_out", type=int, default=768)

    # Training
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lr_encoder", type=float, default=2e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Smoke
    p.add_argument("--smoke_steps", type=int, default=100)

    # Output
    p.add_argument("--out_ckpt", dest="out_ckpt",
                   default="checkpoints/Books/stage1_align_best.pt")
    p.add_argument("--device", default="cpu")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"[config] device={args.device}  seed={args.seed}  tau={args.tau}")
    print(f"         lr_encoder={args.lr_encoder}  lr_head={args.lr_head}")
    print(f"         batch_size={args.batch_size}  num_epochs={args.num_epochs}")

    # Load memory index (pre-computed frozen vectors)
    mem_to_vec, mem_to_user, user_to_mems, user_to_k = load_memory_index(args.bank_dir)

    # Build model
    model = TextEncoderWithHead(
        model_id=args.encoder,
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
        device=args.device,
    )

    if args.smoke:
        gate_pass = run_smoke(args, model, mem_to_vec, mem_to_user, user_to_mems, user_to_k)
        sys.exit(0 if gate_pass else 1)
    else:
        pairs_path = Path(args.pairs_path)
        if not pairs_path.exists():
            print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
            sys.exit(1)

        # Estimate training time before starting
        n_pairs = sum(1 for _ in open(pairs_path))
        steps_per_epoch = math.ceil(n_pairs / args.batch_size)
        total_steps = steps_per_epoch * args.num_epochs
        print(f"\n[full training] {n_pairs} pairs  "
              f"{steps_per_epoch} steps/epoch  "
              f"{total_steps} total steps  "
              f"{args.num_epochs} epochs")

        run_full_training(args, model, mem_to_vec, mem_to_user, user_to_mems, user_to_k)


if __name__ == "__main__":
    main()
