"""Stage2: F7 hybrid training skeleton.

Architecture (plan.md v0.4.5 §3 M3/M4):
  Frozen F6a SASRec + trainable text_encoder (bge-base) + trainable Projector
  Loss: L = L_retrieval + α · L_align
    L_retrieval = BPR(pos_logits, neg_logits)   (from SASRec.forward)
    L_align = 1 - cos_sim(prefix[:,0,:], item_emb(pos_target).detach())

  α grid (plan.md §7 #9): {0.01, 0.1, 0.5} — start low to avoid alignment
  swamping retrieval signal. Both raw loss sizes + grad norm logged per step.

On-the-fly re-encoding (plan.md §7 #11):
  h_query and h_memory re-encoded every step via current text_encoder
  (no cached vectors) because text_encoder is being fine-tuned.

Usage — CPU smoke (20-user mock):
    docker exec -e PYTHONPATH=/qaim-rec qaim-rec python3 \\
        src/training/train_hybrid.py --smoke

Usage — full training (after bank build, GPU):
    docker exec -e PYTHONPATH=/qaim-rec qaim-rec python3 \\
        src/training/train_hybrid.py \\
        --bank_jsonl data/memory_full/memory_full.jsonl \\
        --checkpoint checkpoints/Books/sasrec_pretrain.pt \\
        --category Books --device cuda:0 --alpha 0.1 --num_epochs 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data

# BGE query prefix (matches embed.py convention)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Text encoder wrapper (gradient-enabled for Stage2)

class TextEncoder(nn.Module):
    """Thin wrapper around bge-base transformer for Stage2 fine-tuning.

    Unlike EmbeddingModel (inference, numpy output), this module keeps
    grad enabled so the transformer weights update with the projector.
    """

    def __init__(self, model_id: str = "BAAI/bge-base-en-v1.5", device: str = "cpu") -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self._model = AutoModel.from_pretrained(model_id, local_files_only=True)
        self.device = device
        self._model.to(device)

    @property
    def dim(self) -> int:
        return self._model.config.hidden_size

    def encode(self, texts: list[str], is_query: bool = False) -> torch.Tensor:
        """Encode texts → [N, d_text] float32 tensor (gradient-enabled).

        Returns L2-normalized vectors, same convention as EmbeddingModel.
        """
        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        enc = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        out = self._model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vecs, p=2, dim=-1)  # [N, d_text]


# ---------------------------------------------------------------------------
# Bank mock builder

def _load_mock_from_jsonl(jsonl_path: str, n: int = 5) -> list[dict]:
    """Load first n users from bank JSONL as mock records.

    Returns list of {user_id, k_personal, source_texts (flat list)}.
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Flatten all source_texts across cluster_summaries
            texts = []
            for cs in rec.get("cluster_summaries", []):
                texts.extend(cs.get("source_texts", []))
            if texts:
                records.append({
                    "user_id": rec["user_id"],
                    "k_personal": rec["k_personal"],
                    "source_texts": texts,
                })
            if len(records) >= n:
                break
    return records


# ---------------------------------------------------------------------------
# BPR loss

def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """BPR loss over non-padding positions (pos != 0)."""
    mask = (pos_logits != 0).float()
    loss = -F.logsigmoid(pos_logits - neg_logits)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Alignment loss

def align_loss(
    prefix_embeds: torch.Tensor,   # [B, 1, d_sasrec]
    item_emb: nn.Embedding,        # SASRec item embedding (frozen)
    pos_target_ids: torch.Tensor,  # [B] — last positive item per user
) -> torch.Tensor:
    """1 - cos_sim(prefix[:,0,:], item_emb(pos_target).detach()).

    Pushes the prefix toward the target item in SASRec space.
    item_emb is detached so gradients only flow to prefix / projector.
    """
    prefix_vec = prefix_embeds[:, 0, :]                      # [B, d_sasrec]
    target_vec = item_emb(pos_target_ids).detach()            # [B, d_sasrec]
    cos = F.cosine_similarity(prefix_vec, target_vec, dim=-1) # [B]
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# One training step

def train_step(
    model: SASRec,
    text_encoder: TextEncoder,
    projector: IntentProjector,
    optimizer: torch.optim.Optimizer,
    query_texts: list[str],         # [B] query stand-ins
    memory_texts: list[str],        # [B] memory source texts
    wrong_memory_texts: list[str],  # [B] swapped memory source texts
    seq_np: np.ndarray,             # [B, maxlen]
    pos_np: np.ndarray,             # [B, maxlen]
    neg_np: np.ndarray,             # [B, maxlen]
    alpha: float,
    device: str,
) -> dict:
    """Single Stage2 training step. Returns {L, L_retrieval, L_align, grad_norm}."""
    optimizer.zero_grad()

    # On-the-fly encode (§7 #11) — text_encoder is in training mode
    h_query = text_encoder.encode(query_texts, is_query=True)    # [B, d_text]
    h_memory = text_encoder.encode(memory_texts, is_query=False)  # [B, d_text]

    # Projector → prefix
    prefix_embeds = projector(h_query, h_memory)  # [B, 1, d_sasrec]

    # Frozen SASRec forward
    B = seq_np.shape[0]
    uid_dummy = np.zeros(B, dtype=np.int32)
    pos_logits, neg_logits = model(
        uid_dummy, seq_np, pos_np, neg_np,
        prefix_embeds=prefix_embeds,
    )

    # L_retrieval: BPR
    pos_t = torch.LongTensor(pos_np).to(device)
    neg_t = torch.LongTensor(neg_np).to(device)
    # Recompute with tensor inputs for loss (pos_logits already computed above)
    l_retrieval = bpr_loss(pos_logits, neg_logits)

    # L_align: cosine distance between prefix and last positive item
    last_pos = torch.LongTensor(pos_np[:, -1]).to(device)  # [B]
    l_align = align_loss(prefix_embeds, model.item_emb, last_pos)

    # Combined loss
    loss = l_retrieval + alpha * l_align
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in list(text_encoder.parameters()) + list(projector.parameters())
        if p.grad is not None
    ) ** 0.5

    optimizer.step()

    return {
        "L": round(loss.item(), 6),
        "L_retrieval": round(l_retrieval.item(), 6),
        "L_align": round(l_align.item(), 6),
        "grad_norm": round(grad_norm, 6),
    }


# ---------------------------------------------------------------------------
# Smoke test

def run_smoke(args: SimpleNamespace) -> None:
    """CPU smoke: mock bank → 1 step × 3 α values.

    Verifies: (a) prefix forward with correct P offset,
              (b) 4 conditions produce different outputs,
              (c) loss/backprop works,
              (d) α-wise loss size logging.
    """
    print("[smoke] Loading mock bank ...")
    mock_users = _load_mock_from_jsonl(
        "data/memory_full_test/memory_b_u20_seed42.jsonl", n=5
    )
    assert len(mock_users) >= 2, "Need at least 2 users for wrong-condition swap"
    print(f"[smoke] {len(mock_users)} mock users loaded")

    print("[smoke] Loading text encoder (bge-base, CPU) ...")
    text_enc = TextEncoder(device="cpu")
    print(f"[smoke] text_encoder.dim = {text_enc.dim}")

    # Load F6a checkpoint
    print("[smoke] Loading F6a checkpoint ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt["args"]
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)
    eval_args = SimpleNamespace(
        maxlen=saved.maxlen, hidden_units=saved.hidden_units,
        num_blocks=saved.num_blocks, num_heads=saved.num_heads,
        dropout_rate=saved.dropout_rate, norm_first=saved.norm_first,
        device="cpu",
    )
    dataset = load_data(args.category, args.data_dir)
    model = SASRec(dataset["usernum"], dataset["itemnum"], eval_args)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)  # frozen
    model.train()  # train mode for dropout consistency
    print("[smoke] SASRec frozen.")

    d_sasrec = saved.hidden_units
    d_text = text_enc.dim
    projector = IntentProjector(d_text=d_text, d_sasrec=d_sasrec)
    print(f"[smoke] Projector: [{2*d_text}→256→{d_sasrec}]  params={sum(p.numel() for p in projector.parameters()):,}")

    # Build mock batch (B=2)
    B = 2
    query_texts = [mock_users[i]["source_texts"][0] for i in range(B)]
    memory_texts = [mock_users[i]["source_texts"][0] for i in range(B)]
    wrong_texts = [mock_users[(i + 1) % B]["source_texts"][0] for i in range(B)]

    # Fake sequences — random items within valid range
    rng = random.Random(42)
    itemnum = dataset["itemnum"]
    maxlen = saved.maxlen
    seq_np = np.zeros((B, maxlen), dtype=np.int32)
    pos_np = np.zeros((B, maxlen), dtype=np.int32)
    neg_np = np.zeros((B, maxlen), dtype=np.int32)
    for b in range(B):
        # Fill last 5 positions with random items
        for j in range(maxlen - 5, maxlen):
            item = rng.randint(1, itemnum)
            seq_np[b, j] = item
            pos_np[b, j] = rng.randint(1, itemnum)
            neg_np[b, j] = rng.randint(1, itemnum)

    # ── (a) Prefix forward with P offset ────────────────────────────────────
    print("\n[smoke] (a) Testing prefix P offset ...")
    with torch.no_grad():
        h_q = text_enc.encode(query_texts, is_query=True)
        h_m = text_enc.encode(memory_texts, is_query=False)
    prefix = projector(h_q, h_m)  # [B, 1, d_sasrec]
    assert prefix.shape == (B, 1, d_sasrec), f"Wrong prefix shape: {prefix.shape}"
    with torch.no_grad():
        log_feats, P = model.log2feats(seq_np, prefix_embeds=prefix)
    assert P == 1, f"Expected P=1, got P={P}"
    assert log_feats.shape[1] == maxlen + 1, \
        f"Expected seq_len={maxlen+1}, got {log_feats.shape[1]}"
    print(f"  prefix.shape={prefix.shape}  P={P}  log_feats.shape={log_feats.shape}  OK")

    # ── (b) 4 conditions produce different outputs ───────────────────────────
    print("\n[smoke] (b) Testing 4 conditions produce different outputs ...")
    from src.eval.conditions import make_prefix

    condition_outputs = {}
    with torch.no_grad():
        for cond in ("vanilla", "correct", "wrong", "default"):
            pfx = make_prefix(
                condition=cond,
                projector=projector,
                default_intent=model.default_intent,
                h_query=h_q,
                h_memory_correct=h_m,
                h_memory_wrong=text_enc.encode(wrong_texts, is_query=False),
                device="cpu",
            )
            lf, _ = model.log2feats(seq_np, prefix_embeds=pfx)
            last = lf[0, -1, :].clone()
            condition_outputs[cond] = last

    # All 4 must differ
    cond_list = list(condition_outputs.keys())
    all_different = all(
        not torch.allclose(condition_outputs[a], condition_outputs[b], atol=1e-5)
        for i, a in enumerate(cond_list) for b in cond_list[i+1:]
    )
    for c, v in condition_outputs.items():
        print(f"  {c:8s}: final_feat norm={v.norm().item():.4f}")
    assert all_different, "Some conditions produced identical outputs!"
    print("  All 4 conditions differ: OK")

    # ── (c) + (d) Loss/backprop and α logging ───────────────────────────────
    print("\n[smoke] (c)+(d) Loss/backprop + α logging ...")
    alpha_grid = [0.01, 0.1, 0.5]
    for alpha in alpha_grid:
        proj_fresh = IntentProjector(d_text=d_text, d_sasrec=d_sasrec)
        opt = torch.optim.Adam(
            list(text_enc.parameters()) + list(proj_fresh.parameters()),
            lr=1e-4,
        )
        metrics = train_step(
            model=model,
            text_encoder=text_enc,
            projector=proj_fresh,
            optimizer=opt,
            query_texts=query_texts,
            memory_texts=memory_texts,
            wrong_memory_texts=wrong_texts,
            seq_np=seq_np,
            pos_np=pos_np,
            neg_np=neg_np,
            alpha=alpha,
            device="cpu",
        )
        print(f"  α={alpha:<4}  L={metrics['L']:.5f}  "
              f"L_ret={metrics['L_retrieval']:.5f}  "
              f"L_align={metrics['L_align']:.5f}  "
              f"grad_norm={metrics['grad_norm']:.5f}")
        assert metrics["L"] > 0, "Loss should be positive"
        assert metrics["grad_norm"] > 0, "Grad norm should be non-zero"
    print("  Loss/backprop OK, α scaling reflected in total L.")

    print("\n[smoke] ALL CHECKS PASSED (CPU, mock data — not real results)")


# ---------------------------------------------------------------------------
# Full training loop skeleton

def run_training(args: SimpleNamespace) -> None:
    raise NotImplementedError(
        "Full training loop to be wired after bank build + pseudo-query generation (P2). "
        "Run with --smoke to verify skeleton."
    )


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run CPU smoke test")
    parser.add_argument("--bank_jsonl", type=str, default=None)
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/Books/sasrec_pretrain.pt")
    parser.add_argument("--category", type=str, default="Books")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    cli = parser.parse_args()

    if cli.smoke or cli.bank_jsonl is None:
        run_smoke(cli)
    else:
        run_training(cli)


if __name__ == "__main__":
    main()
