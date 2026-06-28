"""STEP 1 scale diagnostic: h_intent norm vs SASRec item_emb norm.

Measures:
  1. item_emb raw norm + scaled norm (×√d) — what SASRec actually sees in attention
  2. projector output (h_intent) norm from bank samples — what prefix sees
  3. scale ratio = item_scaled_norm / h_intent_norm
  4. Simulated prefix attention weight fraction at layer-0

Usage:
    python3 scripts/diag_scale.py \
        --sasrec_ckpt checkpoints/Books/sasrec_pretrain.pt \
        --stage2_ckpt checkpoints/Books/stage2_stage1enc_best.pt \
        --bank_jsonl data/memory/Books/f3_bank.jsonl \
        --category Books --n_samples 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.sasrec import SASRec
from src.models.projector import IntentProjector

# ── BGE import (optional; fallback to projector weight-only analysis) ──
try:
    from transformers import AutoModel, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_bank_texts(jsonl_path: str, n: int = 200) -> list[str]:
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "intent_description" in rec:
                texts.append(rec["intent_description"])
            elif "cluster_summaries" in rec:
                for s in rec["cluster_summaries"].values():
                    texts.append(s)
            if len(texts) >= n:
                break
    return texts[:n]


def bge_encode(texts: list[str], model, tok, device: str, is_query: bool = False) -> torch.Tensor:
    if is_query:
        texts = [_BGE_QUERY_PREFIX + t for t in texts]
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(vecs, p=2, dim=-1)


def simulate_prefix_attention_weight(
    item_norm: float, prefix_norm: float, d: int, seq_len: int = 10
) -> float:
    """Rough analytical estimate: in QK dot-product attention, weight ∝ exp(q·k/√d).
    Assume Q comes from a random sequence token, and we compare:
      - prefix key: norm ≈ prefix_norm
      - history key: norm ≈ item_norm (already scaled by √d in SASRec)
    The ratio of attention allocated to the prefix token (P=1) vs history (L tokens)
    in expectation under uniform random Q direction:
      softmax score ∝ |q||k|/√d  (expected dot product between random unit vectors ∝ 0)
      but variance of dot product ∝ |k|²/d
    A simple proxy: share = 1 / (1 + L × (item_norm/prefix_norm)²)
    (from attention concentration result — larger K norm → higher expected max attention)
    """
    ratio = (item_norm / (prefix_norm + 1e-9)) ** 2
    share = 1.0 / (1.0 + seq_len * ratio)
    return share


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_ckpt", default="checkpoints/Books/sasrec_pretrain.pt")
    parser.add_argument("--stage2_ckpt", default="checkpoints/Books/stage2_stage1enc_best.pt")
    parser.add_argument("--bank_jsonl", default="data/memory/Books/f3_bank.jsonl")
    parser.add_argument("--category", default="Books")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--bge_model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[diag] device={args.device}")

    # ── 1. Load SASRec + measure item_emb norms ──────────────────────────
    ckpt = torch.load(args.sasrec_ckpt, map_location="cpu")
    # Standard SASRec checkpoint format: {'model_state_dict': {...}, ...}
    sd = ckpt.get("model_state_dict", ckpt)
    item_emb_w = sd.get("item_emb.weight", None)

    if item_emb_w is not None:
        d = item_emb_w.shape[1]
        raw_norms = item_emb_w[1:].norm(dim=-1)  # skip padding idx 0
        scaled_norms = raw_norms * (d ** 0.5)
        print(f"\n[item_emb]  d={d}  n_items={item_emb_w.shape[0]-1}")
        print(f"  raw norm   : mean={raw_norms.mean():.4f}  std={raw_norms.std():.4f}  "
              f"p5={raw_norms.quantile(0.05):.4f}  p95={raw_norms.quantile(0.95):.4f}")
        print(f"  scaled norm: mean={scaled_norms.mean():.4f}  std={scaled_norms.std():.4f}  "
              f"p5={scaled_norms.quantile(0.05):.4f}  p95={scaled_norms.quantile(0.95):.4f}")
        print(f"  (SASRec multiplies ×√d={d**0.5:.1f} before attention)")
    else:
        print("[warn] Could not find item_emb.weight")
        d = 256
        scaled_norms = None

    # ── 2. Load projector from stage2 ckpt + measure h_intent norms ──────
    stage2_ckpt = torch.load(args.stage2_ckpt, map_location="cpu")
    # Standard stage2 format: {'projector_state': {...}, ...}
    proj_state = stage2_ckpt.get("projector_state", None)
    if proj_state is None:
        # Fallback: scan for projector-like keys
        proj_state = {}
        for k, v in stage2_ckpt.items():
            if k.startswith("projector."):
                proj_state[k[len("projector."):]] = v

    if proj_state:
        proj = IntentProjector(d_text=768, d_sasrec=d, hidden_dim=256)
        missing, unexpected = proj.load_state_dict(proj_state, strict=False)
        proj.eval()
        print(f"\n[projector] loaded from {args.stage2_ckpt}")
        if missing:
            print(f"  missing keys: {missing[:5]}")

        # Projector last-layer weight norm as proxy for output scale
        last_w = proj_state.get("net.4.weight", None)
        if last_w is not None:
            col_norms = last_w.norm(dim=0)
            print(f"  last Linear weight col norms: mean={col_norms.mean():.4f}  "
                  f"std={col_norms.std():.4f}")
            # Expected output norm ≈ ||W||_F / sqrt(d_in) × input_norm (rough)
            # But input_norm after LayerNorm+GELU ≈ 1
            expected_output_norm = last_w.norm(dim=-1).mean()
            print(f"  expected h_intent norm (‖Wx‖ approx): {expected_output_norm:.4f}")
    else:
        print(f"\n[projector] no projector state found in {args.stage2_ckpt}")
        proj = None

    # ── 3. Encode bank samples and get ACTUAL h_intent norms ─────────────
    if _HAS_TRANSFORMERS:
        print(f"\n[encoding] loading bge from {args.bge_model} ...")
        try:
            tok = AutoTokenizer.from_pretrained(args.bge_model, local_files_only=True)
            bge = AutoModel.from_pretrained(args.bge_model, local_files_only=True)
            bge.eval()
            has_bge = True
        except Exception as e:
            print(f"  [warn] could not load bge: {e}")
            has_bge = False
    else:
        has_bge = False

    if has_bge and proj is not None:
        texts = load_bank_texts(args.bank_jsonl, n=args.n_samples)
        print(f"  loaded {len(texts)} bank texts")

        # Encode in batches of 32
        h_list = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            h = bge_encode(batch, bge, tok, device="cpu", is_query=False)  # [B, 768]
            h_list.append(h)
        h_mem = torch.cat(h_list, dim=0)  # [N, 768]

        # Use h_mem as both query and memory (proxy — real query would be user history)
        with torch.no_grad():
            h_intent = proj(h_mem, h_mem)  # [N, 1, 256]
        h_intent = h_intent.squeeze(1)   # [N, 256]

        intent_norms = h_intent.norm(dim=-1)
        print(f"\n[h_intent]  n={len(intent_norms)}")
        print(f"  norm: mean={intent_norms.mean():.4f}  std={intent_norms.std():.4f}  "
              f"p5={intent_norms.quantile(0.05):.4f}  p95={intent_norms.quantile(0.95):.4f}")

        if scaled_norms is not None:
            ratio = scaled_norms.mean().item() / (intent_norms.mean().item() + 1e-9)
            print(f"\n[SCALE RATIO] item_scaled_norm / h_intent_norm = {ratio:.2f}×")
            if ratio > 3:
                print("  ⚠  item embedding dominates — prefix is effectively SMALL in attention space")
            elif ratio < 0.3:
                print("  ⚠  prefix dominates — may suppress history signal")
            else:
                print("  ✓  norms roughly matched (within 3×)")

            # Prefix attention weight estimate
            attn_share = simulate_prefix_attention_weight(
                item_norm=scaled_norms.mean().item(),
                prefix_norm=intent_norms.mean().item(),
                d=d,
                seq_len=10,
            )
            print(f"\n[prefix attention share estimate]")
            print(f"  P=1 prefix vs L=10 history: prefix gets ~{attn_share*100:.1f}% of attention")
            print(f"  (analytic proxy — actual values depend on trained Q directions)")
    else:
        # Fallback: use projector weight-based estimate only
        print("\n[h_intent] BGE unavailable — using weight-based norm estimate only")
        if proj is not None and scaled_norms is not None:
            # Use expected output norm from above
            last_w = proj_state.get("net.4.weight")
            if last_w is not None:
                expected_norm = last_w.norm(dim=-1).mean().item()
                ratio = scaled_norms.mean().item() / (expected_norm + 1e-9)
                print(f"  [SCALE RATIO estimate] item_scaled / h_intent ≈ {ratio:.2f}×")

    print("\n[diag] done.")


if __name__ == "__main__":
    main()
