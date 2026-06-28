"""Verify item_target_norm fix: padding-excluded mean vs old padding-included mean.

Loads a real SASRec checkpoint, builds a representative batch from the Books
dataset, and prints:
  - old item_target_norm (padding-included mean, the bug)
  - new item_target_norm (non-padding mean, the fix)
  - prefix attention share estimate for both
  - per-user sparsity (fraction of non-padding positions)

Usage:
    python3 scripts/verify_scale_fix.py \
        --sasrec_ckpt checkpoints/Books/sasrec_pretrain.pt \
        --category Books --n_users 500
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.sasrec import SASRec
from src.models.dataloader import load_data


def make_seq_batch(user_train: dict, all_users: list[int], maxlen: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    chosen = rng.choice(all_users, size=min(n, len(all_users)), replace=False)
    seqs = np.zeros((len(chosen), maxlen), dtype=np.int32)
    for i, u in enumerate(chosen):
        items = user_train.get(int(u), [])
        start = max(0, len(items) - maxlen)
        chunk = items[start:]
        offset = maxlen - len(chunk)
        for j, it in enumerate(chunk):
            seqs[i, offset + j] = it
    return seqs


def simulate_attention_share(item_norm: float, prefix_norm: float, L: int = 10) -> float:
    """Proxy: share = 1 / (1 + L × (item_norm / prefix_norm)²)"""
    r = (item_norm / (prefix_norm + 1e-9)) ** 2
    return 1.0 / (1.0 + L * r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec_ckpt", default="checkpoints/Books/sasrec_pretrain.pt")
    parser.add_argument("--category", default="Books")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--n_users", type=int, default=500)
    args = parser.parse_args()

    ckpt = torch.load(args.sasrec_ckpt, map_location="cpu")
    saved = ckpt["args"]
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)

    model_args = SimpleNamespace(
        maxlen=saved.maxlen, hidden_units=saved.hidden_units,
        num_blocks=saved.num_blocks, num_heads=saved.num_heads,
        dropout_rate=0.0,  # no dropout for deterministic norms
        norm_first=saved.norm_first, device="cpu",
    )
    dataset = load_data(args.category, args.data_dir)
    model = SASRec(dataset["usernum"], dataset["itemnum"], model_args)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_users = [u for u in range(1, dataset["usernum"] + 1) if dataset["user_train"].get(u)]
    seq_np = make_seq_batch(dataset["user_train"], all_users, saved.maxlen, args.n_users)
    print(f"[batch] shape={seq_np.shape}  maxlen={saved.maxlen}  d={saved.hidden_units}")

    with torch.no_grad():
        seqs = model.item_emb(torch.LongTensor(seq_np))
        seqs = seqs * (saved.hidden_units ** 0.5)
        poss = np.tile(np.arange(1, seq_np.shape[1] + 1), [seq_np.shape[0], 1])
        poss = poss * (seq_np != 0)
        seqs = seqs + model.pos_emb(torch.LongTensor(poss))
        # No dropout (rate=0.0) — norms are deterministic

    item_norms = seqs.norm(dim=-1)  # [B, L]

    # ── Old: padding-included mean (the bug) ────────────────────────────────
    old_norm = item_norms.mean().item()

    # ── New: non-padding mean (the fix) ─────────────────────────────────────
    nonpad_mask = item_norms > 1e-6
    new_norm = item_norms[nonpad_mask].mean().item() if nonpad_mask.any() else old_norm

    # ── Sparsity ─────────────────────────────────────────────────────────────
    sparsity = 1.0 - nonpad_mask.float().mean().item()
    nonpad_per_user = nonpad_mask.float().sum(dim=1)
    print(f"\n[sparsity]  padding fraction = {sparsity*100:.1f}%  "
          f"(non-pad per user: mean={nonpad_per_user.mean():.1f}  "
          f"p5={nonpad_per_user.quantile(0.05):.0f}  "
          f"p95={nonpad_per_user.quantile(0.95):.0f})")

    print(f"\n[item_target_norm]")
    print(f"  OLD (padding-included mean, BUG): {old_norm:.4f}")
    print(f"  NEW (non-padding mean,      FIX): {new_norm:.4f}")
    print(f"  ratio NEW/OLD: {new_norm/old_norm:.1f}×")

    # Projector LayerNorm output norm ≈ sqrt(d_sasrec) heuristic
    d = saved.hidden_units
    projector_approx_norm = d ** 0.5  # LayerNorm output: unit-ish × sqrt(d)
    print(f"\n[projector output norm approx] √d = {projector_approx_norm:.1f}")

    # The rescaling sets prefix_norm := item_target_norm.
    # But what ACTUALLY matters is prefix_norm vs ACTUAL item norms in the sequence.
    # OLD bug: prefix rescaled to old_norm (0.2573), but actual items have norm new_norm (1.7236)
    #          → prefix is UNDERSCALED relative to actual items
    # NEW fix: prefix rescaled to new_norm (1.7236), matches actual item norms
    L_typical = nonpad_per_user.median().int().item()
    L_typical = max(L_typical, 1)

    share_old_vs_actual = simulate_attention_share(
        item_norm=new_norm,    # actual item norm (non-padding)
        prefix_norm=old_norm,  # what prefix was rescaled to (old bug)
        L=L_typical,
    )
    share_new_vs_actual = simulate_attention_share(
        item_norm=new_norm,    # actual item norm
        prefix_norm=new_norm,  # what prefix is rescaled to (fix)
        L=L_typical,
    )

    print(f"\n[ACTUAL prefix attention share — prefix_norm vs real item_norm, L={L_typical}]")
    print(f"  OLD bug: prefix_target={old_norm:.4f}  actual_item={new_norm:.4f}  → share={share_old_vs_actual*100:.2f}%")
    print(f"  NEW fix: prefix_target={new_norm:.4f}  actual_item={new_norm:.4f}  → share={share_new_vs_actual*100:.1f}%")
    print(f"  Improvement: {share_new_vs_actual/(share_old_vs_actual+1e-9):.0f}× more prefix influence")

    print(f"\n[CONCLUSION]")
    print(f"  Padding fraction = {sparsity*100:.1f}% → old mean diluted {new_norm/old_norm:.1f}× below true item norm.")
    print(f"  OLD: prefix rescaled to {old_norm:.4f} while items are at {new_norm:.4f}")
    print(f"       → prefix was {new_norm/old_norm:.1f}× UNDERSIZED → ~{share_old_vs_actual*100:.2f}% attention share")
    print(f"  NEW: prefix rescaled to {new_norm:.4f} matching items")
    print(f"       → ~{share_new_vs_actual*100:.1f}% attention share (intended ~9.1%)")
    print(f"")
    print(f"  Training was running with prefix ~{share_old_vs_actual*100:.2f}% attention share.")
    print(f"  Projector learned to produce direction under near-zero influence → weights are arbitrary.")
    print(f"  → RETRAIN REQUIRED after this fix.")


if __name__ == "__main__":
    main()
