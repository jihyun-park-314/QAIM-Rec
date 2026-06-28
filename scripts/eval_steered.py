"""STEP 2: First real steered Recall measurement using existing checkpoints.

Runs 3 conditions on the *same* SASRec backbone (sasrec_pretrain.pt):
  vanilla   — no prefix (baseline)
  steered-2a — stage2_stage1enc_best.pt  (stage1 encoder + projector)
  steered-2b — stage2_rawbge_best.pt     (raw-BGE + projector)

The prefix path here is identical to train_hybrid.py training:
  query text → text_encoder.encode() → h_query
  bank.route(h_query, top-1) → h_memory
  projector(h_query, h_memory) → prefix_embeds [1, 1, d_sasrec]
  model.log2feats(seq_np, prefix_embeds=prefix_embeds) → use [-1] position

Usage (inside docker or with PYTHONPATH set):
    python scripts/eval_steered.py \\
        --category Books \\
        --sasrec_ckpt checkpoints/Books/sasrec_pretrain.pt \\
        --ckpt_2a checkpoints/Books/stage2_stage1enc_best.pt \\
        --ckpt_2b checkpoints/Books/stage2_rawbge_best.pt \\
        --bank_jsonl data/memory/Books/f3_bank.jsonl \\
        --pairs_jsonl data/processed/Books/align_pairs.jsonl \\
        --device cuda:0 \\
        --split test
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data
from src.eval.full_ranking import evaluate_full, evaluate_full_with_ranks, print_metrics
from src.training.train_hybrid import (
    TextEncoder,
    load_stage1_weights,
    _load_bank_full,
    _load_pseudo_queries,
)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Build prefix_fn closure for a steered condition

def make_steered_prefix_fn(
    text_enc: TextEncoder,
    projector: IntentProjector,
    user_queries: dict[str, list[str]],
    user_memories: dict[str, list[dict]],
    device: str,
    seed: int = 42,
) -> callable:
    """Return a prefix_fn(user_id, seq_np) -> torch.Tensor | None.

    Uses the first available query for each user (deterministic).
    Routes to top-1 memory by cosine similarity with h_query.
    Falls back to None (vanilla) if user has no query or no memory.
    """
    rng = random.Random(seed)

    @torch.no_grad()
    def prefix_fn(user_id: int, seq_np: np.ndarray) -> "torch.Tensor | None":
        uid_s = str(user_id)
        queries = user_queries.get(uid_s)
        mems = user_memories.get(uid_s)
        if not queries or not mems:
            return None

        q_text = queries[0]  # first query; deterministic across conditions
        h_q = text_enc.encode([q_text], is_query=True)  # [1, d_text]

        if len(mems) == 1:
            h_m = text_enc.encode([mems[0]["text"]], is_query=False)
        else:
            mem_texts = [m["text"] for m in mems]
            h_mems = text_enc.encode(mem_texts, is_query=False)  # [k, d_text]
            sims = (h_mems @ h_q.T).squeeze(-1)
            best_idx = sims.argmax().item()
            h_m = h_mems[best_idx : best_idx + 1]  # [1, d_text]

        pfx = projector(h_q, h_m)  # [1, 1, d_sasrec]
        return pfx.to(device)

    return prefix_fn


# ---------------------------------------------------------------------------
# Load a stage2 checkpoint and return (text_encoder, projector)

def load_stage2_ckpt(
    ckpt_path: str,
    device: str,
    is_stage1enc: bool,
) -> tuple[TextEncoder, IntentProjector]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    d_text = cfg.get("d_text", 768)
    d_sasrec = cfg.get("d_sasrec", 256)

    text_enc = TextEncoder(model_id="BAAI/bge-base-en-v1.5", device=device)
    if is_stage1enc and cfg.get("stage1_ckpt"):
        # Try to load encoder from stage1_ckpt path stored in config
        try:
            load_stage1_weights(cfg["stage1_ckpt"], text_enc)
        except Exception:
            pass
    # Load the fine-tuned encoder state saved in stage2 checkpoint
    text_enc_state = ckpt.get("text_encoder_state")
    if text_enc_state:
        text_enc.load_state_dict(text_enc_state)
        print(f"  [encoder] loaded from {ckpt_path}")
    else:
        print(f"  [encoder] WARNING: no text_encoder_state in {ckpt_path}")

    projector = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
    proj_state = ckpt.get("projector_state")
    if proj_state:
        projector.load_state_dict(proj_state)
        print(f"  [projector] loaded from {ckpt_path}")
    else:
        print(f"  [projector] WARNING: no projector_state in {ckpt_path}")

    text_enc.eval()
    projector.eval()
    return text_enc, projector


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="STEP2: first real steered Recall measurement")
    parser.add_argument("--category", default="Books")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--sasrec_ckpt", default="checkpoints/Books/sasrec_pretrain.pt",
                        help="Frozen SASRec backbone (F6a)")
    parser.add_argument("--ckpt_2a", default="checkpoints/Books/stage2_stage1enc_best.pt",
                        help="Stage2a checkpoint (stage1 encoder)")
    parser.add_argument("--ckpt_2b", default="checkpoints/Books/stage2_rawbge_best.pt",
                        help="Stage2b checkpoint (raw-BGE baseline)")
    parser.add_argument("--bank_jsonl", default="data/memory/Books/f3_bank.jsonl")
    parser.add_argument("--pairs_jsonl", default=None,
                        help="align_pairs.jsonl for query texts; defaults to data/processed/{category}/align_pairs.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--max_users", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    pairs_jsonl = args.pairs_jsonl or f"data/processed/{args.category}/align_pairs.jsonl"

    # ── Load SASRec backbone ─────────────────────────────────────────────────
    print(f"\n[model] Loading SASRec from {args.sasrec_ckpt} ...")
    ckpt_sas = torch.load(args.sasrec_ckpt, map_location="cpu")
    saved = ckpt_sas["args"]
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)
    model_args = SimpleNamespace(
        maxlen=saved.maxlen, hidden_units=saved.hidden_units,
        num_blocks=saved.num_blocks, num_heads=saved.num_heads,
        dropout_rate=saved.dropout_rate, norm_first=saved.norm_first,
        device=device,
    )
    dataset = load_data(args.category, args.data_dir)
    model = SASRec(dataset["usernum"], dataset["itemnum"], model_args).to(device)
    model.load_state_dict(ckpt_sas["model_state_dict"])
    model.eval()
    print(f"  users={dataset['usernum']}  items={dataset['itemnum']}  maxlen={saved.maxlen}")

    eval_args = SimpleNamespace(maxlen=saved.maxlen, device=device)

    # ── Load bank + queries ──────────────────────────────────────────────────
    print(f"\n[data] Loading bank from {args.bank_jsonl} ...")
    user_memories = _load_bank_full(args.bank_jsonl)
    mid_to_uid = {m["mid"]: uid for uid, mems in user_memories.items() for m in mems}
    print(f"  {len(user_memories)} users in bank")

    print(f"[data] Loading queries from {pairs_jsonl} ...")
    user_queries = _load_pseudo_queries(pairs_jsonl, mid_to_uid=mid_to_uid)
    print(f"  {len(user_queries)} users with queries")
    n_with_both = sum(1 for u in user_queries if u in user_memories)
    print(f"  {n_with_both} users have both query + memory (steerable)")

    # ── Vanilla condition ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Condition: VANILLA (no prefix)")
    print('='*60)
    metrics_vanilla, ranks_vanilla = evaluate_full_with_ranks(
        model, dataset, eval_args,
        split=args.split, max_users=args.max_users,
        prefix_fn=None,
    )
    print_metrics(metrics_vanilla, prefix="vanilla")

    results = {"vanilla": metrics_vanilla}

    # ── Steered conditions ───────────────────────────────────────────────────
    for tag, ckpt_path, is_stage1enc in [
        ("steered-2a", args.ckpt_2a, True),
        ("steered-2b", args.ckpt_2b, False),
    ]:
        if not Path(ckpt_path).exists():
            print(f"\n[skip] {tag}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Condition: {tag.upper()}  ({ckpt_path})")
        print('='*60)

        text_enc, projector = load_stage2_ckpt(ckpt_path, device, is_stage1enc)

        prefix_fn = make_steered_prefix_fn(
            text_enc, projector, user_queries, user_memories, device, seed=args.seed
        )

        metrics, ranks_steered = evaluate_full_with_ranks(
            model, dataset, eval_args,
            split=args.split, max_users=args.max_users,
            prefix_fn=prefix_fn,
        )
        print_metrics(metrics, prefix=tag)
        results[tag] = metrics

        # Delta vs vanilla
        print(f"\n  Delta vs vanilla:")
        for k in [5, 10, 20]:
            dr = metrics.get(f"Recall@{k}", 0) - metrics_vanilla.get(f"Recall@{k}", 0)
            dn = metrics.get(f"NDCG@{k}", 0) - metrics_vanilla.get(f"NDCG@{k}", 0)
            sign_r = "+" if dr >= 0 else ""
            sign_n = "+" if dn >= 0 else ""
            print(f"    @{k:2d}: Recall={sign_r}{dr:.4f}  NDCG={sign_n}{dn:.4f}")

        # Per-user rank delta distribution
        common_users = sorted(set(ranks_vanilla) & set(ranks_steered))
        if common_users:
            deltas = [ranks_vanilla[u] - ranks_steered[u] for u in common_users]
            # positive delta = rank improved (lower rank = better)
            improve = sum(1 for d in deltas if d > 0)
            degrade = sum(1 for d in deltas if d < 0)
            same    = sum(1 for d in deltas if d == 0)
            mean_delta = sum(deltas) / len(deltas)
            std_delta  = (sum((d - mean_delta)**2 for d in deltas) / len(deltas)) ** 0.5
            se = std_delta / (len(deltas) ** 0.5)
            se_ratio = mean_delta / (se + 1e-9)
            c_pass = "PASS ✓" if se_ratio >= 2.0 and mean_delta > 0 else "FAIL ✗"
            print(f"\n  Per-user rank delta (vanilla_rank - steered_rank, positive=better):")
            print(f"    n={len(common_users)}  improve={improve}  degrade={degrade}  same={same}")
            print(f"    mean_delta={mean_delta:+.2f}  std={std_delta:.2f}  SE={se:.2f}  SE_ratio={se_ratio:+.2f}  [C criterion: {c_pass}]")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — split={args.split}  (first prefix-injected Recall)")
    print('='*60)
    header = f"{'condition':<15} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'N@10':>7}"
    print(header)
    print("-" * len(header))
    for cond, m in results.items():
        print(f"  {cond:<13} "
              f"{m.get('Recall@5', 0):.4f}  "
              f"{m.get('Recall@10', 0):.4f}  "
              f"{m.get('Recall@20', 0):.4f}  "
              f"{m.get('NDCG@10', 0):.4f}")

    # Save to results/
    out_dir = Path("results/eval_steered")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.category}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump({"split": args.split, "category": args.category, "results": results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
