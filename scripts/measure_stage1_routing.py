"""Stage1 routing accuracy measurement: frozen-bge vs Stage1 checkpoint.

In-distribution  : pseudo-queries from align_pairs.jsonl (train distribution).
OOD              : memory source_text as queries (raw review text, different style
                   from the P2 pseudo-queries the encoder was trained on).

For each query the GT is the provenance positive_memory_id.
Routing: encode query → cosine sim against all memories for that user → top-1.

Usage:
    python3 scripts/measure_stage1_routing.py \
        --ckpt checkpoints/Books/stage1_align_best.pt \
        --pairs data/processed/Books/align_pairs.jsonl \
        --bank_dir data/processed/Books/memory_bank \
        --device cuda:0 \
        --n_indist 1000 \
        --n_ood 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train_align import TextEncoderWithHead, load_memory_index, compute_routing_accuracy

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@torch.no_grad()
def compute_routing_ood(
    model: TextEncoderWithHead,
    ood_pairs: list[dict],  # [{query: source_text, positive_memory_id, ...}]
    mem_to_vec: dict,
    mem_to_user: dict,
    user_to_mems: dict,
    use_frozen_bge: bool = False,
    batch_size: int = 32,
) -> float:
    """Same as compute_routing_accuracy but accepts pre-built dicts directly."""
    return compute_routing_accuracy(
        model, ood_pairs, mem_to_vec, mem_to_user, user_to_mems,
        use_frozen_bge=use_frozen_bge, batch_size=batch_size,
    )


def build_ood_pairs(
    mem_to_vec: dict,
    mem_to_user: dict,
    user_to_mems: dict,
    bank_dir: str,
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Sample n OOD pairs: query = source_text of a memory, GT = that memory_id.

    source_text is raw review text (different style from P2 pseudo-queries).
    """
    rng = random.Random(seed)
    all_mids = list(mem_to_vec.keys())
    sampled = rng.sample(all_mids, min(n, len(all_mids)))

    # Need source_text: read from bank files. Build mid->source_text cache.
    mid_to_source: dict[str, str] = {}
    bank_path = Path(bank_dir)
    needed = set(sampled)
    for fpath in bank_path.glob("*.json"):
        if not needed:
            break
        if fpath.name.startswith("_"):
            continue
        with open(fpath) as f:
            ub = json.load(f)
        if not isinstance(ub, dict):
            continue
        for unit in ub.get("units", []):
            mid = unit["memory_id"]
            if mid in needed:
                mid_to_source[mid] = unit["embedding"]["source_text"]
                needed.discard(mid)

    pairs = []
    for mid in sampled:
        if mid not in mid_to_source:
            continue
        pairs.append({
            "query": mid_to_source[mid],
            "positive_memory_id": mid,
        })
    return pairs


def load_stage1_model(ckpt_path: str, device: str) -> TextEncoderWithHead:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = TextEncoderWithHead(
        model_id=cfg.get("encoder", "BAAI/bge-base-en-v1.5"),
        proj_hidden=cfg.get("proj_hidden", 768),
        proj_out=cfg.get("proj_out", 768),
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[ckpt] loaded from {ckpt_path}  (epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 'n/a'):.4f})")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/Books/stage1_align_best.pt")
    p.add_argument("--pairs", default="data/processed/Books/align_pairs.jsonl")
    p.add_argument("--bank_dir", default="data/processed/Books/memory_bank")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_indist", type=int, default=1000, help="in-distribution eval samples")
    p.add_argument("--n_ood", type=int, default=500, help="OOD eval samples (source_text queries)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    # Load memory index
    mem_to_vec, mem_to_user, user_to_mems = load_memory_index(args.bank_dir)

    # Load Stage1 model
    model = load_stage1_model(args.ckpt, args.device)

    # ── In-distribution pairs (pseudo-queries, same style as training) ──
    all_pairs = []
    with open(args.pairs) as f:
        for line in f:
            all_pairs.append(json.loads(line))

    rng = random.Random(args.seed)
    indist_pairs = rng.sample(all_pairs, min(args.n_indist, len(all_pairs)))

    print(f"\n{'='*60}")
    print(f"IN-DISTRIBUTION (pseudo-query, n={len(indist_pairs)})")
    print(f"{'='*60}")

    frozen_indist = compute_routing_accuracy(
        model, indist_pairs, mem_to_vec, mem_to_user, user_to_mems,
        use_frozen_bge=True,
    )
    stage1_indist = compute_routing_accuracy(
        model, indist_pairs, mem_to_vec, mem_to_user, user_to_mems,
        use_frozen_bge=False,
    )
    delta_indist = stage1_indist - frozen_indist
    print(f"  frozen-bge : {frozen_indist:.4f}")
    print(f"  Stage1     : {stage1_indist:.4f}  (Δ={delta_indist:+.4f})")

    # ── OOD pairs (source_text = raw review text, different style) ──
    print(f"\n{'='*60}")
    print(f"OOD (source_text / raw review, n={args.n_ood})")
    print(f"  source_text is the original review bge was trained on —")
    print(f"  different style from P2 pseudo-queries (train↔real gap test)")
    print(f"{'='*60}")

    ood_pairs = build_ood_pairs(
        mem_to_vec, mem_to_user, user_to_mems,
        args.bank_dir, n=args.n_ood, seed=args.seed,
    )
    print(f"  built {len(ood_pairs)} OOD pairs")

    frozen_ood = compute_routing_accuracy(
        model, ood_pairs, mem_to_vec, mem_to_user, user_to_mems,
        use_frozen_bge=True,
    )
    stage1_ood = compute_routing_accuracy(
        model, ood_pairs, mem_to_vec, mem_to_user, user_to_mems,
        use_frozen_bge=False,
    )
    delta_ood = stage1_ood - frozen_ood
    print(f"  frozen-bge : {frozen_ood:.4f}")
    print(f"  Stage1     : {stage1_ood:.4f}  (Δ={delta_ood:+.4f})")

    # ── Correct-vs-wrong memory sim gap (contribution B signal) ──
    print(f"\n{'='*60}")
    print(f"CORRECT vs WRONG MEMORY SIM GAP (in-dist, n={len(indist_pairs)})")
    print(f"{'='*60}")

    for label, use_frozen in [("frozen-bge", True), ("Stage1    ", False)]:
        gaps = []
        model.eval()
        with torch.no_grad():
            for rec in indist_pairs[:200]:
                q = rec["query"]
                pos_id = rec["positive_memory_id"]
                uid = mem_to_user.get(pos_id)
                if uid is None:
                    continue
                cand_ids = user_to_mems[uid]
                if len(cand_ids) < 2:
                    continue

                if use_frozen:
                    hq = model.encode_queries_frozen_bge([q])[0]
                else:
                    hq = model.encode_queries([q])[0]

                cand_vecs = torch.tensor(
                    np.stack([mem_to_vec[mid] for mid in cand_ids]),
                    device=hq.device, dtype=torch.float32,
                )
                sims = (cand_vecs @ hq).cpu().tolist()
                pos_idx = cand_ids.index(pos_id)
                pos_sim = sims[pos_idx]
                neg_sims = [s for i, s in enumerate(sims) if i != pos_idx]
                if neg_sims:
                    gaps.append(pos_sim - max(neg_sims))

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"  {label}: avg(sim_correct - sim_best_wrong) = {avg_gap:+.4f}  (n={len(gaps)})")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  In-dist   frozen={frozen_indist:.4f}  Stage1={stage1_indist:.4f}  Δ={delta_indist:+.4f}")
    print(f"  OOD       frozen={frozen_ood:.4f}  Stage1={stage1_ood:.4f}  Δ={delta_ood:+.4f}")
    if delta_ood > 0:
        print(f"  PASS: Stage1 > frozen-bge on OOD → train↔real gap closed (v0.4.13 contribution B)")
    else:
        print(f"  WARN: Stage1 ≤ frozen-bge on OOD — gap not closed. Check epoch count or lr.")


if __name__ == "__main__":
    main()
