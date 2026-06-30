"""STEP 3-2: Steered Recall + recovery@N evaluation using STEP3-1 checkpoints.

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
import math
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
    routing_log: dict | None = None,
) -> callable:
    """Return a prefix_fn(user_id, seq_np) -> torch.Tensor | None.

    Uses the first available query for each user (deterministic).
    Routes to top-1 memory by cosine similarity with h_query.
    Falls back to None (vanilla) if user has no query or no memory.

    If routing_log dict is provided, records {user_id: routed_mid} for recovery@N.
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
            best_idx = 0
        else:
            mem_texts = [m["text"] for m in mems]
            h_mems = text_enc.encode(mem_texts, is_query=False)  # [k, d_text]
            sims = (h_mems @ h_q.T).squeeze(-1)
            best_idx = sims.argmax().item()
            h_m = h_mems[best_idx : best_idx + 1]  # [1, d_text]

        if routing_log is not None:
            routing_log[user_id] = mems[best_idx]["mid"]

        pfx = projector(h_q, h_m)  # [1, 1, d_sasrec]
        return pfx.to(device)

    return prefix_fn


def _load_eval_queries(
    eval_queries_jsonl: str,
) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    """Load pseudo_queries_eval.jsonl → ({uid_str: [query_text]}, {uid_str: [target_item]}).

    Eval format (from generate_pseudo_queries.py --mode eval):
      {user_id, target_item, query_text, failed_reason, ...}
    Only records with non-null query_text are loaded.
    """
    user_queries: dict[str, list[str]] = {}
    user_source_ids: dict[str, list[int]] = {}
    with open(eval_queries_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))
            q = rec.get("query_text") or ""
            x = rec.get("target_item")
            if uid and q and x is not None:
                user_queries.setdefault(uid, []).append(q)
                user_source_ids.setdefault(uid, []).append(int(x))
    return user_queries, user_source_ids


def _load_user_first_pos_mid(pairs_jsonl: str, mid_to_uid: dict) -> dict[str, str]:
    """Load {uid_str: positive_memory_id} for each user's *first* query.

    'First' matches queries[0] used in make_steered_prefix_fn — deterministic.
    A routing is 'correct' iff routed_mid == this specific mid (not any of the user's mids).
    """
    user_first_mid: dict[str, str] = {}
    with open(pairs_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("user_id", ""))
            if not uid and mid_to_uid:
                uid = mid_to_uid.get(rec.get("positive_memory_id", ""), "")
            mid = str(rec.get("positive_memory_id", ""))
            if uid and mid and uid not in user_first_mid:
                user_first_mid[uid] = mid
    return user_first_mid


def compute_recovery_at_n(
    user_ranks: dict[int, int],
    routing_log: dict[int, str],
    user_first_mid: dict[str, str],
    ks: list[int],
    label: str,
) -> None:
    """Print recovery@N breakdown: correct vs wrong routing groups.

    correct routing: routed_mid == user's first-query positive_memory_id exactly.
    wrong routing: a different memory was selected (even from the same user).
    """
    correct_users = []
    wrong_users = []
    for uid, routed_mid in routing_log.items():
        uid_s = str(uid)
        correct_mid = user_first_mid.get(uid_s, "")
        if uid not in user_ranks:
            continue
        if correct_mid and routed_mid == correct_mid:
            correct_users.append(uid)
        else:
            wrong_users.append(uid)

    n_correct = len(correct_users)
    n_wrong = len(wrong_users)
    routing_acc = n_correct / (n_correct + n_wrong) if (n_correct + n_wrong) > 0 else 0.0
    print(f"\n  Recovery@N [{label}]:")
    print(f"    routing accuracy = {routing_acc:.4f}  ({n_correct} correct / {n_correct+n_wrong} routed)")

    for k in ks:
        r_correct = (sum(1 for u in correct_users if user_ranks[u] < k) / n_correct
                     if n_correct > 0 else float("nan"))
        r_wrong = (sum(1 for u in wrong_users if user_ranks[u] < k) / n_wrong
                   if n_wrong > 0 else float("nan"))
        gap_str = f"{r_correct - r_wrong:+.4f}" if (n_correct > 0 and n_wrong > 0) else "n/a"
        print(f"    Recall@{k:2d}:  correct={r_correct:.4f} (n={n_correct})  "
              f"wrong={r_wrong:.4f} (n={n_wrong})  gap={gap_str}")


@torch.no_grad()
def evaluate_x_target(
    model,
    dataset: dict,
    args,
    prefix_fn,
    user_source_ids: dict[str, list[int]],
    max_users: int = 10_000,
) -> tuple[dict[str, float], dict[int, int]]:
    """Evaluate steered ranking where the target is source_item_id (X).

    X = the item the query was written about — guaranteed in positive memory cluster.
    This is Headline-1 (causal-closed eval): prefix should push representation toward X.

    Masking: seen items excluding X itself (so X is always rankable).
    Only evaluates users that have a query with a valid source_item_id.
    """
    model.eval()
    dev = args.device
    itemnum = dataset["itemnum"]
    user_train = dataset["user_train"]

    eligible = [
        u for u in range(1, dataset["usernum"] + 1)
        if user_train.get(u) and user_source_ids.get(str(u))
    ]
    if len(eligible) > max_users:
        rng = random.Random(42)
        eligible = rng.sample(eligible, max_users)

    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)

    ks = [10, 20, 50, 100]
    accum = {f"Recall@{k}": 0.0 for k in ks}
    accum.update({f"NDCG@{k}": 0.0 for k in ks})
    n_valid = 0
    user_ranks: dict[int, int] = {}

    for u in eligible:
        src = user_source_ids.get(str(u), [])
        x = src[0] if src else None
        if x is None or not (1 <= x <= itemnum):
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(user_train[u]):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1

        seq_np = seq[np.newaxis, :]
        pfx = prefix_fn(u, seq_np) if prefix_fn is not None else None
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat = log_feats[0, -1, :]
        scores = all_item_embs.matmul(final_feat)

        seen = set(user_train[u])
        seen.discard(0)
        seen.discard(x)  # keep X rankable even if it's in training history
        seen_idx = torch.tensor([i - 1 for i in seen if 1 <= i <= itemnum],
                                dtype=torch.long, device=dev)
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        rank_of_x = (scores > scores[x - 1]).sum().item()
        user_ranks[u] = rank_of_x

        n_valid += 1
        for k in ks:
            if rank_of_x < k:
                accum[f"Recall@{k}"] += 1.0
                accum[f"NDCG@{k}"] += 1.0 / math.log2(rank_of_x + 2)

    if n_valid == 0:
        return {k: 0.0 for k in accum}, {}

    metrics = {k: round(v / n_valid, 6) for k, v in accum.items()}
    return metrics, user_ranks


# ---------------------------------------------------------------------------
# Rank bucket display helper (STEP 1c)

_RANK_BUCKETS = [(0, 9, "0-9"), (10, 49, "10-49"), (50, 99, "50-99"),
                 (100, 199, "100-199"), (200, int(1e9), "200+")]


def print_rank_buckets(ranks: "dict[int, int]", label: str) -> None:
    total = len(ranks)
    if total == 0:
        return
    print(f"\n  Rank buckets [{label}] (n={total}):")
    for lo, hi, name in _RANK_BUCKETS:
        cnt = sum(1 for r in ranks.values() if lo <= r <= hi)
        print(f"    {name:>7}: {cnt:5d}  ({100*cnt/total:5.1f}%)")


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
    parser.add_argument("--eval_queries_jsonl", default=None,
                        help="pseudo_queries_eval.jsonl for 3-layer comparison (condition iii)")
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
    user_queries, user_source_ids = _load_pseudo_queries(pairs_jsonl, mid_to_uid=mid_to_uid)
    print(f"  {len(user_queries)} users with queries")
    n_with_both = sum(1 for u in user_queries if u in user_memories)
    print(f"  {n_with_both} users have both query + memory (steerable)")

    user_first_mid = _load_user_first_pos_mid(pairs_jsonl, mid_to_uid)
    print(f"  {len(user_first_mid)} users with first-query positive_memory_id (for recovery@N)")

    # ── Load eval queries (condition iii) if provided ────────────────────────
    user_queries_eval: dict[str, list[str]] = {}
    user_source_ids_eval: dict[str, list[int]] = {}
    if args.eval_queries_jsonl:
        print(f"[data] Loading eval queries from {args.eval_queries_jsonl} ...")
        user_queries_eval, user_source_ids_eval = _load_eval_queries(args.eval_queries_jsonl)
        n_eval_both = sum(1 for u in user_queries_eval if u in user_memories)
        print(f"  {len(user_queries_eval)} users with eval queries")
        print(f"  {n_eval_both} users have eval query + memory (steerable, condition iii)")

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
    # Each checkpoint runs up to 2 query sets:
    #   (ii) train queries  — past intent, misaligned lower bound
    #   (iii) eval queries  — target intent, aligned upper bound (if --eval_queries_jsonl given)
    for ckpt_base_tag, ckpt_path, is_stage1enc in [
        ("steered-2a", args.ckpt_2a, True),
        ("steered-2b", args.ckpt_2b, False),
    ]:
        if not Path(ckpt_path).exists():
            print(f"\n[skip] {ckpt_base_tag}: {ckpt_path} not found")
            continue

        text_enc, projector = load_stage2_ckpt(ckpt_path, device, is_stage1enc)

        # Build query set list: always run train; add eval if provided
        query_sets = [("train", user_queries, user_source_ids)]
        if user_queries_eval:
            query_sets.append(("eval", user_queries_eval, user_source_ids_eval))

        for qlabel, qs, src_ids in query_sets:
            tag = f"{ckpt_base_tag}-{qlabel}"

            print(f"\n{'='*60}")
            print(f"Condition: {tag.upper()}  ({ckpt_path})  query={qlabel}")
            print('='*60)

            routing_log: dict[int, str] = {}
            prefix_fn = make_steered_prefix_fn(
                text_enc, projector, qs, user_memories, device, seed=args.seed,
                routing_log=routing_log,
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

            # Recovery@N: only for train queries (positive_memory_id defined)
            if qlabel == "train" and routing_log and user_first_mid:
                compute_recovery_at_n(ranks_steered, routing_log, user_first_mid, [5, 10, 20], tag)

            # ── X/W-target eval ──
            # train query → target = source item X (causal-closed)
            # eval query  → target = W (intent-aligned, circularity-robust)
            xtgt_suffix = "Xsrc" if qlabel == "train" else "Wtgt"
            x_label = "source_item_id X (causal-closed)" if qlabel == "train" else "target_item W (intent-aligned)"
            print(f"\n  [{xtgt_suffix} eval] target = {x_label}")
            x_metrics_steered, x_ranks_steered = evaluate_x_target(
                model, dataset, eval_args, prefix_fn, src_ids, max_users=args.max_users,
            )
            x_metrics_vanilla, x_ranks_vanilla = evaluate_x_target(
                model, dataset, eval_args, None, src_ids, max_users=args.max_users,
            )
            for k in [10, 20, 50, 100]:
                rs = x_metrics_steered.get(f"Recall@{k}", 0)
                rv = x_metrics_vanilla.get(f"Recall@{k}", 0)
                ns = x_metrics_steered.get(f"NDCG@{k}", 0)
                nv = x_metrics_vanilla.get(f"NDCG@{k}", 0)
                dr = rs - rv
                sign = "+" if dr >= 0 else ""
                print(f"    @{k:3d}: R={sign}{dr:.4f}(s={rs:.4f}/v={rv:.4f})  "
                      f"N={ns:.4f}(v={nv:.4f})")
            # SE_ratio (rank delta)
            common_x = sorted(set(x_ranks_vanilla) & set(x_ranks_steered))
            if common_x:
                x_deltas = [x_ranks_vanilla[u] - x_ranks_steered[u] for u in common_x]
                imp = sum(1 for d in x_deltas if d > 0)
                deg = sum(1 for d in x_deltas if d < 0)
                mean_xd = sum(x_deltas) / len(x_deltas)
                std_xd = (sum((d - mean_xd)**2 for d in x_deltas) / len(x_deltas)) ** 0.5
                se_x = std_xd / (len(x_deltas) ** 0.5)
                ser_x = mean_xd / (se_x + 1e-9)
                c_pass = "PASS ✓" if ser_x >= 2.0 and mean_xd > 0 else "FAIL ✗"
                print(f"    n={len(common_x)}  improve={imp}  degrade={deg}")
                print(f"    mean_delta={mean_xd:+.2f}  SE={se_x:.2f}  SE_ratio={ser_x:+.2f}  [C: {c_pass}]")
            print_rank_buckets(x_ranks_steered, f"{tag}-{xtgt_suffix} steered")
            results[f"{tag}-{xtgt_suffix}"] = x_metrics_steered

            # Per-user rank delta distribution (W-target LOO, secondary reference)
            print(f"\n  Per-user rank delta W-target (LOO, secondary reference):")
            common_users = sorted(set(ranks_vanilla) & set(ranks_steered))
            if common_users:
                deltas = [ranks_vanilla[u] - ranks_steered[u] for u in common_users]
                improve = sum(1 for d in deltas if d > 0)
                degrade = sum(1 for d in deltas if d < 0)
                same    = sum(1 for d in deltas if d == 0)
                mean_delta = sum(deltas) / len(deltas)
                std_delta  = (sum((d - mean_delta)**2 for d in deltas) / len(deltas)) ** 0.5
                se = std_delta / (len(deltas) ** 0.5)
                se_ratio = mean_delta / (se + 1e-9)
                print(f"    n={len(common_users)}  improve={improve}  degrade={degrade}  same={same}")
                print(f"    mean_delta={mean_delta:+.2f}  std={std_delta:.2f}  SE={se:.2f}  SE_ratio={se_ratio:+.2f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — split={args.split}  (first prefix-injected Recall)")
    print('='*60)
    header = f"{'condition':<25} {'R@10':>7} {'R@20':>7} {'R@50':>7} {'R@100':>8} {'N@10':>7} {'N@20':>7}"
    print(header)
    print("-" * len(header))
    for cond, m in results.items():
        print(f"  {cond:<23} "
              f"{m.get('Recall@10', 0):.4f}  "
              f"{m.get('Recall@20', 0):.4f}  "
              f"{m.get('Recall@50', 0):.4f}  "
              f"{m.get('Recall@100', 0):.5f}  "
              f"{m.get('NDCG@10', 0):.4f}  "
              f"{m.get('NDCG@20', 0):.4f}")

    # Save to results/
    out_dir = Path("results/eval_steered")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.category}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump({"split": args.split, "category": args.category, "results": results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
