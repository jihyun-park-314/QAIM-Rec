"""Recovery@N analysis — circularity-robust headline metric.

Three prefix conditions per test user (same query text for all):
  correct  — provenance-positive memory (correct memory for this user)
  wrong    — first hard-negative memory (wrong memory from a different user)
  vanilla  — no prefix (SASRec baseline)

Recovery@N(cond) = fraction of users where:
    target ∈ TopN(cond)  AND  target ∉ TopN(vanilla)
Gap = Recovery@N(correct) − Recovery@N(wrong)

Circularity note: both correct and wrong conditions use train-split queries,
so circularity is symmetric and cancels in the gap.

B-signal: Stage1 gap > raw-BGE gap → alignment training improves discrimination.

Ablation: pass --ckpt_2a_ablation to compare two 2a checkpoints
(e.g. trained with vs without L_contrastive active).

Usage:
    python scripts/recovery_analysis.py \\
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
from src.training.train_hybrid import (
    TextEncoder,
    load_stage1_weights,
    _load_bank_full,
)


# ---------------------------------------------------------------------------
# Data loading helpers

def load_mid_to_text(bank_jsonl: str) -> dict[str, str]:
    """Return {memory_id → intent_description} for all bank entries."""
    mid_to_text: dict[str, str] = {}
    with open(bank_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mid = str(rec.get("memory_id", ""))
            text = rec.get("intent_description", "")
            if mid and text:
                mid_to_text[mid] = text
    return mid_to_text


def load_user_pairs(
    pairs_jsonl: str,
    mid_to_uid: dict[str, str],
) -> dict[str, dict]:
    """Load align_pairs → per-user {query, correct_mid, wrong_mid}.

    Takes the first query per user and the first hard-negative as wrong_mid.
    Returns {uid_str: {query: str, correct_mid: str, wrong_mid: str|None}}.
    """
    # Collect all rows by uid (may have multiple queries per uid)
    per_uid: dict[str, dict] = {}
    with open(pairs_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pos_mid = rec.get("positive_memory_id", "")
            uid = mid_to_uid.get(pos_mid, "")
            if not uid:
                continue
            q = rec.get("query", "")
            if not q:
                continue
            hard_negs = rec.get("hard_negative_memory_ids", [])
            wrong_mid = hard_negs[0] if hard_negs else None
            if uid not in per_uid:
                # First query wins (deterministic, matches eval_steered.py)
                per_uid[uid] = {
                    "query": q,
                    "correct_mid": pos_mid,
                    "wrong_mid": wrong_mid,
                }
    return per_uid


# ---------------------------------------------------------------------------
# Checkpoint loading (shared with eval_steered.py)

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
        try:
            load_stage1_weights(cfg["stage1_ckpt"], text_enc)
        except Exception:
            pass
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
# Prefix factories

def make_fixed_memory_prefix_fn(
    text_enc: TextEncoder,
    projector: IntentProjector,
    user_pairs: dict[str, dict],
    mid_to_text: dict[str, str],
    device: str,
    use_correct: bool,  # True → correct memory, False → wrong memory
) -> callable:
    """Return prefix_fn(user_id, seq_np) that uses a fixed memory (no routing).

    correct=True  → provenance-positive memory
    correct=False → first hard-negative memory
    Falls back to None (vanilla) if query/memory unavailable.
    """
    @torch.no_grad()
    def prefix_fn(user_id: int, seq_np: np.ndarray) -> "torch.Tensor | None":
        uid_s = str(user_id)
        pair = user_pairs.get(uid_s)
        if pair is None:
            return None

        q_text = pair["query"]
        mid = pair["correct_mid"] if use_correct else pair.get("wrong_mid")
        if not mid:
            return None
        mem_text = mid_to_text.get(mid)
        if not mem_text:
            return None

        h_q = text_enc.encode([q_text], is_query=True)   # [1, d_text]
        h_m = text_enc.encode([mem_text], is_query=False) # [1, d_text]
        pfx = projector(h_q, h_m)                          # [1, 1, d_sasrec]
        return pfx.to(device)

    return prefix_fn


# ---------------------------------------------------------------------------
# Per-user rank collection

@torch.no_grad()
def collect_ranks(
    model,
    dataset: dict,
    args,
    split: str = "test",
    max_users: int = 10_000,
    prefix_fn=None,
) -> dict[int, int]:
    """Run full-catalog ranking for all users, return {uid: rank_of_target}."""
    model.eval()
    dev = args.device

    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test  = dataset["user_test"]
    usernum    = dataset["usernum"]
    itemnum    = dataset["itemnum"]

    if split == "test":
        target_dict    = user_test
        history_extra  = user_valid
    else:
        target_dict    = user_valid
        history_extra  = {}

    all_users = [u for u in range(1, usernum + 1)
                 if user_train.get(u) and target_dict.get(u)]
    if len(all_users) > max_users:
        import random
        rng = random.Random(42)
        all_users = rng.sample(all_users, max_users)

    all_items     = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)  # [I, d]

    user_ranks: dict[int, int] = {}

    for u in all_users:
        target_items = target_dict[u]
        if not target_items:
            continue
        target = target_items[0]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if split == "test" and history_extra.get(u):
            for extra in reversed(history_extra[u]):
                seq[idx] = extra
                idx -= 1
                if idx == -1:
                    break
        for i in reversed(user_train[u]):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1

        seq_np = seq[np.newaxis, :]
        pfx = prefix_fn(u, seq_np) if prefix_fn is not None else None
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat    = log_feats[0, -1, :]
        scores        = all_item_embs.matmul(final_feat)

        seen = set(user_train[u])
        seen.discard(0)
        if split == "test" and history_extra.get(u):
            seen.update(history_extra[u])
        seen_idx = torch.tensor(
            [i - 1 for i in seen if 1 <= i <= itemnum],
            dtype=torch.long, device=dev,
        )
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        rank = int((scores > scores[target - 1]).sum().item())
        user_ranks[u] = rank

    return user_ranks


# ---------------------------------------------------------------------------
# Recovery metrics

def compute_recovery(
    ranks_cond: dict[int, int],
    ranks_vanilla: dict[int, int],
    ks: list[int],
) -> dict[str, float]:
    """Recovery@N = fraction of shared users where cond hits and vanilla misses.

    Only counts users that have a valid cond rank (i.e. prefix was available).
    Denominator = all users in ranks_vanilla (comparable to vanilla Recall@N).
    """
    common = set(ranks_cond) & set(ranks_vanilla)
    if not common:
        return {f"Recovery@{k}": 0.0 for k in ks}

    n_denom = len(ranks_vanilla)  # total evaluable users (vanilla denominator)
    results: dict[str, float] = {}
    for k in ks:
        hits = sum(
            1 for u in common
            if ranks_cond[u] < k and ranks_vanilla[u] >= k
        )
        results[f"Recovery@{k}"] = round(hits / n_denom, 6)
        results[f"Recovery@{k}_n"] = hits
    results["n_steerable"] = len(common)
    results["n_total"]     = n_denom
    return results


def compute_recall(ranks: dict[int, int], ks: list[int]) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {f"Recall@{k}": 0.0 for k in ks}
    return {
        f"Recall@{k}": round(sum(1 for r in ranks.values() if r < k) / n, 6)
        for k in ks
    }


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Recovery@N — circularity-robust headline")
    parser.add_argument("--category",    default="Books")
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--sasrec_ckpt", default="checkpoints/Books/sasrec_pretrain.pt")
    parser.add_argument("--ckpt_2a",     default="checkpoints/Books/stage2_stage1enc_best.pt",
                        help="Stage2a: stage1 encoder + projector")
    parser.add_argument("--ckpt_2b",     default="checkpoints/Books/stage2_rawbge_best.pt",
                        help="Stage2b: raw-BGE + projector")
    parser.add_argument("--ckpt_2a_ablation", default=None,
                        help="Optional: second 2a ckpt for contrastive on/off ablation")
    parser.add_argument("--bank_jsonl",  default="data/memory/Books/f3_bank.jsonl")
    parser.add_argument("--pairs_jsonl", default=None)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--split",       default="test", choices=["test", "val"])
    parser.add_argument("--max_users",   type=int, default=10_000)
    args = parser.parse_args()

    device     = args.device
    pairs_jsonl = args.pairs_jsonl or f"data/processed/{args.category}/align_pairs.jsonl"
    KS = [10, 20]

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
    eval_args = SimpleNamespace(maxlen=saved.maxlen, device=device)
    print(f"  users={dataset['usernum']}  items={dataset['itemnum']}")

    # ── Load bank + align_pairs ──────────────────────────────────────────────
    print(f"\n[data] Loading bank from {args.bank_jsonl} ...")
    user_memories = _load_bank_full(args.bank_jsonl)
    mid_to_uid  = {m["mid"]: uid for uid, mems in user_memories.items() for m in mems}
    mid_to_text = load_mid_to_text(args.bank_jsonl)
    print(f"  {len(user_memories)} users  {len(mid_to_text)} memories")

    print(f"[data] Loading pairs from {pairs_jsonl} ...")
    user_pairs = load_user_pairs(pairs_jsonl, mid_to_uid)
    n_wrong = sum(1 for p in user_pairs.values() if p.get("wrong_mid") and mid_to_text.get(p["wrong_mid"]))
    print(f"  {len(user_pairs)} users with query+correct_mid  {n_wrong} with wrong_mid in bank")

    # ── Vanilla baseline ranks (run once, shared across all conditions) ──────
    print(f"\n{'='*60}")
    print(f"[1/4] Vanilla ranks (no prefix)  split={args.split}")
    print('='*60)
    ranks_vanilla = collect_ranks(
        model, dataset, eval_args,
        split=args.split, max_users=args.max_users, prefix_fn=None,
    )
    recall_vanilla = compute_recall(ranks_vanilla, KS)
    for k in KS:
        print(f"  Vanilla Recall@{k} = {recall_vanilla[f'Recall@{k}']:.4f}  (n={len(ranks_vanilla)})")

    summary_rows: list[dict] = []

    # ── Helper: run one checkpoint condition ─────────────────────────────────
    def run_ckpt(tag: str, ckpt_path: str, is_stage1enc: bool) -> dict | None:
        if not Path(ckpt_path).exists():
            print(f"\n[skip] {tag}: {ckpt_path} not found")
            return None

        print(f"\n{'='*60}")
        print(f"[ckpt] {tag}  ({ckpt_path})")
        print('='*60)

        text_enc, projector = load_stage2_ckpt(ckpt_path, device, is_stage1enc)

        pfn_correct = make_fixed_memory_prefix_fn(
            text_enc, projector, user_pairs, mid_to_text, device, use_correct=True)
        pfn_wrong = make_fixed_memory_prefix_fn(
            text_enc, projector, user_pairs, mid_to_text, device, use_correct=False)

        print(f"  Computing correct-prefix ranks ...")
        ranks_correct = collect_ranks(
            model, dataset, eval_args,
            split=args.split, max_users=args.max_users, prefix_fn=pfn_correct,
        )
        print(f"  Computing wrong-prefix ranks ...")
        ranks_wrong = collect_ranks(
            model, dataset, eval_args,
            split=args.split, max_users=args.max_users, prefix_fn=pfn_wrong,
        )

        # Recall@N for each condition
        recall_correct = compute_recall(ranks_correct, KS)
        recall_wrong   = compute_recall(ranks_wrong,   KS)

        # Recovery@N
        r_correct = compute_recovery(ranks_correct, ranks_vanilla, KS)
        r_wrong   = compute_recovery(ranks_wrong,   ranks_vanilla, KS)

        print(f"\n  Recall comparison (n_vanilla={len(ranks_vanilla)}):")
        for k in KS:
            vc = recall_vanilla[f"Recall@{k}"]
            rc = recall_correct[f"Recall@{k}"]
            rw = recall_wrong[f"Recall@{k}"]
            print(f"  @{k:2d}  vanilla={vc:.4f}  correct={rc:.4f} (Δ{rc-vc:+.4f})  "
                  f"wrong={rw:.4f} (Δ{rw-vc:+.4f})")

        print(f"\n  Recovery@N (correct recovers, vanilla misses):")
        for k in KS:
            c = r_correct[f"Recovery@{k}"]
            w = r_wrong[f"Recovery@{k}"]
            gap = c - w
            sign = "+" if gap >= 0 else ""
            print(f"  @{k:2d}  correct={c:.4f} (n={r_correct[f'Recovery@{k}_n']:4d})  "
                  f"wrong={w:.4f} (n={r_wrong[f'Recovery@{k}_n']:4d})  "
                  f"gap={sign}{gap:.4f}")

        row = {
            "tag": tag,
            "ckpt": ckpt_path,
            "split": args.split,
        }
        for k in KS:
            row[f"recall_vanilla_@{k}"] = recall_vanilla[f"Recall@{k}"]
            row[f"recall_correct_@{k}"] = recall_correct[f"Recall@{k}"]
            row[f"recall_wrong_@{k}"]   = recall_wrong[f"Recall@{k}"]
            row[f"recovery_correct_@{k}"] = r_correct[f"Recovery@{k}"]
            row[f"recovery_wrong_@{k}"]   = r_wrong[f"Recovery@{k}"]
            row[f"recovery_gap_@{k}"]     = round(
                r_correct[f"Recovery@{k}"] - r_wrong[f"Recovery@{k}"], 6)
        row["n_steerable"] = r_correct["n_steerable"]
        row["n_total"]     = r_correct["n_total"]
        return row

    # ── 2a: stage1 encoder ───────────────────────────────────────────────────
    row_2a = run_ckpt("2a-stage1enc", args.ckpt_2a, is_stage1enc=True)
    if row_2a:
        summary_rows.append(row_2a)

    # ── 2b: raw-BGE ──────────────────────────────────────────────────────────
    row_2b = run_ckpt("2b-rawbge", args.ckpt_2b, is_stage1enc=False)
    if row_2b:
        summary_rows.append(row_2b)

    # ── Optional ablation: 2a vs 2a_ablation ─────────────────────────────────
    if args.ckpt_2a_ablation:
        row_abl = run_ckpt("2a-ablation", args.ckpt_2a_ablation, is_stage1enc=True)
        if row_abl:
            summary_rows.append(row_abl)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — Recovery@N  split={args.split}")
    print('='*60)
    print(f"  circularity note: both correct/wrong use train queries — gap is circularity-robust")
    print(f"  B-signal criterion: 2a gap > 2b gap at @10 and @20\n")

    hdr = f"  {'tag':<20}  {'R@10':>7}  {'R@20':>7}  {'Rcov@10':>9}  {'Rcov@20':>9}  {'gap@10':>8}  {'gap@20':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in summary_rows:
        print(f"  {row['tag']:<20}  "
              f"{row.get('recall_correct_@10', 0):.4f}   "
              f"{row.get('recall_correct_@20', 0):.4f}   "
              f"{row.get('recovery_correct_@10', 0):.5f}   "
              f"{row.get('recovery_correct_@20', 0):.5f}   "
              f"{row.get('recovery_gap_@10', 0):+.5f}  "
              f"{row.get('recovery_gap_@20', 0):+.5f}")

    if len(summary_rows) >= 2:
        gap_2a_10 = summary_rows[0].get("recovery_gap_@10", 0)
        gap_2b_10 = summary_rows[1].get("recovery_gap_@10", 0)
        b_signal  = gap_2a_10 > gap_2b_10
        print(f"\n  B-signal (@10): 2a_gap={gap_2a_10:+.5f}  2b_gap={gap_2b_10:+.5f}  "
              f"→ {'CONFIRMED ✓' if b_signal else 'NOT confirmed ✗'}")

    if args.ckpt_2a_ablation and len(summary_rows) >= 3:
        gap_base = summary_rows[0].get("recovery_gap_@10", 0)
        gap_abl  = summary_rows[2].get("recovery_gap_@10", 0)
        print(f"\n  L_contrastive ablation (@10):")
        print(f"    2a (base)={gap_base:+.5f}  ablation={gap_abl:+.5f}  diff={gap_abl - gap_base:+.5f}")
        if gap_abl < gap_base:
            print("    → contrastive HARMS gap (supports Claude Code hypothesis)")
        elif gap_abl > gap_base:
            print("    → contrastive HELPS gap")
        else:
            print("    → contrastive has negligible effect on gap")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir  = Path("results/recovery")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.category}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump({
            "category": args.category,
            "split":    args.split,
            "vanilla_recall": recall_vanilla,
            "results":  summary_rows,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
