"""STEP 2-D: Author-mentioning vs non-mentioning query NDCG difference.

Runs 2b+eval-query condition, stratifies by author-mention heuristic.
"""
import argparse, json, math, re, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data
from src.eval.full_ranking import _KS
from src.training.train_hybrid import TextEncoder, _load_bank_full
from scripts.eval_steered import make_steered_prefix_fn, _load_eval_queries

AUTHOR_RE = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')


def has_author_mention(text: str) -> bool:
    return bool(AUTHOR_RE.search(text))


def load_stage2_enc_proj(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    d_text = cfg.get("d_text", 768)
    d_sasrec = cfg.get("d_sasrec", 256)

    text_enc = TextEncoder(model_id="BAAI/bge-base-en-v1.5", device=device)
    text_enc_state = ckpt.get("text_encoder_state")
    if text_enc_state:
        text_enc.load_state_dict(text_enc_state)
        print(f"  [encoder] loaded from {ckpt_path}")

    projector = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
    proj_state = ckpt.get("projector_state")
    if proj_state:
        projector.load_state_dict(proj_state)
        print(f"  [projector] loaded from {ckpt_path}")

    text_enc.eval()
    projector.eval()
    return text_enc, projector


@torch.no_grad()
def eval_stratified(model, dataset, eval_args, prefix_fn, user_flag):
    dev = eval_args.device
    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test  = dataset["user_test"]
    usernum    = dataset["usernum"]
    itemnum    = dataset["itemnum"]

    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)

    accum = {flag: {f"{m}@{k}": 0.0 for m in ["Recall", "NDCG"] for k in _KS}
             for flag in (True, False)}
    counts = {True: 0, False: 0}

    all_users = [u for u in range(1, usernum + 1)
                 if user_train.get(u) and user_test.get(u)]

    for u in all_users:
        target = user_test[u][0]
        seq = np.zeros([eval_args.maxlen], dtype=np.int32)
        idx = eval_args.maxlen - 1
        if user_valid.get(u):
            for vi in reversed(user_valid[u]):
                seq[idx] = vi; idx -= 1
                if idx == -1: break
        for i in reversed(user_train[u]):
            if idx == -1: break
            seq[idx] = i; idx -= 1

        seq_np = seq[np.newaxis, :]
        pfx = prefix_fn(u, seq_np)
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat = log_feats[0, -1, :]
        scores = all_item_embs.matmul(final_feat)

        seen = set(user_train[u]) | set(user_valid.get(u, []))
        seen.discard(0)
        seen_idx = torch.tensor([i - 1 for i in seen if 1 <= i <= itemnum],
                                dtype=torch.long, device=dev)
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        rank = int((scores > scores[target - 1]).sum().item())
        flag = user_flag.get(u, False)
        counts[flag] += 1
        for k in _KS:
            if rank < k:
                accum[flag][f"Recall@{k}"] += 1.0
                accum[flag][f"NDCG@{k}"]   += 1.0 / math.log2(rank + 2)

    def norm(a, n):
        return {k: round(v / n, 4) for k, v in a.items()} if n else {}

    return (norm(accum[True], counts[True]), counts[True],
            norm(accum[False], counts[False]), counts[False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Books")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--ckpt_2b", required=True)
    ap.add_argument("--bank_jsonl", required=True)
    ap.add_argument("--eval_queries_jsonl", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load eval queries + author flag
    user_queries_eval, _ = _load_eval_queries(args.eval_queries_jsonl)
    user_flag = {}
    for uid_s, queries in user_queries_eval.items():
        qt = queries[0] if queries else ""
        user_flag[int(uid_s)] = has_author_mention(qt)

    n_auth = sum(user_flag.values())
    n_total = len(user_flag)
    print(f"Author-mentioning: {n_auth}/{n_total} ({100*n_auth/n_total:.1f}%)")

    # Load SASRec from checkpoint (args embedded in ckpt)
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
    eval_args = SimpleNamespace(maxlen=saved.maxlen, device=device)

    dataset = load_data(args.category, args.data_dir)
    model = SASRec(dataset["usernum"], dataset["itemnum"], model_args).to(device)
    model.load_state_dict(ckpt_sas["model_state_dict"])
    model.eval()
    print(f"SASRec loaded: users={dataset['usernum']}  items={dataset['itemnum']}")

    # Load encoder + projector
    text_enc, projector = load_stage2_enc_proj(args.ckpt_2b, device)

    # Load bank
    user_memories = _load_bank_full(args.bank_jsonl)

    # Build prefix_fn using eval queries
    prefix_fn = make_steered_prefix_fn(
        text_enc, projector, user_queries_eval, user_memories, device
    )

    print("\nRunning stratified eval (2b + eval queries)...")
    auth_m, n_a, nonauth_m, n_na = eval_stratified(
        model, dataset, eval_args, prefix_fn, user_flag
    )

    print(f"\n{'Metric':<12} {'Author(n='+str(n_a)+')':>14} {'Non-author(n='+str(n_na)+')':>18} {'Diff':>8}")
    print("-" * 58)
    for k in _KS:
        for m in ["Recall", "NDCG"]:
            key = f"{m}@{k}"
            a = auth_m.get(key, 0)
            na = nonauth_m.get(key, 0)
            diff = a - na
            print(f"{key:<12} {a:>14.4f} {na:>18.4f} {diff:>+8.4f}")


if __name__ == "__main__":
    main()
