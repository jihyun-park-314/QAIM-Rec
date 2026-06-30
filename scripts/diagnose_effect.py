"""효과크기 원인 진단 스크립트.

Per-user rank delta를 저장해 다음 4가지를 분석:
1. 원인1: rank delta 분포 — top-K 내/외 이동 비율
2. 원인3: 도메인 희소도 확인(시퀀스 길이별 개선/저하)
3. 원인4: K≥2 vs K=1 유저 개선 비교
4. STEP2: 2a vs 2b paired (동일 유저 대상)

추가: @50/@100 Recall로 "어디서 개선되나" 직접 확인
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data
from src.training.train_hybrid import TextEncoder, _load_bank_full
from scripts.eval_steered import make_steered_prefix_fn, _load_eval_queries

_KS_EXT = [5, 10, 20, 50, 100]


def load_stage2(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    d_text = cfg.get("d_text", 768)
    d_sasrec = cfg.get("d_sasrec", 256)
    text_enc = TextEncoder(model_id="BAAI/bge-base-en-v1.5", device=device)
    state = ckpt.get("text_encoder_state")
    if state:
        text_enc.load_state_dict(state)
    proj = IntentProjector(d_text=d_text, d_sasrec=d_sasrec).to(device)
    pstate = ckpt.get("projector_state")
    if pstate:
        proj.load_state_dict(pstate)
    text_enc.eval(); proj.eval()
    return text_enc, proj


@torch.no_grad()
def run_eval(model, dataset, eval_args, prefix_fn):
    """Returns {user_id: rank}."""
    dev = eval_args.device
    user_train = dataset["user_train"]
    user_valid = dataset["user_valid"]
    user_test  = dataset["user_test"]
    usernum    = dataset["usernum"]
    itemnum    = dataset["itemnum"]

    all_items = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items)

    user_ranks = {}
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
        pfx = prefix_fn(u, seq_np) if prefix_fn else None
        log_feats, _ = model.log2feats(seq_np, prefix_embeds=pfx)
        final_feat = log_feats[0, -1, :]
        scores = all_item_embs.matmul(final_feat)

        seen = set(user_train[u]) | set(user_valid.get(u, []))
        seen.discard(0)
        seen_idx = torch.tensor([i - 1 for i in seen if 1 <= i <= itemnum],
                                dtype=torch.long, device=dev)
        if len(seen_idx) > 0:
            scores[seen_idx] = -1e9

        user_ranks[u] = int((scores > scores[target - 1]).sum().item())

    return user_ranks


def print_stratified(label, deltas, vanilla_ranks):
    n = len(deltas)
    if n == 0:
        print(f"  {label}: n=0")
        return
    improve = sum(1 for d in deltas if d > 0)
    degrade = sum(1 for d in deltas if d < 0)
    same    = n - improve - degrade
    mean_d  = sum(deltas) / n
    import statistics
    std_d = statistics.stdev(deltas) if n > 1 else 0
    se = std_d / (n ** 0.5)
    se_ratio = mean_d / se if se > 0 else 0

    # Recall@K for vanilla and steered
    v_hits = {k: sum(1 for r in vanilla_ranks if r < k) for k in _KS_EXT}
    s_hits = {k: sum(1 for (r, d) in zip(vanilla_ranks, deltas) if r - d < k) for k in _KS_EXT}

    print(f"\n  [{label}] n={n}  improve={improve}  degrade={degrade}  same={same}")
    print(f"    mean_delta={mean_d:+.1f}  std={std_d:.1f}  SE_ratio={se_ratio:+.2f}")
    print(f"    Recall:  ", end="")
    for k in _KS_EXT:
        vr = v_hits[k] / n
        sr = s_hits[k] / n
        print(f"@{k}:{vr:.4f}→{sr:.4f}({sr-vr:+.4f})  ", end="")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Books")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--sasrec_ckpt", required=True)
    ap.add_argument("--ckpt_2a", required=True)
    ap.add_argument("--ckpt_2b", required=True)
    ap.add_argument("--bank_jsonl", required=True)
    ap.add_argument("--eval_queries_jsonl", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load SASRec
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
    print(f"SASRec: users={dataset['usernum']} items={dataset['itemnum']}")

    # Load bank + eval queries
    user_memories = _load_bank_full(args.bank_jsonl)
    user_queries_eval, _ = _load_eval_queries(args.eval_queries_jsonl)
    print(f"bank users={len(user_memories)}  eval_query users={len(user_queries_eval)}")

    # Build K map (per user)
    user_k = {}
    for uid_s, mems in user_memories.items():
        user_k[int(uid_s)] = len(mems)

    # Load seq lengths
    with open(f"{args.data_dir}/{args.category}/splits.json") as f:
        splits = json.load(f)
    user_seqlen = {int(uid): len(v.get("train", [])) for uid, v in splits["users"].items()}

    # ── Vanilla run ──
    print("\nRunning vanilla...")
    vanilla_ranks = run_eval(model, dataset, eval_args, None)

    results = {}
    for name, ckpt_path in [("2a", args.ckpt_2a), ("2b", args.ckpt_2b)]:
        print(f"\nRunning {name}...")
        text_enc, proj = load_stage2(ckpt_path, device)
        pfn = make_steered_prefix_fn(text_enc, proj, user_queries_eval, user_memories, device)
        steered_ranks = run_eval(model, dataset, eval_args, pfn)
        results[name] = steered_ranks

    # ── Analysis ──
    all_users = list(vanilla_ranks.keys())

    for name, steered_ranks in results.items():
        print(f"\n{'='*60}")
        print(f"  {name} vs vanilla")
        print(f"{'='*60}")

        all_deltas = [vanilla_ranks[u] - steered_ranks[u] for u in all_users if u in steered_ranks]
        all_vranks = [vanilla_ranks[u] for u in all_users if u in steered_ranks]
        print_stratified("ALL", all_deltas, all_vranks)

        # K 계층화
        for k_label, k_filter in [("K=1", lambda k: k==1),
                                   ("K>=2", lambda k: k>=2)]:
            us = [u for u in all_users if u in steered_ranks and k_filter(user_k.get(u, 0))]
            deltas = [vanilla_ranks[u] - steered_ranks[u] for u in us]
            vranks = [vanilla_ranks[u] for u in us]
            print_stratified(k_label, deltas, vranks)

        # 시퀀스 길이 계층화
        for sl_label, sl_filter in [("seq<=5", lambda s: s<=5),
                                     ("seq 6-15", lambda s: 6<=s<=15),
                                     ("seq>15", lambda s: s>15)]:
            us = [u for u in all_users if u in steered_ranks and sl_filter(user_seqlen.get(u, 0))]
            deltas = [vanilla_ranks[u] - steered_ranks[u] for u in us]
            vranks = [vanilla_ranks[u] for u in us]
            print_stratified(sl_label, deltas, vranks)

        # Rank delta 분포: 개선이 어느 범위에서 일어나는가
        buckets = [(0, 10, "rank 0-9(top10)"), (10, 50, "rank 10-49"), (50, 200, "rank 50-199"),
                   (200, 9999, "rank 200+")]
        print("\n  Vanilla rank bucket별 steered 개선율:")
        for lo, hi, blabel in buckets:
            us = [u for u in all_users if u in steered_ranks and lo <= vanilla_ranks[u] < hi]
            if not us: continue
            improved = sum(1 for u in us if steered_ranks[u] < vanilla_ranks[u])
            degraded = sum(1 for u in us if steered_ranks[u] > vanilla_ranks[u])
            mean_d = sum(vanilla_ranks[u] - steered_ranks[u] for u in us) / len(us)
            print(f"    {blabel}: n={len(us)}  improved={improved}({100*improved/len(us):.0f}%)  "
                  f"degraded={degraded}({100*degraded/len(us):.0f}%)  mean_delta={mean_d:+.1f}")

    # ── 2a vs 2b paired ──
    print(f"\n{'='*60}")
    print("  2a vs 2b PAIRED (same users)")
    print(f"{'='*60}")
    paired = [(u, results["2a"][u], results["2b"][u])
              for u in all_users if u in results["2a"] and u in results["2b"]]
    a_better = sum(1 for _, a, b in paired if a < b)
    b_better = sum(1 for _, a, b in paired if b < a)
    tied     = len(paired) - a_better - b_better
    print(f"  n={len(paired)}  2a_better={a_better}  2b_better={b_better}  tied={tied}")
    deltas_ab = [b - a for _, a, b in paired]  # positive = 2a is better (lower rank)
    mean_ab = sum(deltas_ab) / len(deltas_ab)
    import statistics
    std_ab = statistics.stdev(deltas_ab)
    se_ab = std_ab / len(deltas_ab) ** 0.5
    print(f"  mean(2b_rank - 2a_rank) = {mean_ab:+.2f}  SE_ratio={mean_ab/se_ab:+.2f}")

    for k_label, k_filter in [("K=1", lambda k: k==1), ("K>=2", lambda k: k>=2)]:
        sub = [(u, a, b) for u, a, b in paired if k_filter(user_k.get(u, 0))]
        if not sub: continue
        d = [b - a for _, a, b in sub]
        m = sum(d) / len(d)
        s = statistics.stdev(d) if len(d) > 1 else 0
        se = s / len(d) ** 0.5
        a_b = sum(1 for x in d if x > 0)
        b_b = sum(1 for x in d if x < 0)
        print(f"  {k_label}: n={len(sub)}  2a_better={a_b}  2b_better={b_b}  "
              f"mean={m:+.2f}  SE_ratio={m/se if se>0 else 0:+.2f}")


if __name__ == "__main__":
    main()
