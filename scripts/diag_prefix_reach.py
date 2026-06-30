# -*- coding: utf-8 -*-
"""STEP 1: prefix->h_last 전달 진단.

측정:
  A. rel_change = |h_last(with_pfx) - h_last(no_pfx)| / |h_last(no_pfx)|
     prefix가 있을 때 h_last가 얼마나 바뀌나. 큰 값 → 전달 됨.

  B. h_diff = |h_last(m+) - h_last(m-)| / |h_last(no_pfx)|
     m+ vs m- 차이가 h_last에 얼마나 남나.
     D_p = |prefix(m+) - prefix(m-)| 와 비교하면
     attenuation ratio = h_diff / D_p.

  C. 방향 정렬 (direction alignment):
     Δh = h_last(m+) - h_last(m-)
     item_emb_y = embedding of test target W
     cos(Δh, item_emb_y) — 양수면 m+쪽으로 이동이 W 방향과 일치.
     pos_frac: cos>0 유저 비율. 50%이면 완전 무작위.

  D. Δz_y를 h_last 변화로 분해:
     Δz_y = item_emb_y · (h_last(m+) - h_last(m-))
           = |h_diff| · cos(Δh, item_emb_y) · |item_emb_y|
     Δz_y≈0의 원인: |h_diff|≈0(전달 실패=B) vs cos≈0(방향 무작위=A).

판정 기준:
  - rel_change < 0.01 → prefix가 h_last에 거의 도달 안 함 (원인 B 확정)
  - rel_change ≥ 0.01 + h_diff/D_p 비율 측정 후:
      h_diff/D_p ≥ 0.01 이면 전달은 됨
      cos_align pos_frac ≈ 50% → 방향 무작위 (원인 A 확정)
      cos_align pos_frac >> 50% → 방향 정렬됨 (다른 원인)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sasrec import SASRec
from src.models.projector import IntentProjector
from src.models.dataloader import load_data
from src.training.train_hybrid import (
    TextEncoder,
    _load_bank_full,
    _load_pseudo_queries,
)
from scripts.eval_steered import _load_user_first_pos_mid
from scripts.measure_boundary import load_stage2


# ---------------------------------------------------------------------------

@torch.no_grad()
def run_step1(
    model,
    dataset: dict,
    eval_args,
    text_enc: TextEncoder,
    proj: IntentProjector,
    user_queries: dict[str, list[str]],
    user_memories: dict[str, list[dict]],
    user_first_mid: dict[str, str],
    label: str,
    n_sample: int = 500,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    dev = eval_args.device
    itemnum = dataset["itemnum"]
    user_train = dataset["user_train"]
    user_test = dataset["user_test"]

    # candidates: query + ≥2 memories (intra-user neg) + test target
    candidates = [
        u for u in range(1, dataset["usernum"] + 1)
        if user_test.get(u)
        and user_queries.get(str(u))
        and user_memories.get(str(u))
        and user_first_mid.get(str(u))
        and len(user_memories[str(u)]) >= 2  # 인트라유저 neg
    ]
    if not candidates:
        # fallback: cross-user neg
        candidates = [
            u for u in range(1, dataset["usernum"] + 1)
            if user_test.get(u)
            and user_queries.get(str(u))
            and user_memories.get(str(u))
            and user_first_mid.get(str(u))
        ]

    if len(candidates) > n_sample:
        candidates = random.Random(seed).sample(candidates, n_sample)

    all_uid_strs = list(user_memories.keys())
    all_items_t = torch.arange(1, itemnum + 1, dtype=torch.long, device=dev)
    all_item_embs = model.item_emb(all_items_t)  # [I, d]

    rel_changes: list[float] = []   # A: |h_pos - h_none| / |h_none|
    h_diffs: list[float] = []       # B: |h_pos - h_neg| (raw L2)
    dp_vals: list[float] = []       # D_p: |pfx_pos - pfx_neg|
    cos_aligns: list[float] = []    # C: cos(Δh, item_emb_y)
    dz_vals: list[float] = []       # Δz_y = z_y(m+) - z_y(m-)
    h_none_norms: list[float] = []  # |h_last(no prefix)|

    n_intra = 0

    for u in candidates:
        uid_s = str(u)
        mems = user_memories[uid_s]
        pos_mid = user_first_mid[uid_s]
        pos_mem = next((m for m in mems if m["mid"] == pos_mid), None)
        if pos_mem is None:
            continue

        # negative memory
        if len(mems) >= 2:
            neg_mem = next((m for m in mems if m["mid"] != pos_mid), None)
            if neg_mem is None:
                continue
            n_intra += 1
        else:
            other_uid = uid_s
            for _ in range(20):
                other_uid = rng.choice(all_uid_strs)
                if other_uid != uid_s:
                    break
            if other_uid == uid_s:
                continue
            neg_mem = user_memories[other_uid][0]

        query_text = user_queries[uid_s][0]
        target = user_test[u][0]
        if not (1 <= target <= itemnum):
            continue

        # encode
        h_q = text_enc.encode([query_text], is_query=True)
        h_mpos = text_enc.encode([pos_mem["text"]], is_query=False)
        h_mneg = text_enc.encode([neg_mem["text"]], is_query=False)

        pfx_pos = proj(h_q, h_mpos)   # [1, 1, d_sasrec]
        pfx_neg = proj(h_q, h_mneg)

        # D_p
        dp = (pfx_pos - pfx_neg).norm(dim=-1).mean().item()
        dp_vals.append(dp)

        # build sequence
        seq = np.zeros([eval_args.maxlen], dtype=np.int32)
        idx = eval_args.maxlen - 1
        for i in reversed(user_train.get(u, [])):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1
        seq_np = seq[np.newaxis, :]

        # h_last: no prefix, m+, m-
        feats_none, _ = model.log2feats(seq_np, prefix_embeds=None)
        feats_pos,  _ = model.log2feats(seq_np, prefix_embeds=pfx_pos.to(dev))
        feats_neg,  _ = model.log2feats(seq_np, prefix_embeds=pfx_neg.to(dev))

        h_none = feats_none[0, -1, :]   # [d]
        h_pos  = feats_pos[0,  -1, :]
        h_neg  = feats_neg[0,  -1, :]

        h_none_norm = h_none.norm().item()
        h_none_norms.append(h_none_norm)

        # A: relative change (m+ vs no prefix)
        rel = (h_pos - h_none).norm().item() / (h_none_norm + 1e-8)
        rel_changes.append(rel)

        # B: |h_last(m+) - h_last(m-)| raw L2
        hdiff = (h_pos - h_neg).norm().item()
        h_diffs.append(hdiff)

        # C: cos alignment of Δh with target item emb
        delta_h = h_pos - h_neg   # [d]
        item_emb_y = all_item_embs[target - 1]   # [d]
        cos = torch.nn.functional.cosine_similarity(
            delta_h.unsqueeze(0), item_emb_y.unsqueeze(0)
        ).item()
        cos_aligns.append(cos)

        # Δz_y
        z_pos = item_emb_y.dot(h_pos).item()
        z_neg = item_emb_y.dot(h_neg).item()
        dz_vals.append(z_pos - z_neg)

    n = len(rel_changes)
    if n == 0:
        print(f"  [STEP1:{label}] n=0, skip")
        return {}

    def _stats(vals):
        m = sum(vals) / len(vals)
        std = (sum((x - m)**2 for x in vals) / len(vals)) ** 0.5
        s = sorted(vals)
        return m, std, s[int(0.1*len(s))], s[int(0.5*len(s))], s[int(0.9*len(s))]

    rel_m, rel_std, rel_p10, rel_p50, rel_p90 = _stats(rel_changes)
    hdiff_m, hdiff_std, _, hdiff_p50, _ = _stats(h_diffs)
    dp_m, dp_std, _, dp_p50, _ = _stats(dp_vals)
    cos_m, cos_std, cos_p10, cos_p50, cos_p90 = _stats(cos_aligns)
    dz_m, dz_std, dz_p10, dz_p50, dz_p90 = _stats(dz_vals)
    h_none_m = sum(h_none_norms) / len(h_none_norms)

    # attenuation ratio
    atten = hdiff_m / (dp_m + 1e-8)

    cos_pos_frac = sum(1 for c in cos_aligns if c > 0) / len(cos_aligns)
    dz_pos_frac = sum(1 for x in dz_vals if x > 0) / len(dz_vals)

    print(f"\n{'─'*60}")
    print(f"  STEP 1 — prefix→h_last 전달 진단  [{label}]  n={n}  intra_neg={n_intra}")
    print(f"{'─'*60}")
    print(f"  h_none norm (no prefix):  mean={h_none_m:.4f}")
    print()
    print(f"  A. rel_change |h(m+)-h(none)| / |h(none)|:")
    print(f"     mean={rel_m:.4f}  std={rel_std:.4f}  "
          f"p10={rel_p10:.4f}  p50={rel_p50:.4f}  p90={rel_p90:.4f}")
    print()
    print(f"  B. h_diff |h(m+)-h(m-)| (raw L2):")
    print(f"     mean={hdiff_m:.4f}  p50={hdiff_p50:.4f}")
    print(f"     D_p  |pfx(m+)-pfx(m-)| : mean={dp_m:.4f}  p50={dp_p50:.4f}")
    print(f"     attenuation h_diff/D_p  : {atten:.4f}  "
          f"({'LOW → 전달 실패' if atten < 0.01 else 'OK → 전달됨'})")
    print()
    print(f"  C. cos(Δh, item_emb_y) — m+ 이동이 target W 방향인가:")
    print(f"     mean={cos_m:+.4f}  std={cos_std:.4f}  "
          f"p10={cos_p10:+.4f}  p50={cos_p50:+.4f}  p90={cos_p90:+.4f}")
    print(f"     pos_frac={cos_pos_frac:.2%}  "
          f"({'≈50% → 방향 무작위 = 원인A' if 0.4 < cos_pos_frac < 0.6 else '유의 편향'})")
    print()
    print(f"  D. Δz_y = z_y(m+) - z_y(m-):")
    print(f"     mean={dz_m:+.4f}  std={dz_std:.4f}  "
          f"p10={dz_p10:+.4f}  p50={dz_p50:+.4f}  p90={dz_p90:+.4f}")
    print(f"     pos_frac={dz_pos_frac:.2%}  "
          f"({'≈50% 무작위' if 0.4 < dz_pos_frac < 0.6 else '유의 편향'})")

    print()
    # Verdict
    if rel_m < 0.01:
        verdict = "원인B: prefix가 h_last에 거의 도달하지 않음 → prefix 주입 위치/방식 재검토"
    elif atten < 0.01:
        verdict = "원인B: D_p는 큰데 h_last 차이가 작음 → 다층 attention에서 감쇠"
    elif 0.4 < cos_pos_frac < 0.6:
        verdict = "원인A: h_last는 바뀌나 방향이 무작위 → 목적함수에 score-space 신호 없음"
    else:
        verdict = f"복합 또는 다른 원인: cos_pos_frac={cos_pos_frac:.2%}, atten={atten:.4f}"
    print(f"  ★ 판정: {verdict}")

    return {
        "n": n, "n_intra": n_intra,
        "h_none_norm_mean": h_none_m,
        "rel_change_mean": rel_m, "rel_change_p50": rel_p50,
        "hdiff_mean": hdiff_m, "dp_mean": dp_m,
        "attenuation": atten,
        "cos_align_mean": cos_m, "cos_pos_frac": cos_pos_frac,
        "dz_mean": dz_m, "dz_pos_frac": dz_pos_frac,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP1: prefix→h_last 전달 진단")
    ap.add_argument("--category",    default="Books")
    ap.add_argument("--data_dir",    default="data/processed")
    ap.add_argument("--sasrec_ckpt", default="checkpoints/Books/sasrec_pretrain.pt")
    ap.add_argument("--ckpt_2a",     default="checkpoints/Books/stage2_stage1enc_best.pt")
    ap.add_argument("--ckpt_2b",     default="checkpoints/Books/stage2_rawbge_best.pt")
    ap.add_argument("--bank_jsonl",  default="data/memory/Books/f3_bank.jsonl")
    ap.add_argument("--pairs_jsonl", default=None)
    ap.add_argument("--device",      default="cpu")
    ap.add_argument("--n_sample",    type=int, default=500)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    device = args.device
    pairs_jsonl = args.pairs_jsonl or f"data/processed/{args.category}/align_pairs.jsonl"

    # SASRec
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

    # Data
    print(f"\n[data] Loading bank + pairs ...")
    user_memories = _load_bank_full(args.bank_jsonl)
    mid_to_uid = {m["mid"]: uid for uid, mems in user_memories.items() for m in mems}
    user_queries, _ = _load_pseudo_queries(pairs_jsonl, mid_to_uid=mid_to_uid)
    user_first_mid = _load_user_first_pos_mid(pairs_jsonl, mid_to_uid)
    print(f"  bank_users={len(user_memories)}  query_users={len(user_queries)}  "
          f"first_mid_users={len(user_first_mid)}")

    all_results = {}

    for tag, ckpt_path in [("2a", args.ckpt_2a), ("2b", args.ckpt_2b)]:
        if not Path(ckpt_path).exists():
            print(f"\n[skip] {tag}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Checkpoint: {tag}  ({ckpt_path})")
        print(f"{'='*60}")

        text_enc, proj = load_stage2(ckpt_path, device)
        result = run_step1(
            model, dataset, eval_args, text_enc, proj,
            user_queries, user_memories, user_first_mid,
            label=tag, n_sample=args.n_sample, seed=args.seed,
        )
        all_results[tag] = result

    # Also run STEP2: L_retrieval signal check (static analysis)
    print(f"\n{'='*60}")
    print("  STEP 2 — L_retrieval BPR 목적함수 신호 분석 (정적)")
    print(f"{'='*60}")
    print("""  L_retrieval = BPR(pos_logits, neg_logits)
  - pos_logits[t] = h_t · item_emb(next_item_t)   (t = 모든 train 위치)
  - neg_logits[t] = h_t · item_emb(random_neg_t)
  - prefix(m+)를 쓸 때 h_t가 바뀌므로 BPR은 간접적으로 prefix에 신호를 줌.
  - 단, 신호 방향: "train-history next-item 예측이 잘 되게" — test target W(미래)에 대한
    직접 신호는 없음 (W ∉ train_history → LOO).
  - 따라서 L_retrieval이 "m+ prefix → W score ↑" 신호를 주는 경로는 없음.
  - L_align: cos(prefix[:,0,:], item_emb(X)) — prefix vector를 X 방향으로 밀지만
    *h_last가 W를 더 높게 score하도록* 직접 가르치지 않음 (prefix space ≠ h_last space).
  ★ 결론: BPR + L_align 조합에서 "m+ → h_last(m+) 방향이 m- 대비 W에 가깝다"는
    직접적인 gradient 경로가 없음 → Δz_y≈0 + pos50%의 학습 측면 원인 확정.""")

    # Save
    out_dir = Path("results/prefix_reach")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.category}_prefix_reach.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
