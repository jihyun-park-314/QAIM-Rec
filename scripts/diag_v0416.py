"""v0.4.16 진단 스크립트 — STEP1 + STEP2

STEP1: Stage1 encoder로 K≥2 intra-user routing accuracy 측정 (K별 분해)
       0.958이 K=1 trivial로 부풀려진건지 확인.
STEP2: align_pair 쿼리 출처 item X vs LOO 타겟 item W 정렬도 측정
       (a) W가 X의 positive 클러스터에 있는 비율  → L_align이 정확히 W방향을 학습하는 비율
       (b) W가 유저의 어떤 메모리에도 없는 비율  → steering signal이 아예 없는 유저 비율

Usage:
    python3 scripts/diag_v0416.py \
        --bank_dir   data/processed/Books/memory_bank \
        --pairs      data/processed/Books/align_pairs.jsonl \
        --queries    data/processed/Books/pseudo_queries_train.jsonl \
        --splits     data/processed/Books/splits.json \
        --stage1_ckpt checkpoints/Books/stage1_align_best.pt \
        --device     cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────
# 1. Loaders
# ─────────────────────────────────────────────

def load_bank(bank_dir: str):
    """Returns:
      mem_to_vec  {memory_id: np.float32 [768]}
      mem_to_user {memory_id: str(uid)}
      user_to_mems {str(uid): [memory_id, ...]}
      mem_to_items {memory_id: set(int item_id)}  (evidence)
    """
    mem_to_vec: dict[str, np.ndarray] = {}
    mem_to_user: dict[str, str] = {}
    user_to_mems: dict[str, list[str]] = {}
    mem_to_items: dict[str, set] = {}

    for fpath in sorted(Path(bank_dir).glob("[0-9]*.json")):
        with open(fpath, encoding="utf-8") as f:
            ub = json.load(f)
        uid = str(ub["user_id"])
        k = ub.get("k_personal", 0)
        if k < 1:
            continue
        mids = []
        for unit in ub.get("units", []):
            mid = unit["memory_id"]
            vec = unit.get("embedding", {}).get("vector")
            if vec:
                mem_to_vec[mid] = np.array(vec, dtype=np.float32)
            mem_to_user[mid] = uid
            item_ids = set(unit.get("evidence", {}).get("item_ids", []))
            mem_to_items[mid] = item_ids
            mids.append(mid)
        user_to_mems[uid] = mids

    print(f"[bank] {len(mem_to_vec)} memories / {len(user_to_mems)} users")
    return mem_to_vec, mem_to_user, user_to_mems, mem_to_items


def load_align_pairs(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"[pairs] {len(pairs)} align_pairs loaded")
    return pairs


def load_queries(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[queries] {len(rows)} pseudo queries loaded")
    return rows


def load_splits(path: str) -> dict[str, int]:
    """Returns {str(uid): test_item_id (int)}"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    users = d.get("users", d)
    result = {}
    for uid, split in users.items():
        test_item = split.get("test")
        if test_item is not None:
            result[str(uid)] = int(test_item)
    print(f"[splits] {len(result)} users with test item")
    return result


# ─────────────────────────────────────────────
# 2. Stage1 encoder loader
# ─────────────────────────────────────────────

def load_stage1_encoder(ckpt_path: str, device: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.training.train_align import TextEncoderWithHead
    model = TextEncoderWithHead(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"[stage1] loaded checkpoint: {ckpt_path}")
    return model


def embed_queries_with_model(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs = model.encode_queries(batch)
            all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0)


def embed_queries_frozen_bge(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs = model.encode_queries_frozen_bge(batch)
            all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0)


# ─────────────────────────────────────────────
# 3. STEP 1: K≥2 intra-user routing accuracy
# ─────────────────────────────────────────────

def run_step1(
    pairs: list[dict],
    mem_to_vec: dict,
    mem_to_user: dict,
    user_to_mems: dict,
    model,
    device: str,
):
    """Routing accuracy stratified by K value.

    For K=1 users: trivially 1.0 (only 1 candidate).
    Report K=1 / K=2 / K=3 / K=4 / K≥5 / K≥2(aggregate).
    """
    print("\n" + "="*60)
    print("STEP 1 — K≥2 intra-user routing accuracy (Stage1 encoder)")
    print("="*60)

    # K map per user
    k_map = {uid: len(mids) for uid, mids in user_to_mems.items()}

    # Filter valid pairs (query non-null, pos in bank, user has ≥1 memory)
    valid = []
    for p in pairs:
        q = p.get("query", "")
        if not q:
            continue
        pos_mid = p["positive_memory_id"]
        if pos_mid not in mem_to_vec:
            continue
        uid = mem_to_user.get(pos_mid)
        if uid is None or uid not in user_to_mems:
            continue
        valid.append((uid, q, pos_mid))

    print(f"  valid pairs: {len(valid)} (out of {len(pairs)})")

    # Embed all queries
    print(f"  embedding {len(valid)} queries with Stage1 encoder...")
    texts = [v[1] for v in valid]
    q_vecs = embed_queries_with_model(model, texts)  # [N, 768]

    # Also compute frozen-bge baseline for comparison
    print(f"  embedding {len(valid)} queries with frozen-bge baseline...")
    q_vecs_frozen = embed_queries_frozen_bge(model, texts)

    # Evaluate
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}  # 5 = K≥5

    for i, (uid, query, pos_mid) in enumerate(valid):
        user_mids = user_to_mems[uid]
        k = k_map[uid]
        mem_matrix = np.stack([mem_to_vec[m] for m in user_mids])  # [K, 768]

        # Stage1
        sim = mem_matrix @ q_vecs[i]
        top1 = user_mids[int(np.argmax(sim))]
        hit_s1 = int(top1 == pos_mid)

        # Frozen-bge
        sim_f = mem_matrix @ q_vecs_frozen[i]
        top1_f = user_mids[int(np.argmax(sim_f))]
        hit_frozen = int(top1_f == pos_mid)

        bucket_key = min(k, 5)
        buckets[bucket_key].append((hit_s1, hit_frozen))

    # Report
    print(f"\n  {'K':>6}  {'n':>7}  {'Stage1 acc':>12}  {'frozen-bge':>12}  {'delta':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*8}")

    all_s1, all_frozen = [], []
    kge2_s1, kge2_frozen = [], []

    for k_key in sorted(buckets.keys()):
        data = buckets[k_key]
        if not data:
            continue
        s1_hits = [d[0] for d in data]
        fr_hits = [d[1] for d in data]
        acc_s1 = np.mean(s1_hits)
        acc_fr = np.mean(fr_hits)
        k_label = f"K={k_key}" if k_key < 5 else "K≥5"
        print(f"  {k_label:>6}  {len(data):>7}  {acc_s1:>12.4f}  {acc_fr:>12.4f}  {acc_s1-acc_fr:>+8.4f}")
        all_s1.extend(s1_hits)
        all_frozen.extend(fr_hits)
        if k_key >= 2:
            kge2_s1.extend(s1_hits)
            kge2_frozen.extend(fr_hits)

    print(f"  {'──────':>6}  {'──────':>7}  {'──────────':>12}  {'──────────':>12}  {'────────':>8}")
    print(f"  {'K≥2':>6}  {len(kge2_s1):>7}  {np.mean(kge2_s1):>12.4f}  {np.mean(kge2_frozen):>12.4f}  {np.mean(kge2_s1)-np.mean(kge2_frozen):>+8.4f}")
    print(f"  {'ALL':>6}  {len(all_s1):>7}  {np.mean(all_s1):>12.4f}  {np.mean(all_frozen):>12.4f}  {np.mean(all_s1)-np.mean(all_frozen):>+8.4f}")

    # Hard-neg composition analysis
    print("\n  [hard-neg composition in align_pairs]")
    intra_total, cross_total, total_negs = 0, 0, 0
    for p in pairs:
        pos_mid = p.get("positive_memory_id", "")
        uid = mem_to_user.get(pos_mid)
        if uid is None:
            continue
        user_mems_set = set(user_to_mems.get(uid, []))
        for neg_mid in p.get("hard_negative_memory_ids", []):
            total_negs += 1
            if neg_mid in user_mems_set:
                intra_total += 1
            else:
                cross_total += 1

    if total_negs > 0:
        print(f"    total hard-negs: {total_negs}")
        print(f"    intra-user: {intra_total} ({100*intra_total/total_negs:.1f}%)")
        print(f"    cross-user: {cross_total} ({100*cross_total/total_negs:.1f}%)")

    result = {
        "step1_kge2_stage1": float(np.mean(kge2_s1)) if kge2_s1 else None,
        "step1_kge2_frozen": float(np.mean(kge2_frozen)) if kge2_frozen else None,
        "step1_all_stage1": float(np.mean(all_s1)) if all_s1 else None,
        "step1_all_frozen": float(np.mean(all_frozen)) if all_frozen else None,
        "step1_k_breakdown": {
            k_key: {
                "n": len(buckets[k_key]),
                "stage1_acc": float(np.mean([d[0] for d in buckets[k_key]])) if buckets[k_key] else None,
                "frozen_acc": float(np.mean([d[1] for d in buckets[k_key]])) if buckets[k_key] else None,
            }
            for k_key in sorted(buckets.keys()) if buckets[k_key]
        },
        "hard_neg_intra_rate": intra_total / total_negs if total_negs > 0 else None,
        "hard_neg_cross_rate": cross_total / total_negs if total_negs > 0 else None,
    }
    return result


# ─────────────────────────────────────────────
# 4. STEP 2: X-W alignment
# ─────────────────────────────────────────────

def run_step2(
    queries: list[dict],
    mem_to_items: dict[str, set],
    user_to_mems: dict[str, list[str]],
    splits: dict[str, int],
    mem_to_user: dict[str, str],
):
    """Measure alignment between query source item X and LOO target item W.

    For each pseudo query:
      X = source_item_id
      W = splits[user_id]['test']
      positive_memory_id → evidence.item_ids → does W appear? (direct hit)
      Also: does W appear in ANY memory of this user? (any coverage)
    """
    print("\n" + "="*60)
    print("STEP 2 — X vs W 정렬도 (쿼리출처 vs LOO타겟)")
    print("="*60)

    n_total = 0
    n_w_missing = 0       # W not in splits (shouldn't happen)
    n_direct_hit = 0      # W is in X's positive cluster evidence
    n_any_coverage = 0    # W is in ANY memory of this user
    n_x_eq_w = 0          # X == W (query source is the test item itself)
    n_same_cluster = 0    # X and W are in the same positive cluster

    # K 별 직접 hit 분석
    k_direct: dict[int, list] = defaultdict(list)

    # 유저별 W coverage 상세
    uncovered_users = 0

    for row in queries:
        uid = str(row.get("user_id", ""))
        x = row.get("source_item_id")
        pos_mid = row.get("positive_memory_id", "")

        if x is None or uid not in splits:
            n_w_missing += 1
            continue

        w = splits[uid]
        n_total += 1

        if x == w:
            n_x_eq_w += 1

        # W in positive cluster?
        pos_items = mem_to_items.get(pos_mid, set())
        direct_hit = w in pos_items
        if direct_hit:
            n_direct_hit += 1
            n_same_cluster += 1

        # W in ANY memory of user?
        user_mems = user_to_mems.get(uid, [])
        any_cov = any(w in mem_to_items.get(m, set()) for m in user_mems)
        if any_cov:
            n_any_coverage += 1

        k = len(user_mems)
        k_direct[min(k, 5)].append(int(direct_hit))

    # Users with no W coverage at all
    user_seen = set()
    for row in queries:
        uid = str(row.get("user_id", ""))
        if uid in user_seen or uid not in splits:
            continue
        user_seen.add(uid)
        w = splits[uid]
        user_mems = user_to_mems.get(uid, [])
        any_cov = any(w in mem_to_items.get(m, set()) for m in user_mems)
        if not any_cov:
            uncovered_users += 1

    print(f"\n  총 쿼리 (W 있음): {n_total}")
    print(f"  W not in splits (skip): {n_w_missing}")
    print(f"  X == W (쿼리출처 = 테스트아이템): {n_x_eq_w} ({100*n_x_eq_w/n_total:.1f}%)")
    print()
    print(f"  ★ W ∈ positive cluster evidence (직접 정렬): {n_direct_hit} / {n_total}  = {100*n_direct_hit/n_total:.1f}%")
    print(f"    (= L_align이 W방향으로 직접 학습되는 비율)")
    print()
    print(f"  W ∈ any user memory (간접 커버): {n_any_coverage} / {n_total}  = {100*n_any_coverage/n_total:.1f}%")
    print(f"    (= 다른 클러스터라도 W가 메모리에 있는 비율)")
    print()
    print(f"  W가 어떤 메모리에도 없는 유저: {uncovered_users} / {len(user_seen)}")
    print(f"    (= Stage2 steering에 W signal이 전혀 없는 유저 비율: {100*uncovered_users/max(1,len(user_seen)):.1f}%)")

    print(f"\n  [K별 직접 정렬율 (W ∈ positive cluster)]")
    print(f"  {'K':>6}  {'n_queries':>10}  {'direct_hit_rate':>16}")
    for k_key in sorted(k_direct.keys()):
        vals = k_direct[k_key]
        k_label = f"K={k_key}" if k_key < 5 else "K≥5"
        print(f"  {k_label:>6}  {len(vals):>10}  {100*np.mean(vals):>15.1f}%")

    result = {
        "step2_n_total": n_total,
        "step2_x_eq_w_rate": n_x_eq_w / n_total if n_total else None,
        "step2_direct_hit_rate": n_direct_hit / n_total if n_total else None,
        "step2_any_coverage_rate": n_any_coverage / n_total if n_total else None,
        "step2_uncovered_users": uncovered_users,
        "step2_uncovered_user_rate": uncovered_users / len(user_seen) if user_seen else None,
        "step2_k_direct": {
            k_key: {
                "n": len(vals),
                "direct_hit_rate": float(np.mean(vals)) if vals else None,
            }
            for k_key, vals in sorted(k_direct.items())
        },
    }
    return result


# ─────────────────────────────────────────────
# 5. 종합 해석
# ─────────────────────────────────────────────

def interpret(r1: dict, r2: dict):
    print("\n" + "="*60)
    print("종합 해석")
    print("="*60)

    kge2_s1 = r1.get("step1_kge2_stage1")
    kge2_fr = r1.get("step1_kge2_frozen")
    all_s1  = r1.get("step1_all_stage1")

    print(f"\n[문제 2 — routing 과대평가]")
    print(f"  보고된 전체 acc: {all_s1:.4f}  ← K=1(trivially 1.0) 포함")
    if kge2_s1 is not None:
        verdict = "과대평가 확인" if kge2_s1 < 0.85 else ("경계" if kge2_s1 < 0.90 else "양호")
        print(f"  K≥2 intra-user acc: {kge2_s1:.4f}  [{verdict}]")
        print(f"  Stage1 gain vs frozen: {kge2_s1 - kge2_fr:+.4f}")
    intra_rate = r1.get("hard_neg_intra_rate", 0)
    print(f"  hard-neg intra-user 비율: {100*(intra_rate or 0):.1f}%")
    if (intra_rate or 0) < 0.15:
        print(f"  → 진단: hard-neg가 대부분 cross-user → K≥2 구별 학습 불충분")

    direct = r2.get("step2_direct_hit_rate", 0)
    any_cov = r2.get("step2_any_coverage_rate", 0)
    uncov_rate = r2.get("step2_uncovered_user_rate", 0)

    print(f"\n[문제 1 — L_align 타겟 미정렬]")
    print(f"  W ∈ positive cluster (직접 정렬): {100*direct:.1f}%")
    print(f"  W ∈ any memory (간접 커버):        {100*any_cov:.1f}%")
    print(f"  W 완전 미커버 유저:                {100*uncov_rate:.1f}%")

    print(f"\n[설계결정 권고]")

    if direct < 0.10:
        print(f"  ★ L_align 타겟 → 선택지 A (쿼리출처 item으로 대체) 강권")
        print(f"    근거: W ∈ positive cluster {100*direct:.1f}% < 10% — LOO 타겟과 학습방향 거의 무관")
    elif direct < 0.30:
        print(f"  △ L_align 타겟 → 선택지 A 권장 (직접 정렬 {100*direct:.1f}%로 미흡)")
    else:
        print(f"  ○ direct hit {100*direct:.1f}% — 현재 LOO 타겟도 약하게는 작동 가능")
        print(f"    단, {100*(1-direct):.1f}%는 여전히 미정렬이므로 A 또는 혼합 검토 권장")

    if (intra_rate or 0) < 0.15 and kge2_s1 is not None and kge2_s1 < 0.88:
        print(f"\n  ★ Stage1 재설계 필요:")
        print(f"    hard-neg를 intra-user 1순위로 재생성 (현재 intra {100*(intra_rate or 0):.1f}%)")
        print(f"    → K≥2 내부 구별 학습이 핵심 과제")


# ─────────────────────────────────────────────
# 6. main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_dir",    default="data/processed/Books/memory_bank")
    ap.add_argument("--pairs",       default="data/processed/Books/align_pairs.jsonl")
    ap.add_argument("--queries",     default="data/processed/Books/pseudo_queries_train.jsonl")
    ap.add_argument("--splits",      default="data/processed/Books/splits.json")
    ap.add_argument("--stage1_ckpt", default="checkpoints/Books/stage1_align_best.pt")
    ap.add_argument("--device",      default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",         default="data/processed/Books/diag_v0416.json")
    args = ap.parse_args()

    print(f"device: {args.device}")

    # Load data
    mem_to_vec, mem_to_user, user_to_mems, mem_to_items = load_bank(args.bank_dir)
    pairs = load_align_pairs(args.pairs)
    queries = load_queries(args.queries)
    splits = load_splits(args.splits)

    # Load Stage1 model
    model = load_stage1_encoder(args.stage1_ckpt, args.device)

    # Run steps
    r1 = run_step1(pairs, mem_to_vec, mem_to_user, user_to_mems, model, args.device)
    r2 = run_step2(queries, mem_to_items, user_to_mems, splits, mem_to_user)
    interpret(r1, r2)

    # Save
    out = {"step1": r1, "step2": r2}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
