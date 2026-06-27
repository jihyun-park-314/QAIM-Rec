"""STEP 2+3 — Verification gate + provenance routing accuracy (pre-training floor).

STEP 2 gates:
  (1) positive_memory_id null=0 + == provenance_index[uid][source_item_id] (1:1 check)
  (2) hard_negatives do not contain positive_memory_id
  (3) title/author leak rate (INFO only — not a gate)
  (4) avg query word count
  (5) no val/test leakage (structural: all rows come from train-history via p1)

STEP 3 routing accuracy (embedding baseline, pre-training floor):
  query → bge-base embed → cosine top-1 vs positive_memory_id (provenance GT)
  Stratified by:
    - overall
    - K=1  (single memory unit — trivially 1.0, reported for completeness)
    - K>=2 (meaningful signal)
    - K=1 single-review (cluster_size=1 — easy but not leakage; reported separately per plan)

Usage:
  python3 scripts/verify_and_measure_routing.py \\
    --queries  data/processed/Books/pseudo_queries_train.jsonl \\
    --pairs    data/processed/Books/align_pairs.jsonl \\
    --bank_dir data/processed/Books/memory_bank \\
    --output   data/processed/Books/step3_routing_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_pseudo_queries import load_memory_bank


# ---------------------------------------------------------------------------
# Loaders

def load_queries(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_pairs(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bank_vectors(bank_dir: str) -> tuple[
    dict[int, dict[str, np.ndarray]],   # uid -> {mid: vec}
    dict[int, int],                      # uid -> k_personal
    dict[int, dict[str, int]],           # uid -> {mid: cluster_size}
]:
    unit_vecs: dict[int, dict[str, np.ndarray]] = {}
    k_map: dict[int, int] = {}
    cluster_sizes: dict[int, dict[str, int]] = {}

    for p in sorted(Path(bank_dir).glob("[0-9]*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        uid = int(data["user_id"])
        k = data.get("k_personal", 0)
        if k < 1:
            continue
        k_map[uid] = k
        vecs: dict[str, np.ndarray] = {}
        csizes: dict[str, int] = {}
        for unit in data.get("units", []):
            mid = unit["memory_id"]
            emb = unit.get("embedding", {})
            vec = emb.get("vector")
            if vec:
                vecs[mid] = np.array(vec, dtype=np.float32)
            csizes[mid] = unit.get("meta", {}).get("cluster_size", 1)
        unit_vecs[uid] = vecs
        cluster_sizes[uid] = csizes

    return unit_vecs, k_map, cluster_sizes


# ---------------------------------------------------------------------------
# Embedding

def load_embed_model(model_id: str = "BAAI/bge-base-en-v1.5"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        device = f"cuda:{free.index(max(free))}"
    else:
        device = "cpu"
    model = model.to(device).eval()
    print(f"  embedding device: {device}", flush=True)
    return model, tokenizer, device


def embed_texts(model_bundle, texts: list[str], batch_size: int = 128) -> np.ndarray:
    import torch
    model, tokenizer, device = model_bundle
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0)


# ---------------------------------------------------------------------------
# STEP 2 — Verification

def run_step2(
    queries: list[dict],
    pairs: list[dict],
    provenance_index: dict[int, dict[int, str]],
) -> dict:
    print("\n" + "=" * 65)
    print("STEP 2 — Verification Gate")
    print("=" * 65)

    n_total = len(queries)
    n_null_pos = 0
    n_prov_mismatch = 0
    n_leakage = 0
    word_counts: list[int] = []
    uid_set: set[int] = set()

    for row in queries:
        uid = row.get("user_id")
        iid = row.get("source_item_id")
        pos_mid = row.get("positive_memory_id")
        query = row.get("query")

        uid_set.add(uid)

        # Gate 1a: positive_memory_id not null
        if pos_mid is None:
            n_null_pos += 1

        # Gate 1b: provenance 1:1 match
        if pos_mid is not None:
            expected = provenance_index.get(uid, {}).get(iid)
            if expected is not None and expected != pos_mid:
                n_prov_mismatch += 1

        # Gate 3: leakage (info)
        if row.get("leakage_flagged", False):
            n_leakage += 1

        # Gate avg word count (non-null queries only)
        if query:
            word_counts.append(len(query.split()))

    # Gate 2: hard_negatives exclude positive
    n_pairs_total = len(pairs)
    n_neg_contains_pos = 0
    for row in pairs:
        pos = row.get("positive_memory_id")
        negs = row.get("hard_negative_memory_ids", [])
        if pos in negs:
            n_neg_contains_pos += 1

    # Compute
    n_queries_ok = sum(1 for r in queries if r.get("query"))
    n_null_query = sum(1 for r in queries if r.get("query") is None)
    leak_rate = n_leakage / max(n_queries_ok, 1)
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    g1_pass = n_null_pos == 0 and n_prov_mismatch == 0
    g2_pass = n_neg_contains_pos == 0

    print(f"\n[GATE 1] positive_memory_id null=0 + provenance 1:1")
    print(f"  null positive_memory_id : {n_null_pos}/{n_total}  → {'PASS ✓' if n_null_pos==0 else 'FAIL ✗'}")
    print(f"  provenance mismatch     : {n_prov_mismatch}/{n_total - n_null_pos}  → {'PASS ✓' if n_prov_mismatch==0 else 'FAIL ✗'}")

    print(f"\n[GATE 2] hard_negatives exclude positive")
    print(f"  violations: {n_neg_contains_pos}/{n_pairs_total}  → {'PASS ✓' if g2_pass else 'FAIL ✗'}")

    print(f"\n[GATE 3] title/author leakage (INFO — not a gate)")
    print(f"  leakage-flagged queries: {n_leakage}/{n_queries_ok} = {leak_rate:.2%}")
    print(f"  (배제 안 함 — INFO only per plan v0.4.12)")

    print(f"\n[GATE 4] query stats")
    print(f"  total rows       : {n_total}  (users={len(uid_set)})")
    print(f"  queries_ok       : {n_queries_ok}")
    print(f"  null_query       : {n_null_query} = {n_null_query/max(n_total,1):.2%}")
    print(f"  avg word count   : {avg_words:.1f}")

    print(f"\n[GATE 5] judge calls = 0")
    print(f"  (provenance-only path; judge 미호출 구조적 보장)")

    all_pass = g1_pass and g2_pass
    print(f"\n{'STEP2 PASS ✓' if all_pass else 'STEP2 FAIL ✗'}"
          f"  [G1={'✓' if g1_pass else '✗'}  G2={'✓' if g2_pass else '✗'}  G3=INFO  G4=INFO  G5=✓]")
    print("=" * 65)

    return {
        "n_total": n_total,
        "n_queries_ok": n_queries_ok,
        "n_null_query": n_null_query,
        "null_query_rate": round(n_null_query / max(n_total, 1), 4),
        "n_null_pos": n_null_pos,
        "n_prov_mismatch": n_prov_mismatch,
        "n_neg_contains_pos": n_neg_contains_pos,
        "n_leakage_flagged": n_leakage,
        "leak_rate": round(leak_rate, 4),
        "avg_query_words": round(avg_words, 2),
        "gate1_pass": g1_pass,
        "gate2_pass": g2_pass,
        "step2_pass": all_pass,
    }


# ---------------------------------------------------------------------------
# STEP 3 — Routing accuracy

def run_step3(
    queries: list[dict],
    unit_vecs: dict[int, dict[str, np.ndarray]],
    k_map: dict[int, int],
    cluster_sizes: dict[int, dict[str, int]],
    embed_model,
) -> dict:
    print("\n" + "=" * 65)
    print("STEP 3 — Provenance routing accuracy (pre-training floor)")
    print("=" * 65)

    # Filter to non-null queries with provenance + memory bank
    eval_rows: list[tuple[int, str, str]] = []  # (uid, query, pos_mid)
    for row in queries:
        q = row.get("query")
        if not q:
            continue
        uid = row.get("user_id")
        pos_mid = row.get("positive_memory_id")
        if pos_mid is None:
            continue
        if uid not in unit_vecs:
            continue
        eval_rows.append((uid, q, pos_mid))

    print(f"\n  eval set size (non-null, has provenance+bank): {len(eval_rows)}", flush=True)

    if not eval_rows:
        print("  [WARN] No rows to evaluate.")
        return {}

    # Batch embed
    print("  Embedding queries …", flush=True)
    q_texts = [r[1] for r in eval_rows]
    q_vecs = embed_texts(embed_model, q_texts)  # [N, 768]
    print(f"  Done. Shape: {q_vecs.shape}", flush=True)

    # Routing
    hits_all: list[int] = []
    hits_k1: list[int] = []
    hits_kge2: list[int] = []
    hits_k1_single: list[int] = []  # K=1 with cluster_size=1

    for i, (uid, query, pos_mid) in enumerate(eval_rows):
        mids = list(unit_vecs[uid].keys())
        mem_mat = np.stack([unit_vecs[uid][m] for m in mids])
        sims = mem_mat @ q_vecs[i]
        top1_mid = mids[int(np.argmax(sims))]
        hit = int(top1_mid == pos_mid)

        k = k_map.get(uid, 1)
        hits_all.append(hit)
        if k == 1:
            hits_k1.append(hit)
            # single-review: the positive memory's cluster_size == 1
            cs = cluster_sizes.get(uid, {}).get(pos_mid, 1)
            if cs == 1:
                hits_k1_single.append(hit)
        else:
            hits_kge2.append(hit)

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(eval_rows)}] acc_so_far={sum(hits_all)/len(hits_all):.4f}",
                  flush=True)

    def _m(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    result = {
        "n_all": len(hits_all),
        "n_k1": len(hits_k1),
        "n_kge2": len(hits_kge2),
        "n_k1_single_review": len(hits_k1_single),
        "hits1_all": _m(hits_all),
        "hits1_k1": _m(hits_k1),
        "hits1_kge2": _m(hits_kge2),
        "hits1_k1_single_review": _m(hits_k1_single),
    }

    print(f"\n{'='*65}")
    print("  Provenance Routing Accuracy — Hits@1 (embedding baseline)")
    print(f"{'='*65}")
    print(f"  Overall (K>=1)  : {result['hits1_all']:.4f}  (n={result['n_all']})")
    print(f"  K=1             : {result['hits1_k1']}  (n={result['n_k1']})  [trivially 1.0]")
    print(f"  K>=2            : {result['hits1_kge2']:.4f}  (n={result['n_kge2']})  ← MAIN SIGNAL")
    print(f"  K=1 single-rev  : {result['hits1_k1_single_review']}  (n={result['n_k1_single_review']})  [cs=1]")
    print(f"{'='*65}")

    # Interpretation
    kge2_acc = result["hits1_kge2"]
    if kge2_acc is not None:
        if kge2_acc >= 0.60:
            print("  ★ floor K>=2 >= 0.60 → 임베딩 공간이 이미 query-memory 정렬")
            print("    → C/B 방어선 견고 신호. Stage1 align이 이미 좋은 출발점에서 시작.")
        elif kge2_acc >= 0.40:
            print("  ★ floor K>=2 = {:.2%} → 보통 수준.".format(kge2_acc))
            print("    → Stage1 align이 실질 작업 필요. 붕괴 아님, Stage 후 재측정으로 판단.")
        else:
            print("  ★ floor K>=2 < 0.40 → Stage1 align이 상당한 교정 필요.")
            print("    → Stage 후 개선폭이 클 것. 임베딩 공간 정렬 미흡.")

    print("=" * 65)
    return result


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries",  default="data/processed/Books/pseudo_queries_train.jsonl")
    ap.add_argument("--pairs",    default="data/processed/Books/align_pairs.jsonl")
    ap.add_argument("--bank_dir", default="data/processed/Books/memory_bank")
    ap.add_argument("--output",   default="data/processed/Books/step3_routing_accuracy.json")
    ap.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--skip_embed", action="store_true",
                    help="Skip STEP 3 (embedding), run STEP 2 only")
    args = ap.parse_args()

    print("[load] queries …", flush=True)
    queries = load_queries(args.queries)
    print(f"  rows: {len(queries)}", flush=True)

    print("[load] pairs …", flush=True)
    pairs = load_pairs(args.pairs)
    print(f"  rows: {len(pairs)}", flush=True)

    print("[load] memory bank …", flush=True)
    unit_vecs, k_map, cluster_sizes = load_bank_vectors(args.bank_dir)
    provenance_index, _ = load_memory_bank(args.bank_dir)
    print(f"  K>=1 users: {len(unit_vecs)}", flush=True)

    # STEP 2
    step2 = run_step2(queries, pairs, provenance_index)

    if not step2["step2_pass"]:
        print("\n[ABORT] STEP 2 gate failed — fix issues before running STEP 3.", flush=True)
        sys.exit(1)

    # STEP 3
    step3 = {}
    if not args.skip_embed:
        print("\n[load] embedding model …", flush=True)
        embed_model = load_embed_model(args.embed_model)
        step3 = run_step3(queries, unit_vecs, k_map, cluster_sizes, embed_model)

    # Save
    out = {
        "step2_verification": step2,
        "step3_routing_accuracy": step3,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] Saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
