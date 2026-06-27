"""Decompose the routing floor 0.90: leak vs non-leak, Jaccard overlap, query length.

Answers: is the floor driven by identifier leakage or lexical copying,
or is it genuine intent-to-memory alignment?

Outputs: data/processed/Books/routing_floor_decomp.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loaders

def load_queries(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("query") is None:
                continue
            rows.append(r)
    return rows


def load_bank(bank_dir: str):
    """Returns:
        unit_vecs   {memory_id: np.ndarray(768,)}
        k_map       {user_id: int}  k_personal
        unit_snippets {memory_id: str}  concatenated review snippets
    """
    unit_vecs = {}
    k_map = {}
    unit_snippets = {}
    for p in sorted(Path(bank_dir).glob("[0-9]*.json")):
        with open(p) as f:
            d = json.load(f)
        uid = int(d["user_id"])
        kp = int(d.get("k_personal", 0))
        if kp < 1:
            continue
        k_map[uid] = kp
        for unit in d.get("units", []):
            mid = unit["memory_id"]
            vec = unit.get("embedding", {}).get("vector")
            if vec is not None:
                unit_vecs[mid] = np.array(vec, dtype=np.float32)
            snippets = unit.get("evidence", {}).get("review_snippets", [])
            unit_snippets[mid] = " ".join(snippets)
    return unit_vecs, k_map, unit_snippets


def load_user_memories(bank_dir: str) -> dict[int, list[str]]:
    um = {}
    for p in sorted(Path(bank_dir).glob("[0-9]*.json")):
        with open(p) as f:
            d = json.load(f)
        uid = int(d["user_id"])
        if int(d.get("k_personal", 0)) < 1:
            continue
        um[uid] = [u["memory_id"] for u in d.get("units", [])]
    return um


# ---------------------------------------------------------------------------
# Helpers

def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def cosine_top1(query_vec: np.ndarray, candidate_vecs: list[np.ndarray]) -> int:
    """Return index of top-1 cosine match."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    mat = np.stack(candidate_vecs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms
    sims = mat @ q
    return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Main

def main():
    bank_dir = "data/processed/Books/memory_bank"
    queries_path = "data/processed/Books/pseudo_queries_train.jsonl"
    out_path = "data/processed/Books/routing_floor_decomp.json"

    print("[load] queries …")
    queries = load_queries(queries_path)
    print(f"  non-null queries: {len(queries)}")

    print("[load] memory bank …")
    unit_vecs, k_map, unit_snippets = load_bank(bank_dir)
    user_memories = load_user_memories(bank_dir)
    print(f"  memory units with vectors: {len(unit_vecs)}")

    print("[embed] loading bge-base-en-v1.5 …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    texts = [r["query"] for r in queries]
    print(f"[embed] encoding {len(texts)} queries …")
    q_vecs = model.encode(texts, batch_size=256, show_progress_bar=True,
                          normalize_embeddings=True)
    print(f"  Done. Shape: {q_vecs.shape}")

    # ---------------------------------------------------------------------------
    # Per-query routing result
    records = []
    skipped = 0
    for i, row in enumerate(queries):
        uid = int(row["user_id"])
        pos_mid = row["positive_memory_id"]
        kp = k_map.get(uid, 0)
        mids = user_memories.get(uid, [])

        if kp < 2 or len(mids) < 2:
            # K=1: skip (trivially 1.0, not informative)
            continue

        if pos_mid not in unit_vecs:
            skipped += 1
            continue
        candidate_vecs = [unit_vecs[m] for m in mids if m in unit_vecs]
        candidate_mids = [m for m in mids if m in unit_vecs]
        if pos_mid not in candidate_mids:
            skipped += 1
            continue

        top_idx = cosine_top1(q_vecs[i], candidate_vecs)
        hit = candidate_mids[top_idx] == pos_mid

        query_text = row["query"]
        mem_text = unit_snippets.get(pos_mid, "")
        jac = jaccard(query_text, mem_text)
        wc = len(query_text.split())
        leak = bool(row.get("leakage_flagged", False))

        records.append({
            "hit": hit,
            "leak": leak,
            "jaccard": jac,
            "wc": wc,
        })

    print(f"[routing] K>=2 records: {len(records)}  skipped: {skipped}")

    # ---------------------------------------------------------------------------
    # Aggregations

    def acc(recs):
        if not recs:
            return None, 0
        return round(sum(r["hit"] for r in recs) / len(recs), 4), len(recs)

    # 1. Overall K>=2
    acc_all, n_all = acc(records)

    # 2. Leak vs non-leak
    leak_recs = [r for r in records if r["leak"]]
    nonleak_recs = [r for r in records if not r["leak"]]
    acc_leak, n_leak = acc(leak_recs)
    acc_nonleak, n_nonleak = acc(nonleak_recs)

    # 3. Jaccard quartiles
    jacs = sorted(r["jaccard"] for r in records)
    q25 = jacs[len(jacs) // 4]
    q50 = jacs[len(jacs) // 2]
    q75 = jacs[3 * len(jacs) // 4]
    jac_buckets = {
        f"Q1 (jac<={q25:.3f})": [r for r in records if r["jaccard"] <= q25],
        f"Q2 ({q25:.3f}<jac<={q50:.3f})": [r for r in records if q25 < r["jaccard"] <= q50],
        f"Q3 ({q50:.3f}<jac<={q75:.3f})": [r for r in records if q50 < r["jaccard"] <= q75],
        f"Q4 (jac>{q75:.3f})": [r for r in records if r["jaccard"] > q75],
    }
    jac_results = {k: {"acc": acc(v)[0], "n": acc(v)[1]} for k, v in jac_buckets.items()}

    # 4. Word count buckets
    wc_buckets = {
        "short (<=20w)": [r for r in records if r["wc"] <= 20],
        "medium (21-35w)": [r for r in records if 21 <= r["wc"] <= 35],
        "long (>35w)": [r for r in records if r["wc"] > 35],
    }
    wc_results = {k: {"acc": acc(v)[0], "n": acc(v)[1]} for k, v in wc_buckets.items()}

    # ---------------------------------------------------------------------------
    # Report
    print()
    print("=" * 65)
    print("  Routing Floor Decomposition — K>=2 only (K=1 excluded)")
    print("=" * 65)
    print(f"\n[1] Overall K>=2           : {acc_all:.4f}  (n={n_all})")
    print(f"\n[2] Leak vs Non-leak")
    print(f"  leak (leakage_flagged)   : {acc_leak:.4f}  (n={n_leak},  {n_leak/n_all:.1%} of K>=2)")
    print(f"  non-leak                 : {acc_nonleak:.4f}  (n={n_nonleak}, {n_nonleak/n_all:.1%} of K>=2)")
    delta = acc_leak - acc_nonleak if acc_leak and acc_nonleak else None
    print(f"  delta (leak - nonleak)   : {delta:+.4f}" if delta is not None else "")
    print(f"\n[3] Jaccard overlap (query ∩ memory source text)")
    for bname, res in jac_results.items():
        print(f"  {bname:<36}: {res['acc']:.4f}  (n={res['n']})")
    print(f"\n[4] Query length buckets")
    for bname, res in wc_results.items():
        print(f"  {bname:<24}: {res['acc']:.4f}  (n={res['n']})")

    result = {
        "K_ge2_overall": {"acc": acc_all, "n": n_all},
        "leak_vs_nonleak": {
            "leak": {"acc": acc_leak, "n": n_leak},
            "nonleak": {"acc": acc_nonleak, "n": n_nonleak},
            "delta": delta,
        },
        "jaccard_buckets": jac_results,
        "wordcount_buckets": wc_results,
        "jaccard_quartile_thresholds": {"q25": round(q25, 4), "q50": round(q50, 4), "q75": round(q75, 4)},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
