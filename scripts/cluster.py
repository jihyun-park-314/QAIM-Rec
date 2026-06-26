"""STEP 1 — Per-user clustering + provenance mapping (LLM-free).

Reads p1_extractions.jsonl (eligible==true), clusters per user with
agglomerative average-linkage (cosine via L2-norm+euclidean,
τ_personal=0.30, k_min=1, k_max=5), saves full item_id/timestamp
provenance per cluster.

Outputs:
  data/processed/Books/cluster_assignments.jsonl   one JSON line per user
  data/processed/Books/provenance_map.json         {uid: {label: {item_ids, timestamps}}}
  reports/cluster_report.json                      gate check report

Gate checks (all must PASS before returning exit 0):
  1. Mapping completeness: every eligible item_id appears in provenance_map,
     per-user count == eligible count.  Missing: 0.
  2. Determinism: cluster 2× from same embeddings → identical labels.
  3. K_personal distribution: 0/1/≥2 fracs, mean/p50/p90/max.
     Degenerate flag if all-K=1 (τ too large) or all-K=K_MAX (τ too small).
  4. Single-review K=1 count (eligible==1 users that became K=1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.cluster import (
    _agglomerative_labels,
    _centroid,
    _extract_pref_summary,
    _keyword_union,
    _medoid_idx,
)
from src.memory.embed import EmbeddingModel

# ---------------------------------------------------------------------------
# Config

P1_PATH = ROOT / "data/processed/Books/p1_extractions.jsonl"
OUT_PATH = ROOT / "data/processed/Books/cluster_assignments.jsonl"
PROV_PATH = ROOT / "data/processed/Books/provenance_map.json"
REPORT_PATH = ROOT / "reports/cluster_report.json"

TAU_PERSONAL = 0.30
K_MIN = 1
K_MAX = 5


# ---------------------------------------------------------------------------
# Load eligible records

def load_eligible(p1_path: Path) -> dict[str, list[dict]]:
    """Return {user_id_str: [records sorted by item_id]} for eligible only."""
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(p1_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("eligible"):
                groups[str(r["user_id"])].append(r)
    # Sort each user's records by item_id (determinism guarantee)
    for uid in groups:
        groups[uid].sort(key=lambda r: r["item_id"])
    return dict(groups)


# ---------------------------------------------------------------------------
# Cluster one user → returns (cluster_result, vecs) keeping vecs for gate-2

def cluster_user(
    records: list[dict],
    emb_model: EmbeddingModel,
    tau: float,
    k_min: int,
    k_max: int,
) -> tuple[dict, np.ndarray]:
    """Cluster one user's eligible records.  Returns (result_dict, vecs)."""
    source_texts = [r["source_text"] for r in records]
    intents = [r.get("contextual_intent", "") for r in records]
    item_ids = [r["item_id"] for r in records]
    timestamps = [r.get("timestamp") for r in records]
    evidence_spans = [r.get("evidence_span", []) for r in records]

    vecs = emb_model.encode_corpus(source_texts)  # [n, d] L2-normalized

    n = len(records)
    if n == 1:
        labels = [0]
    else:
        labels = _agglomerative_labels(vecs, tau, k_min, k_max)

    n_clusters = max(labels) + 1 if labels else 1
    k_personal = min(n_clusters, k_max)

    clusters = []
    for cl in range(n_clusters):
        member_idx = [i for i, lb in enumerate(labels) if lb == cl]
        if not member_idx:
            continue
        embs = vecs[member_idx]
        med_local = _medoid_idx(embs)
        med_global = member_idx[med_local]

        pref_summaries = [
            _extract_pref_summary(source_texts[i], intents[i])
            for i in member_idx
        ]

        clusters.append({
            "label": cl,
            "size": len(member_idx),
            # ★ provenance: all member item_ids and timestamps (not trimmed)
            "item_ids": [item_ids[i] for i in member_idx],
            "timestamps": [timestamps[i] for i in member_idx],
            "source_texts": [source_texts[i] for i in member_idx],
            "intents": [intents[i] for i in member_idx],
            "pref_summaries": pref_summaries,
            "review_snippets": [
                (evidence_spans[i] if isinstance(evidence_spans[i], str)
                 else (evidence_spans[i][0] if evidence_spans[i] else ""))
                for i in member_idx
            ],
            "centroid": _centroid(embs).tolist(),
            "medoid_intent": intents[med_global],
            "medoid_source_text": source_texts[med_global],
            "keyword_union": _keyword_union(pref_summaries),
        })

    return {
        "user_id": records[0]["user_id"],
        "k_personal": k_personal,
        "n_eligible": n,
        "tau_personal": tau,
        "labels": labels,   # kept for gate-2 determinism check (stripped before save)
        "clusters": clusters,
    }, vecs


# ---------------------------------------------------------------------------
# Gate checks

def gate_1_completeness(
    results: list[dict],
    user_eligible_counts: dict[str, int],
) -> dict:
    """Every eligible item_id must appear in exactly one cluster."""
    missing_users = []
    mismatch_users = []
    total_eligible = sum(user_eligible_counts.values())
    total_mapped = 0

    for res in results:
        uid = str(res["user_id"])
        expected = user_eligible_counts.get(uid, 0)
        mapped = sum(len(cl["item_ids"]) for cl in res["clusters"])
        total_mapped += mapped
        if mapped == 0:
            missing_users.append(uid)
        elif mapped != expected:
            mismatch_users.append({"uid": uid, "expected": expected, "got": mapped})

    ok = (len(missing_users) == 0 and len(mismatch_users) == 0
          and total_mapped == total_eligible)
    return {
        "total_eligible": total_eligible,
        "total_mapped": total_mapped,
        "missing_users": missing_users[:10],
        "mismatch_users": mismatch_users[:10],
        "pass": ok,
    }


def gate_2_determinism(
    results: list[dict],
    vecs_by_user: dict[str, np.ndarray],
    sample_n: int = 200,
) -> dict:
    """Re-cluster a sample of users and check labels match."""
    import random

    rng = random.Random(42)
    eligible_users = [r for r in results if r["k_personal"] >= 1]
    sample = rng.sample(eligible_users, min(sample_n, len(eligible_users)))

    mismatches = 0
    for res in sample:
        uid = str(res["user_id"])
        vecs = vecs_by_user[uid]
        original_labels = res["labels"]
        n = len(original_labels)
        if n == 1:
            new_labels = [0]
        else:
            new_labels = _agglomerative_labels(vecs, TAU_PERSONAL, K_MIN, K_MAX)
        if new_labels != original_labels:
            mismatches += 1

    ok = mismatches == 0
    return {
        "sample_n": len(sample),
        "mismatches": mismatches,
        "pass": ok,
        "note": ("" if ok else
                 f"{mismatches} label mismatches in {len(sample)} sampled users — "
                 "check item_id sort order and sklearn version"),
    }


def gate_3_k_distribution(results: list[dict]) -> dict:
    """K_personal distribution + degenerate check."""
    ks = [r["k_personal"] for r in results]
    n = len(ks)
    k0 = sum(1 for k in ks if k == 0)
    k1 = sum(1 for k in ks if k == 1)
    k2plus = sum(1 for k in ks if k >= 2)

    ks_sorted = sorted(ks)
    mean_k = sum(ks) / n if n else 0.0
    p50 = ks_sorted[n // 2] if n else 0
    p90 = ks_sorted[int(n * 0.9)] if n else 0
    max_k = max(ks) if ks else 0

    degenerate = ""
    non_zero = [k for k in ks if k > 0]
    if non_zero:
        if all(k == 1 for k in non_zero):
            degenerate = "ALL_K1: τ may be too large (0.30) — every multi-review user gets K=1"
        elif all(k == K_MAX for k in non_zero):
            degenerate = f"ALL_K{K_MAX}: τ may be too small (0.30) — every multi-review user gets K_MAX"

    return {
        "n_users": n,
        "k0": k0, "k0_rate": round(k0 / n, 4) if n else 0,
        "k1": k1, "k1_rate": round(k1 / n, 4) if n else 0,
        "k2plus": k2plus, "k2plus_rate": round(k2plus / n, 4) if n else 0,
        "mean_k": round(mean_k, 3),
        "p50_k": p50,
        "p90_k": p90,
        "max_k": max_k,
        "degenerate": degenerate,
        "pass": degenerate == "",
    }


def gate_4_single_review(results: list[dict]) -> dict:
    """Count users with n_eligible==1 that correctly got K=1."""
    single_review = [r for r in results if r["n_eligible"] == 1]
    all_k1 = all(r["k_personal"] == 1 for r in single_review)
    return {
        "single_review_users": len(single_review),
        "all_k1": all_k1,
        "pass": all_k1,
    }


# ---------------------------------------------------------------------------
# Save

def save_outputs(
    results: list[dict],
    out_path: Path,
    prov_path: Path,
) -> None:
    """Write cluster_assignments.jsonl + provenance_map.json."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provenance_map: dict = {}

    with open(out_path, "w", encoding="utf-8") as f:
        for res in results:
            uid_str = str(res["user_id"])
            prov_entry: dict = {}
            for cl in res["clusters"]:
                prov_entry[str(cl["label"])] = {
                    "item_ids": cl["item_ids"],
                    "timestamps": cl["timestamps"],
                }
            provenance_map[uid_str] = prov_entry

            # Strip labels (internal) before serialising
            row = {k: v for k, v in res.items() if k != "labels"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance_map, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[save] {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print(f"[save] {prov_path}  ({prov_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Main

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=TAU_PERSONAL)
    parser.add_argument("--k_min", type=int, default=K_MIN)
    parser.add_argument("--k_max", type=int, default=K_MAX)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    tau = args.tau
    k_min = args.k_min
    k_max = args.k_max

    t_start = time.time()
    print("=" * 60)
    print(f"STEP 1 — Clustering  τ={tau}  k_min={k_min}  k_max={k_max}")
    print("=" * 60)

    # Load
    print(f"\n[1/4] Loading eligible records from {P1_PATH} ...")
    user_groups = load_eligible(P1_PATH)
    n_users = len(user_groups)
    n_eligible = sum(len(v) for v in user_groups.values())
    user_eligible_counts = {uid: len(recs) for uid, recs in user_groups.items()}
    print(f"  {n_eligible} eligible records across {n_users} users")

    # Load embedding model
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/4] Loading bge-base-en-v1.5 on {device} ...")
    emb_model = EmbeddingModel(device=device)
    print(f"  dim={emb_model.dim}")

    # Cluster all users
    print(f"\n[3/4] Clustering {n_users} users ...")
    results: list[dict] = []
    vecs_by_user: dict[str, np.ndarray] = {}
    users_sorted = sorted(user_groups.keys(), key=str)

    t0 = time.time()
    for idx, uid in enumerate(users_sorted):
        records = user_groups[uid]
        res, vecs = cluster_user(records, emb_model, tau, k_min, k_max)
        results.append(res)
        vecs_by_user[uid] = vecs
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n_users - idx - 1)
            print(f"  [{idx+1}/{n_users}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    print(f"  Done in {time.time()-t0:.1f}s")

    # Save
    print(f"\n[4/4] Saving outputs ...")
    save_outputs(results, OUT_PATH, PROV_PATH)

    # Gate checks
    print("\n" + "=" * 60)
    print("GATE CHECKS")
    print("=" * 60)

    g1 = gate_1_completeness(results, user_eligible_counts)
    g2 = gate_2_determinism(results, vecs_by_user)
    g3 = gate_3_k_distribution(results)
    g4 = gate_4_single_review(results)

    print(f"\n[GATE 1] Mapping completeness:")
    print(f"  total_eligible={g1['total_eligible']}  total_mapped={g1['total_mapped']}")
    print(f"  missing_users={g1['missing_users']}  mismatch_users={g1['mismatch_users']}")
    print(f"  PASS={g1['pass']}")

    print(f"\n[GATE 2] Determinism (sample_n={g2['sample_n']}):")
    print(f"  mismatches={g2['mismatches']}  PASS={g2['pass']}")
    if g2["note"]:
        print(f"  NOTE: {g2['note']}")

    print(f"\n[GATE 3] K_personal distribution:")
    print(f"  K=0: {g3['k0']} ({g3['k0_rate']*100:.1f}%)")
    print(f"  K=1: {g3['k1']} ({g3['k1_rate']*100:.1f}%)")
    print(f"  K≥2: {g3['k2plus']} ({g3['k2plus_rate']*100:.1f}%)")
    print(f"  mean={g3['mean_k']}  p50={g3['p50_k']}  p90={g3['p90_k']}  max={g3['max_k']}")
    if g3["degenerate"]:
        print(f"  ★ DEGENERATE: {g3['degenerate']}")
    print(f"  PASS={g3['pass']}")

    print(f"\n[GATE 4] Single-review K=1:")
    print(f"  single_review_users={g4['single_review_users']}  all_k1={g4['all_k1']}")
    print(f"  PASS={g4['pass']}")

    elapsed = time.time() - t_start
    all_pass = g1["pass"] and g2["pass"] and g3["pass"] and g4["pass"]
    print(f"\n{'='*60}")
    print(f"ALL GATES PASS: {all_pass}  (elapsed {elapsed:.1f}s)")
    print(f"{'='*60}")

    # ΣK_personal for STEP3 cost estimate
    total_k = sum(r["k_personal"] for r in results)
    print(f"\n[STEP3 cost estimate]")
    print(f"  ΣK_personal = {total_k}")
    print(f"  LLM upper bound = 2 × {total_k} = {2*total_k} calls")
    print(f"  (1 intent+summary call + up to 1 persona call per cluster)")
    print(f"  NOTE: current p1 has no disposition_note → persona always skipped")
    print(f"  Actual LLM calls ≈ {total_k} (intent+summary only)")

    report = {
        "n_users": n_users,
        "n_eligible": n_eligible,
        "tau_personal": tau,
        "k_min": k_min,
        "k_max": k_max,
        "elapsed_s": round(elapsed, 1),
        "gates": {
            "g1_completeness": g1,
            "g2_determinism": g2,
            "g3_k_distribution": g3,
            "g4_single_review": g4,
        },
        "all_gates_pass": all_pass,
        "sum_k_personal": total_k,
        "llm_call_upper_bound": 2 * total_k,
        "llm_call_estimate": total_k,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[report] {REPORT_PATH}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
