"""STEP 1 — Population-level prototype construction (plan.md v0.4.8 §2.2(B)).

Input  : data/memory/Books/f3_bank.jsonl   (15,880 personal IntentMemoryUnits)
Output : data/processed/Books/memory_bank/_prototypes.json

Algorithm:
  Pool all personal-unit embedding vectors → global agglomerative clustering.
  Try τ_global = 0.35 → 0.30 → 0.25 until K_actual ≥ p_min (=8).
  Take top P ≤ p_max (=15) clusters by size (합산 유저 수) → is_prototype=True units.
  memory_id = "PROTOTYPE::Books::p{idx}", user_id = "GLOBAL".
  LLM-free — all fields derived from existing embeddings/text.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BANK_PATH = ROOT / "data/memory/Books/f3_bank.jsonl"
OUTPUT_DIR = ROOT / "data/processed/Books/memory_bank"
OUTPUT_PATH = OUTPUT_DIR / "_prototypes.json"
CATEGORY = "Books"

P_MIN = 8
P_MAX = 15
TAU_FALLBACKS = [0.20, 0.15, 0.10]  # complete linkage: sim=0.80/0.85/0.90
EVIDENCE_SAMPLE = 12  # max item_ids/snippets per prototype


# ---------------------------------------------------------------------------

def load_bank(path: Path) -> list[dict]:
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    return units


def _agglomerative(vecs: np.ndarray, tau: float, k_max: int) -> list[int]:
    """Average-linkage agglomerative; mirrors src/memory/prototypes._agglomerative."""
    n = len(vecs)
    if n <= 1:
        return [0] * n

    tau_euc = (2.0 * tau) ** 0.5

    if n > 5000:
        try:
            import torch
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  [agg] GPU-hybrid N={n} device={dev}")
            t = torch.tensor(vecs, dtype=torch.float32, device=dev)
            chunk = 4096
            rows = []
            for i in range(0, n, chunk):
                rows.append(torch.cdist(t[i : i + chunk], t, p=2).cpu())
            dist_mat = torch.cat(rows, dim=0).numpy()
            condensed = squareform(dist_mat, checks=False)
            del dist_mat, t, rows
            Z = linkage(condensed, method="complete")
            del condensed
            labels = fcluster(Z, t=tau_euc, criterion="distance").tolist()
            k = len(set(labels))
            if k > k_max:
                from scipy.cluster.hierarchy import cut_tree
                labels = cut_tree(Z, n_clusters=k_max).flatten().tolist()
            return labels
        except Exception as e:
            print(f"  [agg] GPU-hybrid failed ({e}), falling back to sklearn")

    from sklearn.cluster import AgglomerativeClustering

    clf = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=tau_euc,
        metric="euclidean",
        linkage="complete",
    )
    labels = clf.fit_predict(vecs)
    k = len(set(labels))
    if k > k_max:
        clf2 = AgglomerativeClustering(
            n_clusters=k_max, metric="euclidean", linkage="complete"
        )
        labels = clf2.fit_predict(vecs)
    return labels.tolist()


def _medoid_idx(vecs: np.ndarray) -> int:
    if len(vecs) == 1:
        return 0
    centroid = vecs.mean(axis=0)
    return int(np.argmin(np.linalg.norm(vecs - centroid, axis=1)))


def build_prototypes_from_bank(
    units: list[dict],
    tau_global: float,
    p_min: int = P_MIN,
    p_max: int = P_MAX,
) -> tuple[list[dict], dict]:
    """Run global agglomerative on personal-unit embeddings, return (prototypes, stats)."""
    vecs = np.array([u["embedding"]["vector"] for u in units], dtype=np.float32)
    print(f"  [proto] N={len(vecs)} vectors, τ_global={tau_global}")

    labels = _agglomerative(vecs, tau_global, k_max=p_max * 4)
    k_actual = len(set(labels))
    print(f"  [proto] K_actual={k_actual}")

    if k_actual < p_min:
        return [], {"k_actual": k_actual, "p_min_met": False, "tau": tau_global}

    # Count unique users per cluster (cluster_size = # unique user_ids)
    cluster_users: dict[int, set] = defaultdict(set)
    for i, lbl in enumerate(labels):
        cluster_users[lbl].add(units[i]["user_id"])

    label_sizes = {lbl: len(users) for lbl, users in cluster_users.items()}
    top_labels = sorted(label_sizes, key=lambda l: -label_sizes[l])[:p_max]
    p_actual = len(top_labels)

    # Coverage: top-P units / total units
    top_set = set(top_labels)
    top_count = sum(1 for l in labels if l in top_set)
    coverage_pct = top_count / len(labels) * 100

    print(f"  [proto] P={p_actual}, coverage={coverage_pct:.1f}%")
    print(f"  [proto] cluster_sizes (top-5): {[label_sizes[l] for l in top_labels[:5]]}")

    prototypes: list[dict] = []
    for idx, lbl in enumerate(top_labels):
        member_indices = [i for i, l in enumerate(labels) if l == lbl]
        member_units = [units[i] for i in member_indices]
        member_vecs = vecs[member_indices]

        med = _medoid_idx(member_vecs)
        centroid = member_vecs.mean(axis=0)
        medoid_unit = member_units[med]

        # Collect evidence: sample item_ids + snippets from member units
        item_ids: list = []
        snippets: list = []
        seen_items: set = set()
        for mu in member_units:
            ev = mu.get("evidence", {})
            for iid, snip in zip(
                ev.get("item_ids", []), ev.get("review_snippets", [])
            ):
                if iid not in seen_items and len(item_ids) < EVIDENCE_SAMPLE:
                    item_ids.append(iid)
                    snippets.append(snip)
                    seen_items.add(iid)
            if len(item_ids) >= EVIDENCE_SAMPLE:
                break

        memory_id = f"PROTOTYPE::{CATEGORY}::p{idx}"
        prototypes.append(
            {
                "memory_id": memory_id,
                "user_id": "GLOBAL",
                "intent_description": medoid_unit["intent_description"],
                "persona": {"tag": None, "description": ""},
                "preference_signal": medoid_unit["preference_signal"],
                "evidence": {
                    "item_ids": item_ids,
                    "review_snippets": snippets,
                    "timestamps": [],
                },
                "embedding": {
                    "vector": centroid.tolist(),
                    "source_text": medoid_unit["embedding"]["source_text"],
                    "model_id": "BAAI/bge-base-en-v1.5",
                },
                "meta": {
                    "k_personal": -1,
                    "cluster_size": label_sizes[lbl],
                    "tau": tau_global,
                    "is_prototype": True,
                    "proto_rank": idx,
                    "created_by": "f3_deterministic",
                },
            }
        )

    stats = {
        "tau": tau_global,
        "k_actual": k_actual,
        "p_actual": p_actual,
        "p_min_met": p_actual >= p_min,
        "coverage_pct": round(coverage_pct, 2),
        "cluster_sizes": [label_sizes[l] for l in top_labels],
        "total_units": len(units),
    }
    return prototypes, stats


def main() -> None:
    print(f"Loading {BANK_PATH}")
    units = load_bank(BANK_PATH)
    print(f"Loaded {len(units)} personal units")

    # τ fallback loop
    prototypes: list[dict] = []
    final_stats: dict = {}
    tau_used: float = -1.0

    for tau in TAU_FALLBACKS:
        print(f"\nTrying τ_global={tau} ...")
        protos, stats = build_prototypes_from_bank(units, tau)
        print(f"  stats: {stats}")
        if stats["p_min_met"]:
            prototypes = protos
            final_stats = stats
            tau_used = tau
            break
        else:
            print(f"  P={stats['k_actual']} < p_min={P_MIN}, trying next τ ...")

    if not prototypes:
        print(
            f"\nFAIL: P<{P_MIN} at all τ fallbacks. Final K={final_stats.get('k_actual')}."
        )
        print("Stopping per gate rule — do not force-assign P.")
        sys.exit(1)

    # Gate report
    p_actual = final_stats["p_actual"]
    print(f"\n=== GATE ===")
    print(f"P={p_actual} ({'≥' if p_actual >= P_MIN else '<'}{P_MIN}) — {'PASS' if p_actual >= P_MIN else 'FAIL'}")
    print(f"τ_global used: {tau_used} (fallback step: {TAU_FALLBACKS.index(tau_used)+1}/{len(TAU_FALLBACKS)})")
    print(f"Coverage of top-{p_actual}: {final_stats['coverage_pct']}% of {final_stats['total_units']} units")
    print("Cluster sizes:", final_stats["cluster_sizes"])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(prototypes)} prototypes → {OUTPUT_PATH}")

    # MD5 for reproducibility check
    with open(OUTPUT_PATH, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    print(f"_prototypes.json md5: {md5}")

    # Save stats alongside
    stats_path = OUTPUT_DIR / "_prototypes_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2)
    print(f"Stats → {stats_path}")


if __name__ == "__main__":
    main()
