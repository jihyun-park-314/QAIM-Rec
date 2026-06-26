"""F3 prototypes: global agglomerative on all users' discriminative embeddings.

All discriminative embeddings (from all users' clusters) are pooled →
global agglomerative clustering → top P clusters by size → is_prototype=True
IntentMemoryUnit (user_id="GLOBAL", timestamps=[]) → _prototypes.json.

P ∈ {8, 15} chosen to balance coverage vs. specificity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _make_proto_id(cluster_label: int, tau: float) -> str:
    tag = f"proto_{cluster_label}_{tau:.2f}"
    return "proto_" + hashlib.md5(tag.encode()).hexdigest()[:12]


def _agglomerative(
    vecs: np.ndarray, tau: float, k_max: int
) -> list[int]:
    """Hierarchical agglomerative clustering (average linkage).

    For large N (>5000): pairwise distances computed on GPU via torch.cdist,
    then passed to scipy.linkage on CPU. Falls back to pure sklearn if no GPU.
    """
    n = len(vecs)
    if n <= 1:
        return [0] * n

    tau_euc = (2.0 * tau) ** 0.5

    # GPU-hybrid path: torch.cdist for distance matrix, scipy for linkage
    if n > 5000:
        try:
            import torch
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[proto] GPU-hybrid agglomerative: N={n}, device={dev}")

            t = torch.tensor(vecs, dtype=torch.float32, device=dev)
            # Chunked distance to avoid OOM on very large N
            chunk = 4096
            rows = []
            for i in range(0, n, chunk):
                rows.append(torch.cdist(t[i:i+chunk], t, p=2).cpu())
            dist_mat = torch.cat(rows, dim=0).numpy()  # [N, N]
            condensed = squareform(dist_mat, checks=False)
            del dist_mat, t, rows

            Z = linkage(condensed, method="average")
            del condensed

            labels = fcluster(Z, t=tau_euc, criterion="distance").tolist()
            k = len(set(labels))
            if k > k_max:
                from scipy.cluster.hierarchy import cut_tree
                labels = cut_tree(Z, n_clusters=k_max).flatten().tolist()
            return labels

        except Exception as e:
            print(f"[proto] GPU-hybrid failed ({e}), falling back to sklearn")

    from sklearn.cluster import AgglomerativeClustering
    clf = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=tau_euc,
        metric="euclidean",
        linkage="average",
    )
    labels = clf.fit_predict(vecs)
    k = len(set(labels))
    if k > k_max:
        clf2 = AgglomerativeClustering(n_clusters=k_max, metric="euclidean", linkage="average")
        labels = clf2.fit_predict(vecs)
    return labels.tolist()


def _medoid_idx(vecs: np.ndarray) -> int:
    if len(vecs) == 1:
        return 0
    centroid = vecs.mean(axis=0)
    return int(np.argmin(np.linalg.norm(vecs - centroid, axis=1)))


def build_prototypes(
    cluster_data: list[dict],
    tau_global: float = 0.35,
    p_min: int = 8,
    p_max: int = 15,
) -> list[dict]:
    """Build global prototype IntentMemoryUnits.

    Args:
        cluster_data: output of cluster.load_cluster_data (all users)
        tau_global: distance threshold for global clustering (slightly looser than personal)
        p_min: minimum number of prototypes to keep
        p_max: maximum number of prototypes

    Returns:
        list of prototype IntentMemoryUnit dicts
    """
    from src.memory.cluster import _keyword_union, _extract_pref_summary

    # Pool all discriminative embeddings + their metadata
    all_vecs: list[np.ndarray] = []
    all_source_texts: list[str] = []
    all_intents: list[str] = []
    all_pref_summaries: list[str] = []

    for user_data in cluster_data:
        for cluster in user_data["clusters"]:
            for i, (st, ci) in enumerate(zip(cluster["source_texts"], cluster["intents"])):
                emb = cluster["embeddings"][i]
                all_vecs.append(emb)
                all_source_texts.append(st)
                all_intents.append(ci)
                all_pref_summaries.append(cluster["pref_summaries"][i])

    if not all_vecs:
        return []

    vecs = np.stack(all_vecs)  # [N, d]
    print(f"[prototypes] Pooled {len(vecs)} discriminative embeddings from {len(cluster_data)} users")

    # Global agglomerative
    k_target = min(p_max, max(p_min, len(vecs) // 20))  # heuristic
    labels = _agglomerative(vecs, tau_global, k_max=k_target)
    k_actual = len(set(labels))
    print(f"[prototypes] Global clustering: τ={tau_global}, K={k_actual}")

    # Count cluster sizes, keep top P by size
    from collections import Counter
    label_counts = Counter(labels)
    # Sort by size descending, take top p_max
    top_labels = [lbl for lbl, _ in label_counts.most_common(p_max)]
    # Ensure at least p_min
    if len(top_labels) < p_min:
        top_labels = [lbl for lbl, _ in label_counts.most_common()][:p_min]

    print(f"[prototypes] Top-P prototypes: {len(top_labels)} (sizes: "
          f"{[label_counts[l] for l in top_labels[:5]]}...)")

    prototypes = []
    for proto_idx, lbl in enumerate(top_labels):
        member_indices = [i for i, l in enumerate(labels) if l == lbl]
        member_vecs = vecs[member_indices]
        member_sts = [all_source_texts[i] for i in member_indices]
        member_cis = [all_intents[i] for i in member_indices]
        member_ps = [all_pref_summaries[i] for i in member_indices]

        med = _medoid_idx(member_vecs)
        centroid = member_vecs.mean(axis=0)
        kw_union = _keyword_union(member_ps, top_n=15)
        ci_med = member_cis[med]
        source_text = f"{ci_med} {' '.join(kw_union)}".strip()

        proto_id = _make_proto_id(proto_idx, tau_global)
        prototypes.append({
            "memory_id": proto_id,
            "user_id": "GLOBAL",
            "intent_description": ci_med,
            "persona": {"tag": None, "description": ""},
            "preference_signal": {
                "attributes": {
                    "feature_priorities": kw_union[:5],
                    "avoid": None,
                },
                "summary": " ".join(kw_union),
            },
            "evidence": {
                "item_ids": [],
                "review_snippets": [],
                "timestamps": [],
            },
            "embedding": {
                "vector": centroid.tolist(),
                "source_text": source_text,
                "model_id": "BAAI/bge-base-en-v1.5",
            },
            "meta": {
                "k_personal": -1,
                "cluster_size": len(member_indices),
                "tau": tau_global,
                "is_prototype": True,
                "proto_rank": proto_idx,
                "created_by": "f3_deterministic",
            },
        })

    return prototypes


def save_prototypes(prototypes: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prototypes, f, indent=2, ensure_ascii=False)
    print(f"[prototypes] Saved {len(prototypes)} prototypes → {output_path}")
