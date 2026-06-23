"""F3 cluster: read dump JSONL → re-embed source_texts → medoids/centroids.

Wraps the inline agglomerative clustering already in the dump.
Cluster assignments are taken as-is from the dump (τ=0.30 run).
Re-embedding on CPU recovers embeddings needed for medoid/centroid computation
and allows optional τ-sweep to compare K distributions across thresholds.

source_text = contextual_intent + " " + preference_summary  (no title/author)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_STOP_WORDS = frozenset([
    "a","an","the","and","or","for","with","in","on","at","to","of","by",
    "from","as","is","it","this","that","my","i","me","was","are","be",
    "been","have","has","had","will","would","can","could","should","very",
    "also","but","not","so","if","all","its","which","when","who","what",
    "their","they","them","these","those","than","then","just","more","some",
    "any","each","both","over","after","before","into","out","up","down",
    "about","through","between","during","without","against","within",
    "seeks","seek","prefers","prefer","values","value","wants","want",
])


# ---------------------------------------------------------------------------
# Utilities

def _extract_pref_summary(source_text: str, contextual_intent: str) -> str:
    """Extract preference_summary part from source_text.

    source_text = contextual_intent + " " + preference_summary
    """
    ci = contextual_intent.strip()
    st = source_text.strip()
    if st.startswith(ci):
        return st[len(ci):].strip()
    # Fallback: return everything after the first sentence-boundary word
    return st


def _keyword_union(pref_summaries: list[str], top_n: int = 12) -> list[str]:
    """Extract unique non-stopword tokens from preference summaries."""
    token_counts: dict[str, int] = {}
    for ps in pref_summaries:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", ps.lower())
        for t in tokens:
            if t not in _STOP_WORDS:
                token_counts[t] = token_counts.get(t, 0) + 1
    # Sort by frequency descending
    sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_tokens[:top_n]]


def _medoid_idx(embeddings: np.ndarray) -> int:
    """Index of the vector closest to the centroid (medoid)."""
    if len(embeddings) == 1:
        return 0
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    return int(np.argmin(dists))


def _centroid(embeddings: np.ndarray) -> np.ndarray:
    return embeddings.mean(axis=0)


# ---------------------------------------------------------------------------
# Τ-sweep re-clustering (optional, for analysis)

def _agglomerative_labels(
    vecs: np.ndarray, tau: float, k_min: int, k_max: int
) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = len(vecs)
    if n <= 1:
        return [0] * n

    tau_euc = (2.0 * tau) ** 0.5
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


# ---------------------------------------------------------------------------
# Per-cluster enrichment

def _enrich_cluster(
    cluster_raw: dict,
    all_source_texts: list[str],
    all_intents: list[str],
    vecs: np.ndarray,
    member_indices: list[int],
) -> dict:
    """Build enriched cluster dict from dump cluster + embeddings."""
    embs = vecs[member_indices]
    med_local = _medoid_idx(embs)
    med_global = member_indices[med_local]

    pref_summaries = [
        _extract_pref_summary(all_source_texts[i], all_intents[i])
        for i in member_indices
    ]

    return {
        "label": cluster_raw["label"],
        "size": len(member_indices),
        "source_texts": [all_source_texts[i] for i in member_indices],
        "intents": [all_intents[i] for i in member_indices],
        "pref_summaries": pref_summaries,
        "embeddings": embs,          # np.ndarray [size, d]
        "centroid": _centroid(embs), # np.ndarray [d]
        "medoid_local_idx": med_local,
        "medoid_intent": all_intents[med_global],
        "medoid_source_text": all_source_texts[med_global],
        "keyword_union": _keyword_union(pref_summaries),
    }


# ---------------------------------------------------------------------------
# Main entry: load dump and enrich

def load_cluster_data(
    dump_path: str | Path,
    emb_model,
    tau_sweep: list[float] | None = None,
    k_min: int = 1,
    k_max: int = 5,
) -> list[dict]:
    """Read dump JSONL, re-embed source_texts, return enriched per-user cluster data.

    Args:
        dump_path: path to memory_b_*.jsonl dump file
        emb_model: EmbeddingModel (should be device="cpu")
        tau_sweep: list of τ values to compare K distributions (default [0.25, 0.30, 0.35])
        k_min: minimum clusters (default 1)
        k_max: maximum clusters (default 5)

    Returns:
        list of per-user dicts with enriched cluster data
    """
    if tau_sweep is None:
        tau_sweep = [0.25, 0.30, 0.35]

    dump_path = Path(dump_path)
    records = []
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results = []
    for rec in records:
        uid = rec["user_id"]
        k_dump = rec["k_personal"]

        # Flatten source_texts and intents across all clusters (preserving order)
        all_source_texts: list[str] = []
        all_intents: list[str] = []
        cluster_member_ranges: list[tuple[int, int]] = []  # (start, end) in flat list

        for cs in rec["cluster_summaries"]:
            start = len(all_source_texts)
            all_source_texts.extend(cs["source_texts"])
            all_intents.extend(cs["intents"])
            cluster_member_ranges.append((start, len(all_source_texts)))

        if not all_source_texts:
            results.append({
                "user_id": uid,
                "k_personal": 0,
                "clusters": [],
                "tau_used": 0.30,
                "tau_sweep": {},
            })
            continue

        # Re-embed all eligible source_texts for this user (CPU)
        vecs = emb_model.encode_corpus(all_source_texts)  # [n, d] L2-normalized

        # Enrich clusters using dump's assignments
        enriched_clusters = []
        for ci, ((start, end), cs) in enumerate(
            zip(cluster_member_ranges, rec["cluster_summaries"])
        ):
            member_indices = list(range(start, end))
            enriched = _enrich_cluster(cs, all_source_texts, all_intents, vecs, member_indices)
            enriched_clusters.append(enriched)

        # τ-sweep: re-cluster to compare K distributions
        sweep_results: dict[float, int] = {}
        if len(all_source_texts) >= 2:
            for tau in tau_sweep:
                labels = _agglomerative_labels(vecs, tau, k_min, k_max)
                sweep_results[tau] = len(set(labels))
        else:
            for tau in tau_sweep:
                sweep_results[tau] = k_dump

        results.append({
            "user_id": uid,
            "k_personal": k_dump,  # from dump (τ=0.30)
            "n_eligible": len(all_source_texts),
            "clusters": enriched_clusters,
            "tau_used": 0.30,
            "tau_sweep": sweep_results,
            # Raw flattened arrays for downstream use
            "_all_vecs": vecs,
            "_all_source_texts": all_source_texts,
            "_all_intents": all_intents,
        })

    return results
