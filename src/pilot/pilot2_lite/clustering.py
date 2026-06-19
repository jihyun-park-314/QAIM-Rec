"""Clustering-lite: agglomerative clustering on discriminative intent texts.

Embedding priority:
  1. sentence-transformers BAAI/bge-small-en-v1.5
     (downloaded to HF_HOME=/home/jhpark/QAIM-Rec/.cache/huggingface)
  2. sentence-transformers/all-MiniLM-L6-v2
  3. TF-IDF cosine fallback (numpy only)

Clustering text fields (preference_attrs and avoid excluded):
  contextual_intent, usage_context, taste_or_style_preference,
  selection_criteria, preference_tradeoff

Only is_discriminative==True interactions are clustering candidates.
Threshold sweep: [0.25, 0.35, 0.45].
"""
from __future__ import annotations

import os
import statistics
from typing import Any

import numpy as np


# ── Embedding ──────────────────────────────────────────────────────────────

_MODEL_NAMES = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]

_EMBEDDING_METHOD: str = "unknown"


def _try_load_sentence_transformer():
    """Try to load sentence-transformers model using project-local HF cache.
    Returns (model, model_name) or (None, None).
    """
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "huggingface"),
    )
    os.makedirs(cache_dir, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[clustering] sentence-transformers not installed, using TF-IDF fallback")
        return None, None

    for model_name in _MODEL_NAMES:
        try:
            print(f"[clustering] Loading {model_name} (cache: {cache_dir}) ...")
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
            )
            print(f"[clustering] Loaded: {model_name}")
            return model, model_name
        except Exception as e:
            print(f"[clustering] Failed to load {model_name}: {e}")

    return None, None


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    """Simple TF-IDF embedding with cosine normalisation (numpy only)."""
    vocab: dict[str, int] = {}
    tokenised = []
    for t in texts:
        toks = t.lower().split()
        tokenised.append(toks)
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    import math
    V = len(vocab)
    N = len(texts)
    mat = np.zeros((N, V), dtype=np.float32)
    for i, toks in enumerate(tokenised):
        for tok in toks:
            mat[i, vocab[tok]] += 1.0

    # IDF
    df = (mat > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1.0
    mat = mat * idf

    # L2 normalise
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


# ── Clustering text builder ────────────────────────────────────────────────

def _build_cluster_text(llm_output: dict) -> str:
    parts = []
    ci = llm_output.get("contextual_intent") or []
    if isinstance(ci, list):
        parts.extend([c for c in ci if c])
    else:
        if ci:
            parts.append(str(ci))

    aspect = llm_output.get("aspect_coverage") or {}
    for field in ("usage_context", "taste_or_style_preference", "preference_tradeoff"):
        val = aspect.get(field)
        if val:
            parts.append(str(val))

    sc = aspect.get("selection_criteria") or []
    if isinstance(sc, list):
        parts.extend([c for c in sc if c])
    elif sc:
        parts.append(str(sc))

    return " ".join(parts).strip()


# ── Agglomerative clustering (complete linkage, cosine distance) ───────────

def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance matrix. Embeddings should be L2-normalised."""
    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def _agglomerative_cluster(dist_matrix: np.ndarray, threshold: float) -> list[int]:
    """Complete-linkage agglomerative clustering using scipy if available,
    else a simple numpy fallback."""
    n = len(dist_matrix)
    if n == 0:
        return []
    if n == 1:
        return [0]

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist_matrix[i, j])
        condensed = np.array(condensed, dtype=np.float64)
        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=threshold, criterion="distance")
        return [int(l) - 1 for l in labels]
    except ImportError:
        pass

    # Numpy fallback: greedy single-pass clustering
    labels = [-1] * n
    next_label = 0
    cluster_members: list[list[int]] = []

    for i in range(n):
        best_cluster = -1
        best_max_dist = float("inf")
        for c_idx, members in enumerate(cluster_members):
            max_dist = max(dist_matrix[i, j] for j in members)
            if max_dist <= threshold and max_dist < best_max_dist:
                best_max_dist = max_dist
                best_cluster = c_idx
        if best_cluster == -1:
            labels[i] = next_label
            cluster_members.append([i])
            next_label += 1
        else:
            labels[i] = best_cluster
            cluster_members[best_cluster].append(i)

    return labels


# ── Main clustering function ───────────────────────────────────────────────

def run_clustering(
    smoke_results: list[dict],
    thresholds: list[float] = [0.25, 0.35, 0.45],
) -> dict:
    """Run clustering-lite on smoke results.

    Returns:
      {
        embedding_method: str,
        per_user: [{user_id, domain, cluster_results: {threshold: {k, labels, texts}}}],
        per_threshold_stats: {threshold: {avg_k, median_k, k_ge2_ratio, ...}},
      }
    """
    global _EMBEDDING_METHOD

    # Load embedding model once
    model, model_name = _try_load_sentence_transformer()
    if model is not None:
        _EMBEDDING_METHOD = f"sentence-transformers:{model_name}"
    else:
        _EMBEDDING_METHOD = "tfidf-cosine"
        print("[clustering] Using TF-IDF cosine fallback")

    per_user_results = []

    for user_data in smoke_results:
        uid = user_data["user_id"]
        domain = user_data["domain"]

        # Collect discriminative interactions
        disc_interactions = []
        for it in user_data.get("interactions", []):
            out = it.get("llm_output")
            if out and out.get("is_discriminative") and it.get("parse_success"):
                text = _build_cluster_text(out)
                if text.strip():
                    disc_interactions.append({
                        "idx": len(disc_interactions),
                        "cluster_text": text,
                        "parent_asin": it["parent_asin"],
                        "title": it["title"],
                        "llm_output": out,
                    })

        n_disc = len(disc_interactions)
        cluster_results: dict[str, Any] = {}

        if n_disc == 0:
            for t in thresholds:
                cluster_results[str(t)] = {"k_personal": 0, "labels": [], "texts": []}
        elif n_disc == 1:
            for t in thresholds:
                cluster_results[str(t)] = {
                    "k_personal": 1,
                    "labels": [0],
                    "texts": [disc_interactions[0]["cluster_text"]],
                }
        else:
            texts = [d["cluster_text"] for d in disc_interactions]

            # Get embeddings
            if model is not None:
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                embeddings = np.array(embeddings, dtype=np.float32)
            else:
                embeddings = _tfidf_embed(texts)

            dist_matrix = _cosine_distance_matrix(embeddings)

            for t in thresholds:
                labels = _agglomerative_cluster(dist_matrix, threshold=t)
                k = len(set(labels))
                cluster_results[str(t)] = {
                    "k_personal": k,
                    "labels": labels,
                    "texts": texts,
                }

        # Count aspect-valid interactions (for reporting)
        n_aspect_valid = sum(
            1 for it in user_data.get("interactions", [])
            if it.get("llm_output") and it["llm_output"].get("_aspect_coverage_valid")
        )

        per_user_results.append({
            "user_id": uid,
            "domain": domain,
            "total_eligible_train_count": user_data.get("total_eligible_train_count", 0),
            "n_interactions_processed": user_data.get("selected_interaction_count", 0),
            "n_discriminative": n_disc,
            "n_aspect_valid": n_aspect_valid,
            "cluster_results": cluster_results,
            "disc_interactions": disc_interactions,
        })

    return {
        "embedding_method": _EMBEDDING_METHOD,
        "per_user": per_user_results,
    }


def compute_threshold_stats(
    clustering_result: dict,
    eligibility_ratio: float,
    thresholds: list[float] = [0.25, 0.35, 0.45],
) -> dict:
    per_user = clustering_result["per_user"]
    stats = {}

    for t in thresholds:
        k_vals = [u["cluster_results"][str(t)]["k_personal"] for u in per_user]
        n = len(k_vals)
        if n == 0:
            continue
        k_ge2 = sum(1 for k in k_vals if k >= 2)
        k0 = sum(1 for k in k_vals if k == 0)
        k1 = sum(1 for k in k_vals if k == 1)

        avg_disc = statistics.mean(u["n_discriminative"] for u in per_user) if per_user else 0.0
        avg_aspect = statistics.mean(u["n_aspect_valid"] for u in per_user) if per_user else 0.0

        k_ge2_ratio = k_ge2 / n
        effective_coverage = eligibility_ratio * k_ge2_ratio

        stats[str(t)] = {
            "avg_k_personal": round(statistics.mean(k_vals), 2),
            "median_k_personal": round(statistics.median(k_vals), 2),
            "k_personal_ge2_ratio": round(k_ge2_ratio, 4),
            "k_personal_eq0_ratio": round(k0 / n, 4),
            "k_personal_eq1_ratio": round(k1 / n, 4),
            "fallback_needed_ratio": round((k0 + k1) / n, 4),
            "avg_discriminative_interactions_per_user": round(avg_disc, 2),
            "avg_aspect_valid_interactions_per_user": round(avg_aspect, 2),
            "effective_coverage": round(effective_coverage, 4),
            "n_sampled_users": n,
            "eligibility_ratio_used": round(eligibility_ratio, 4),
        }

    return stats


def build_representative_examples(
    clustering_result: dict,
    n_users: int = 3,
    threshold: float = 0.35,
) -> list[dict]:
    """Pick n_users with most discriminative interactions and show cluster examples."""
    per_user = clustering_result["per_user"]
    sorted_users = sorted(per_user, key=lambda u: -u["n_discriminative"])
    selected = sorted_users[:n_users]

    examples = []
    for u in selected:
        t_key = str(threshold)
        cr = u["cluster_results"].get(t_key, {})
        k = cr.get("k_personal", 0)
        labels = cr.get("labels", [])
        disc = u["disc_interactions"]

        # Group by cluster label
        clusters: dict[int, list[dict]] = {}
        for i, lbl in enumerate(labels):
            if i < len(disc):
                clusters.setdefault(lbl, []).append(disc[i])

        cluster_summaries = []
        for lbl, members in sorted(clusters.items()):
            # Representative: first member's contextual_intent[0]
            rep_intents = []
            item_titles = []
            avoid_notes = []
            for m in members:
                out = m["llm_output"]
                ci = out.get("contextual_intent") or []
                if ci:
                    rep_intents.append(ci[0])
                item_titles.append(m["title"])
                avoid = (out.get("preference_attrs") or {}).get("avoid") or []
                if avoid:
                    avoid_notes.extend(avoid)
            cluster_summaries.append({
                "cluster_id": lbl,
                "size": len(members),
                "representative_intents": rep_intents[:3],
                "item_titles": item_titles[:4],
                "avoid_ref": avoid_notes[:3],
            })

        examples.append({
            "user_id": u["user_id"],
            "domain": u["domain"],
            "train_eligible_interaction_count": u["total_eligible_train_count"],
            "discriminative_interaction_count": u["n_discriminative"],
            "k_personal_at_0_35": k,
            "clusters": cluster_summaries,
        })

    return examples
