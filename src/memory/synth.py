"""F3 synth: deterministic IntentMemoryUnit construction.

Per cluster:
  intent_description = medoid's contextual_intent
  preference_signal.summary = member preference keyword union
  embedding.source_text = medoid_ci + " " + keyword_union  (NO title/author)
  embedding.vector = centroid of member embeddings (NOT re-embed source_text)
  persona = {"tag": null, "description": ""}  (LLM-free, no disposition)

--llm-synth flag (default OFF): enable LLM-based synthesis.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# ---------------------------------------------------------------------------
# Schema

def make_memory_id(user_id: Any, cluster_label: int, tau: float) -> str:
    tag = f"f3_{user_id}_{cluster_label}_{tau:.2f}"
    return "mem_" + hashlib.md5(tag.encode()).hexdigest()[:12]


def make_intent_memory_unit(
    memory_id: str,
    user_id: Any,
    cluster: dict,
    tau: float,
    is_prototype: bool = False,
    evidence_item_ids: list | None = None,
    evidence_timestamps: list | None = None,
    evidence_snippets: list | None = None,
) -> dict:
    """Build an IntentMemoryUnit dict from an enriched cluster.

    Args:
        memory_id: unique id
        user_id: user identifier ("GLOBAL" for prototypes)
        cluster: enriched cluster dict from cluster.py
        tau: threshold used for clustering
        is_prototype: True for global prototypes
        evidence_item_ids: list of item_ids (optional, from candidates join)
        evidence_timestamps: list of timestamps (optional)
        evidence_snippets: list of review snippet strings (optional)
    """
    ci = cluster["medoid_intent"]
    kw_union = " ".join(cluster["keyword_union"])
    source_text = f"{ci} {kw_union}".strip() if kw_union else ci

    centroid = cluster["centroid"]

    return {
        "memory_id": memory_id,
        "user_id": str(user_id),
        "intent_description": ci,
        "persona": {
            "tag": None,
            "description": "",
        },
        "preference_signal": {
            "attributes": {
                "feature_priorities": cluster["keyword_union"][:5],
                "avoid": None,
            },
            "summary": kw_union,
        },
        "evidence": {
            "item_ids": evidence_item_ids or [],
            "review_snippets": evidence_snippets or [],
            "timestamps": evidence_timestamps or [],
        },
        "embedding": {
            "vector": centroid.tolist(),
            "source_text": source_text,
            "model_id": "BAAI/bge-base-en-v1.5",
        },
        "meta": {
            "k_personal": cluster.get("_k_personal", -1),
            "cluster_size": cluster["size"],
            "tau": tau,
            "is_prototype": is_prototype,
            "created_by": "f3_deterministic",
        },
    }


# ---------------------------------------------------------------------------
# Synthesize one user

def synthesize_user(
    user_data: dict,
    tau: float = 0.30,
    evidence_map: dict | None = None,
) -> list[dict]:
    """Build IntentMemoryUnit list for one user.

    Args:
        user_data: enriched user dict from cluster.load_cluster_data
        tau: cluster threshold to use (for memory_id)
        evidence_map: optional {cluster_label → {item_ids, timestamps, snippets}}

    Returns:
        list of IntentMemoryUnit dicts
    """
    uid = user_data["user_id"]
    k = user_data["k_personal"]
    units = []
    for cluster in user_data["clusters"]:
        label = cluster["label"]
        mid = make_memory_id(uid, label, tau)

        ev = (evidence_map or {}).get(label, {})
        unit = make_intent_memory_unit(
            memory_id=mid,
            user_id=uid,
            cluster={**cluster, "_k_personal": k},
            tau=tau,
            is_prototype=False,
            evidence_item_ids=ev.get("item_ids"),
            evidence_timestamps=ev.get("timestamps"),
            evidence_snippets=ev.get("snippets"),
        )
        units.append(unit)
    return units


# ---------------------------------------------------------------------------
# Synthesize all users

def synthesize_all(
    cluster_data: list[dict],
    tau: float = 0.30,
) -> list[dict]:
    """Synthesize IntentMemoryUnits for all users.

    Returns flat list of all units (across all users).
    """
    all_units = []
    for user_data in cluster_data:
        units = synthesize_user(user_data, tau=tau)
        all_units.extend(units)
    return all_units
