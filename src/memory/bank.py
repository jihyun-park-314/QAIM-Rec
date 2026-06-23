"""F3 bank: assemble per-user memory bank with k_min=1 logic.

Assembly rules (plan.md v0.4.3 §7 #1):
  K_personal >= 2: personal only
  K_personal == 1: personal only (k_min=1, NO prototype padding)
  K_personal == 0: purchase history centroid → nearest prototype + [DEFAULT_INTENT]

Leakage check: all evidence.timestamps ⊂ user's train-history timestamps.
User set must match splits.json train users.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Prototype lookup

def _nearest_prototype(
    centroid: np.ndarray,
    prototypes: list[dict],
) -> dict:
    """Return prototype with highest cosine similarity to centroid."""
    if not prototypes:
        raise ValueError("No prototypes available for fallback")
    proto_vecs = np.array([p["embedding"]["vector"] for p in prototypes])
    # Dot product (both are L2-normalized from bge-base)
    sims = proto_vecs @ centroid
    return prototypes[int(np.argmax(sims))]


def _purchase_centroid(
    user_id: Any,
    candidates_path: str | Path,
    emb_model,
) -> np.ndarray | None:
    """Compute centroid of purchase history embeddings for K=0 users.

    Uses all candidate records for the user (not just eligible ones).
    Embeds item_titles as corpus vectors.
    """
    import json
    candidates_path = Path(candidates_path)
    user_records = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["user_id"] == user_id:
                user_records.append(r)

    if not user_records:
        return None

    titles = [r.get("item_title", "") for r in user_records if r.get("item_title")]
    if not titles:
        return None

    vecs = emb_model.encode_corpus(titles)
    return vecs.mean(axis=0)


# ---------------------------------------------------------------------------
# Leakage check

def _check_evidence_timestamps(
    evidence_timestamps: list,
    train_timestamps: set,
) -> bool:
    """Return True (no leakage) if all evidence timestamps are in train history."""
    if not evidence_timestamps:
        return True
    return all(ts in train_timestamps for ts in evidence_timestamps)


# ---------------------------------------------------------------------------
# Assemble bank for one user

def assemble_user_bank(
    user_data: dict,
    personal_units: list[dict],
    prototypes: list[dict],
    emb_model=None,
    candidates_path: str | Path | None = None,
    train_timestamps: set | None = None,
) -> dict:
    """Build memory bank entry for one user.

    Returns:
        {
          "user_id": ...,
          "k_personal": int,
          "units": [IntentMemoryUnit, ...],
          "fallback_type": "personal" | "prototype" | "default",
          "leakage_ok": bool,
        }
    """
    uid = user_data["user_id"]
    k = user_data["k_personal"]

    if k >= 1:
        # K>=1: personal only (k_min=1 policy — no prototype padding)
        units = personal_units
        fallback_type = "personal"
    else:
        # K=0: nearest prototype fallback
        fallback_type = "prototype"
        centroid = None

        if emb_model is not None and candidates_path is not None:
            centroid = _purchase_centroid(uid, candidates_path, emb_model)

        if centroid is not None and prototypes:
            proto = _nearest_prototype(centroid, prototypes)
            # Mark as fallback in meta
            fallback_unit = {**proto, "meta": {**proto["meta"], "fallback_for_user": str(uid)}}
            units = [fallback_unit]
        elif prototypes:
            # No centroid available: use most popular prototype
            units = [prototypes[0]]
        else:
            # No prototypes: DEFAULT_INTENT placeholder
            units = [{
                "memory_id": f"default_{uid}",
                "user_id": str(uid),
                "intent_description": "[DEFAULT_INTENT]",
                "persona": {"tag": None, "description": ""},
                "preference_signal": {"attributes": {"feature_priorities": [], "avoid": None}, "summary": ""},
                "evidence": {"item_ids": [], "review_snippets": [], "timestamps": []},
                "embedding": {"vector": None, "source_text": "", "model_id": "BAAI/bge-base-en-v1.5"},
                "meta": {"k_personal": 0, "cluster_size": 0, "tau": 0.0, "is_prototype": False, "created_by": "f3_default"},
            }]
            fallback_type = "default"

    # Leakage check
    leakage_ok = True
    if train_timestamps is not None:
        for unit in units:
            ev_ts = unit.get("evidence", {}).get("timestamps", [])
            if not _check_evidence_timestamps(ev_ts, train_timestamps):
                leakage_ok = False
                break

    return {
        "user_id": uid,
        "k_personal": k,
        "units": units,
        "fallback_type": fallback_type,
        "leakage_ok": leakage_ok,
    }


# ---------------------------------------------------------------------------
# Assemble full bank

def assemble_bank(
    cluster_data: list[dict],
    synth_units_by_user: dict[Any, list[dict]],
    prototypes: list[dict],
    splits_path: str | Path | None = None,
    emb_model=None,
    candidates_path: str | Path | None = None,
) -> dict:
    """Assemble full memory bank for all users.

    Args:
        cluster_data: from cluster.load_cluster_data
        synth_units_by_user: {user_id: [IntentMemoryUnit, ...]} from synth.synthesize_user
        prototypes: from prototypes.build_prototypes
        splits_path: path to splits.json (for user set validation)
        emb_model: for K=0 purchase centroid (optional)
        candidates_path: for K=0 purchase centroid (optional)

    Returns:
        {
          "users": {user_id_str: bank_entry},
          "stats": {...},
        }
    """
    # Load train user set from splits if available
    train_user_set: set | None = None
    train_timestamps_by_user: dict[Any, set] = {}
    if splits_path and Path(splits_path).exists():
        with open(splits_path, encoding="utf-8") as f:
            splits = json.load(f)
        # splits.json format: {user_id_str: {"train": [...], "val": ..., "test": ...}}
        train_user_set = set(splits.keys())
        for uid_str, split in splits.items():
            ts_list = [item.get("timestamp") for item in split.get("train", []) if item.get("timestamp")]
            train_timestamps_by_user[uid_str] = set(ts_list)

    bank: dict[str, dict] = {}
    stats = {"k0": 0, "k1": 0, "k2plus": 0, "prototype_fallback": 0, "default": 0, "leakage_violations": 0}

    for user_data in cluster_data:
        uid = user_data["user_id"]
        uid_str = str(uid)

        personal_units = synth_units_by_user.get(uid, [])
        train_ts = train_timestamps_by_user.get(uid_str)

        entry = assemble_user_bank(
            user_data=user_data,
            personal_units=personal_units,
            prototypes=prototypes,
            emb_model=emb_model,
            candidates_path=candidates_path,
            train_timestamps=train_ts,
        )
        bank[uid_str] = entry

        k = user_data["k_personal"]
        if k == 0:
            stats["k0"] += 1
            if entry["fallback_type"] == "prototype":
                stats["prototype_fallback"] += 1
            elif entry["fallback_type"] == "default":
                stats["default"] += 1
        elif k == 1:
            stats["k1"] += 1
        else:
            stats["k2plus"] += 1

        if not entry["leakage_ok"]:
            stats["leakage_violations"] += 1

    # User set validation
    if train_user_set is not None:
        bank_users = set(bank.keys())
        missing_from_bank = train_user_set - bank_users
        extra_in_bank = bank_users - train_user_set
        stats["user_set_validation"] = {
            "train_users": len(train_user_set),
            "bank_users": len(bank_users),
            "missing_from_bank": len(missing_from_bank),
            "extra_in_bank": len(extra_in_bank),
        }

    return {"users": bank, "stats": stats}


def save_bank(bank: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-user files
    for uid_str, entry in bank["users"].items():
        user_path = output_dir / f"{uid_str}.json"
        with open(user_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)

    # Save summary stats
    stats_path = output_dir / "_bank_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(bank["stats"], f, indent=2, ensure_ascii=False)

    print(f"[bank] Saved {len(bank['users'])} user banks → {output_dir}")
    print(f"[bank] Stats: {bank['stats']}")
