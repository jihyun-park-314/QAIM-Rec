"""STEP 0A — F3 Full Bank Build (LLM-free, deterministic).

v0.4.6: Reads p1_extractions.jsonl (single canonical per-review P1 artifact).
Clusters per user, builds evidence_map {cluster_label → {item_ids, timestamps, snippets}},
passes to make_intent_memory_unit → evidence.item_ids always non-empty.

Reads: data/processed/Books/p1_extractions.jsonl
Writes:
  data/processed/Books/memory_bank/{user_id}.json   (per-user bank)
  data/processed/Books/memory_bank/_prototypes.json
  reports/f3_full.json                              (build report)

Validations:
  1. Prototype P guard: tau_global fallback 0.35→0.30→0.25 to land P∈[8,15]
  2. Leakage detector: 2+ consecutive cap-word sequences in source_text
  3. K_personal distribution: K=0/1/≥2 + prototype-fallback vs true-cold split
  4. Splits coherence: bank users ⊂ splits.json train users
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.cluster import (
    _centroid,
    _keyword_union,
    _medoid_idx,
    _extract_pref_summary,
    _agglomerative_labels,
)
from src.memory.synth import make_memory_id, make_intent_memory_unit
from src.memory.prototypes import build_prototypes, save_prototypes
from src.memory.bank import assemble_bank, save_bank
from src.memory.embed import EmbeddingModel


# ---------------------------------------------------------------------------
# Config

P1_PATH = ROOT / "data/processed/Books/p1_extractions.jsonl"
SPLITS_PATH = ROOT / "data/processed/Books/splits.json"
CANDIDATES_PATH = ROOT / "data/processed/Books/candidates.jsonl"
BANK_DIR = ROOT / "data/processed/Books/memory_bank"
PROTO_PATH = BANK_DIR / "_prototypes.json"
REPORT_PATH = ROOT / "reports/f3_full.json"

TAU_PERSONAL = 0.30
K_MIN = 1
K_MAX = 5

TAU_GLOBAL_FALLBACKS = [0.35, 0.30, 0.25]
P_MIN = 8
P_MAX = 15

# Leakage: 2+ consecutive Title-Case words (proper noun phrase heuristic)
_CAP_PHRASE_RE = re.compile(r'(?:[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+)')
_COMMON_CAPS = frozenset([
    "The", "A", "An", "In", "Of", "For", "And", "Or", "But", "With",
    "At", "By", "From", "To", "On", "As", "Is", "Are", "Be", "Been",
    "Has", "Had", "Will", "Can", "The",
])


# ---------------------------------------------------------------------------
# Load per-review P1 records (v0.4.6 canonical artifact)

def load_p1_records(p1_path: str) -> list[dict]:
    """Load p1_extractions.jsonl — flat list of per-review P1 records."""
    records = []
    with open(p1_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n_users = len(set(r["user_id"] for r in records))
    print(f"[load] P1 records: {len(records)}, unique users: {n_users}")
    return records


# ---------------------------------------------------------------------------
# Cluster per-review P1 records → evidence_map

def build_cluster_data(p1_records: list[dict], emb_model: EmbeddingModel) -> list[dict]:
    """Group P1 records by user, cluster, build evidence_map with item_ids.

    evidence_map[cluster_label] = {item_ids, timestamps, snippets}
    — carried through to synthesize_cluster_data → make_intent_memory_unit.
    """
    user_groups: dict = defaultdict(list)
    for r in p1_records:
        user_groups[r["user_id"]].append(r)

    users = sorted(user_groups.keys(), key=str)
    results = []
    t0 = time.time()

    for idx, uid in enumerate(users):
        reviews = user_groups[uid]
        eligible = [r for r in reviews if r.get("eligible", False)]

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(users) - idx - 1)
            print(f"  [cluster] {idx+1}/{len(users)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        if not eligible:
            results.append({
                "user_id": uid, "k_personal": 0, "n_eligible": 0,
                "clusters": [], "tau_used": TAU_PERSONAL,
                "evidence_map": {},
                "_all_vecs": np.zeros((0, 768), dtype=np.float32),
                "_all_source_texts": [], "_all_intents": [],
            })
            continue

        source_texts = [r.get("source_text") or r.get("review_text", "") for r in eligible]
        intents = [r.get("contextual_intent", "") for r in eligible]
        item_ids = [r.get("item_id") for r in eligible]
        timestamps = [r.get("timestamp") for r in eligible]
        evidence_spans = [r.get("evidence_span", "") for r in eligible]

        vecs = emb_model.encode_corpus(source_texts)

        n = len(eligible)
        if n == 1:
            labels = [0]
        else:
            labels = _agglomerative_labels(vecs, TAU_PERSONAL, K_MIN, K_MAX)

        n_clusters = int(max(labels)) + 1 if labels else 0
        k_personal = min(n_clusters, K_MAX)

        enriched_clusters = []
        evidence_map: dict = {}

        for cl in range(n_clusters):
            member_indices = [i for i, lb in enumerate(labels) if lb == cl]
            if not member_indices:
                continue
            embs = vecs[member_indices]
            med_local = _medoid_idx(embs)
            med_global = member_indices[med_local]

            pref_summaries = [
                _extract_pref_summary(source_texts[i], intents[i])
                for i in member_indices
            ]

            enriched_clusters.append({
                "label": cl,
                "size": len(member_indices),
                "source_texts": [source_texts[i] for i in member_indices],
                "intents": [intents[i] for i in member_indices],
                "pref_summaries": pref_summaries,
                "embeddings": embs,
                "centroid": _centroid(embs),
                "medoid_local_idx": med_local,
                "medoid_intent": intents[med_global],
                "medoid_source_text": source_texts[med_global],
                "keyword_union": _keyword_union(pref_summaries),
            })

            evidence_map[cl] = {
                "item_ids": [item_ids[i] for i in member_indices],
                "timestamps": [timestamps[i] for i in member_indices],
                "snippets": [evidence_spans[i] for i in member_indices],
            }

        results.append({
            "user_id": uid,
            "k_personal": k_personal,
            "n_eligible": len(eligible),
            "clusters": enriched_clusters,
            "tau_used": TAU_PERSONAL,
            "evidence_map": evidence_map,
            "_all_vecs": vecs,
            "_all_source_texts": source_texts,
            "_all_intents": intents,
        })

    print(f"[cluster] Done. {len(results)} users enriched in {time.time()-t0:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Synthesize per user

def synthesize_cluster_data(cluster_data: list[dict]) -> dict:
    """Build {user_id: [IntentMemoryUnit]} from enriched cluster data.

    Reads evidence_map from user_data and passes item_ids/timestamps/snippets
    to make_intent_memory_unit — evidence.item_ids will be non-empty.
    """
    units_by_user: dict = {}
    for user_data in cluster_data:
        uid = user_data["user_id"]
        k = user_data["k_personal"]
        evidence_map = user_data.get("evidence_map", {})
        units = []
        for cluster in user_data["clusters"]:
            label = cluster["label"]
            mid = make_memory_id(uid, label, TAU_PERSONAL)
            ev = evidence_map.get(label, {})
            unit = make_intent_memory_unit(
                memory_id=mid,
                user_id=uid,
                cluster={**cluster, "_k_personal": k},
                tau=TAU_PERSONAL,
                is_prototype=False,
                evidence_item_ids=ev.get("item_ids"),
                evidence_timestamps=ev.get("timestamps"),
                evidence_snippets=ev.get("snippets"),
            )
            units.append(unit)
        units_by_user[uid] = units
    return units_by_user


# ---------------------------------------------------------------------------
# Prototype P guard with tau fallback

def build_prototypes_with_fallback(
    cluster_data: list[dict],
) -> tuple[list[dict], float, bool]:
    """Build prototypes with tau_global fallback to ensure P∈[P_MIN, P_MAX].

    Returns (prototypes, tau_used, p_min_met).
    """
    for tau in TAU_GLOBAL_FALLBACKS:
        protos = build_prototypes(cluster_data, tau_global=tau, p_min=P_MIN, p_max=P_MAX)
        p = len(protos)
        p_min_met = p >= P_MIN
        print(f"[proto] tau_global={tau:.2f}: P={p}  p_min_met={p_min_met}")
        if p_min_met:
            return protos, tau, True
        if tau == TAU_GLOBAL_FALLBACKS[-1]:
            # Exhausted fallbacks
            if p < P_MIN:
                print(f"[WARN] tau_global={tau:.2f} still P={p} < P_MIN={P_MIN}. Accepting anyway.")
            return protos, tau, p_min_met
    return [], TAU_GLOBAL_FALLBACKS[-1], False


# ---------------------------------------------------------------------------
# Leakage detector: 2+ consecutive Title-Case words

def _detect_leakage_in_source_text(source_text: str) -> list[str]:
    """Return list of suspicious proper-noun phrases found in source_text."""
    matches = _CAP_PHRASE_RE.findall(source_text)
    suspicious = []
    for m in matches:
        words = m.split()
        # Filter: at least one word that is NOT a common cap word
        non_common = [w for w in words if w not in _COMMON_CAPS]
        if len(non_common) >= 1:
            suspicious.append(m)
    return suspicious


def check_leakage_in_bank(
    units_by_user: dict,
    prototypes: list[dict],
    max_samples: int = 5,
) -> dict:
    """Scan all source_texts for potential leakage (proper noun phrases).

    Returns {
        'total_units': int,
        'violation_count': int,
        'violation_rate': float,
        'samples': [{user_id, source_text, phrases}],
        'spot_check': str,   # manual assessment note
    }
    """
    violations = []
    total = 0

    for uid, units in units_by_user.items():
        for unit in units:
            st = unit.get("embedding", {}).get("source_text", "")
            if not st:
                continue
            total += 1
            phrases = _detect_leakage_in_source_text(st)
            if phrases:
                violations.append({
                    "user_id": uid,
                    "source_text": st,
                    "phrases": phrases,
                    "is_prototype": False,
                })

    for proto in prototypes:
        st = proto.get("embedding", {}).get("source_text", "")
        if not st:
            continue
        total += 1
        phrases = _detect_leakage_in_source_text(st)
        if phrases:
            violations.append({
                "user_id": "GLOBAL",
                "source_text": st,
                "phrases": phrases,
                "is_prototype": True,
            })

    samples = violations[:max_samples]

    # Spot-check: does detector catch series/author names but not genre words?
    genre_words = ["alien", "human", "fantasy", "thriller", "romance", "mystery"]
    false_positive_risk = "LOW"
    for v in violations[:50]:
        for phrase in v["phrases"]:
            low = phrase.lower()
            if any(g in low for g in genre_words):
                false_positive_risk = "MEDIUM"
                break

    return {
        "total_units": total,
        "violation_count": len(violations),
        "violation_rate": round(len(violations) / total, 4) if total else 0.0,
        "samples": samples,
        "false_positive_risk": false_positive_risk,
    }


# ---------------------------------------------------------------------------
# K_personal distribution report

def report_k_distribution(
    cluster_data: list[dict],
    bank: dict,
    splits: dict,
) -> dict:
    """K_personal distribution for 8924 memory_full users."""
    k_counter = Counter(ud["k_personal"] for ud in cluster_data)
    k0 = k_counter.get(0, 0)
    k1 = k_counter.get(1, 0)
    k2plus = sum(v for ki, v in k_counter.items() if ki >= 2)
    total = len(cluster_data)

    # K=0 split: prototype-fallback vs true-cold ([DEFAULT_INTENT])
    k0_prototype = 0
    k0_default = 0
    for ud in cluster_data:
        if ud["k_personal"] == 0:
            uid_str = str(ud["user_id"])
            entry = bank["users"].get(uid_str, {})
            ft = entry.get("fallback_type", "")
            if ft == "prototype":
                k0_prototype += 1
            elif ft == "default":
                k0_default += 1

    mean_k = sum(ud["k_personal"] for ud in cluster_data) / total if total else 0

    return {
        "n_users_memory_full": total,
        "k_distribution": dict(sorted(k_counter.items())),
        "k0": k0, "k0_rate": round(k0 / total, 4),
        "k1": k1, "k1_rate": round(k1 / total, 4),
        "k2plus": k2plus, "k2plus_rate": round(k2plus / total, 4),
        "mean_k": round(mean_k, 3),
        "k0_prototype_fallback": k0_prototype,
        "k0_true_cold_default": k0_default,
    }


# ---------------------------------------------------------------------------
# Splits coherence

def check_splits_coherence(bank: dict, splits: dict) -> dict:
    splits_users = set(splits["users"].keys())
    bank_users = set(bank["users"].keys())
    missing_from_bank = splits_users - bank_users
    extra_in_bank = bank_users - splits_users
    return {
        "splits_users": len(splits_users),
        "bank_users": len(bank_users),
        "missing_from_bank": len(missing_from_bank),
        "extra_in_bank": len(extra_in_bank),
        "bank_subset_ok": len(extra_in_bank) == 0,
    }


# ---------------------------------------------------------------------------
# Timestamp leakage in evidence

def check_evidence_timestamps(bank: dict, splits: dict) -> dict:
    """Verify all evidence.timestamps ⊂ user's train history."""
    violations = 0
    n_checked = 0
    splits_users = splits["users"]

    for uid_str, entry in bank["users"].items():
        user_split = splits_users.get(uid_str)
        if user_split is None:
            continue
        train_items = set(user_split.get("train", []))
        for unit in entry.get("units", []):
            ts_list = unit.get("evidence", {}).get("timestamps", [])
            if not ts_list:
                continue
            n_checked += len(ts_list)
            for ts in ts_list:
                # timestamps in splits are item_ids (int), evidence timestamps = item_ids
                if ts not in train_items:
                    violations += 1

    return {
        "timestamps_checked": n_checked,
        "violations": violations,
        "ok": violations == 0,
    }


# ---------------------------------------------------------------------------
# Handle users in splits but not in memory_full (give DEFAULT_INTENT)

def add_missing_users(bank: dict, splits: dict) -> int:
    """Add DEFAULT_INTENT entries for splits users not in memory_full."""
    splits_users = splits["users"]
    bank_users = bank["users"]
    added = 0
    for uid_str in splits_users:
        if uid_str not in bank_users:
            bank_users[uid_str] = {
                "user_id": uid_str,
                "k_personal": 0,
                "units": [{
                    "memory_id": f"default_{uid_str}",
                    "user_id": uid_str,
                    "intent_description": "[DEFAULT_INTENT]",
                    "persona": {"tag": None, "description": ""},
                    "preference_signal": {
                        "attributes": {"feature_priorities": [], "avoid": None},
                        "summary": "",
                    },
                    "evidence": {"item_ids": [], "review_snippets": [], "timestamps": []},
                    "embedding": {
                        "vector": None,
                        "source_text": "",
                        "model_id": "BAAI/bge-base-en-v1.5",
                    },
                    "meta": {
                        "k_personal": 0, "cluster_size": 0, "tau": 0.0,
                        "is_prototype": False, "created_by": "f3_default",
                    },
                }],
                "fallback_type": "default",
                "leakage_ok": True,
            }
            added += 1
    return added


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    t_start = time.time()
    print("=" * 60)
    print("STEP 0A — F3 Full Bank Build")
    print("=" * 60)

    # 1. Load P1 per-review records (v0.4.6 canonical artifact)
    print("\n[1/6] Loading p1_extractions.jsonl ...")
    if not P1_PATH.exists():
        sys.exit(f"[ABORT] P1 canonical file not found: {P1_PATH}\n"
                 "Run parallel_memory_build.py first.")
    records = load_p1_records(str(P1_PATH))
    print(f"  Loaded {len(records)} records, {len(set(r['user_id'] for r in records))} unique users")

    # 2. Load bge-base (GPU available)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2/6] Loading bge-base-en-v1.5 on {device} ...")
    emb_model = EmbeddingModel(device=device)
    print(f"  Model loaded. dim={emb_model.dim}")

    # 3. Re-embed + enrich clusters
    print(f"\n[3/6] Re-embedding + enriching clusters for {len(records)} users ...")
    cluster_data = build_cluster_data(records, emb_model)

    # 4. Synthesize per-user IntentMemoryUnits
    print(f"\n[4/6] Synthesizing IntentMemoryUnits ...")
    units_by_user = synthesize_cluster_data(cluster_data)
    total_units = sum(len(v) for v in units_by_user.values())
    print(f"  {total_units} units for {len(units_by_user)} users")

    # 5. Build prototypes (with tau fallback)
    print(f"\n[5/6] Building global prototypes (tau fallback: {TAU_GLOBAL_FALLBACKS}) ...")
    prototypes, tau_global_used, p_min_met = build_prototypes_with_fallback(cluster_data)
    if not p_min_met:
        print(f"[WARN] P_MIN={P_MIN} not met at any tau. P={len(prototypes)}. Check manually.")

    # 6. Assemble bank
    print(f"\n[6/6] Assembling bank ...")
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)

    bank = assemble_bank(
        cluster_data=cluster_data,
        synth_units_by_user=units_by_user,
        prototypes=prototypes,
        splits_path=None,  # manual validation below
        emb_model=emb_model,
        candidates_path=str(CANDIDATES_PATH) if CANDIDATES_PATH.exists() else None,
    )

    # Add DEFAULT_INTENT for splits users not in memory_full
    n_added = add_missing_users(bank, splits)
    print(f"  Added DEFAULT_INTENT for {n_added} splits users not in memory_full")

    # === VALIDATIONS ===
    print("\n" + "=" * 60)
    print("VALIDATIONS")
    print("=" * 60)

    # V1: Prototype P guard
    print(f"\n[V1] Prototype P guard:")
    print(f"  tau_global_used={tau_global_used:.2f}, P={len(prototypes)}, "
          f"p_min_met={p_min_met}")
    if len(prototypes) < P_MIN:
        print(f"  [WARN] P={len(prototypes)} < P_MIN={P_MIN}!")

    # V2: Leakage detector
    print(f"\n[V2] Leakage detector (proper noun phrases in source_text) ...")
    leakage_report = check_leakage_in_bank(units_by_user, prototypes)
    print(f"  Total units checked: {leakage_report['total_units']}")
    print(f"  Violations: {leakage_report['violation_count']} "
          f"({leakage_report['violation_rate']*100:.1f}%)")
    print(f"  False-positive risk: {leakage_report['false_positive_risk']}")
    if leakage_report["samples"]:
        print(f"  Samples (up to 5):")
        for s in leakage_report["samples"]:
            print(f"    user={s['user_id']}  phrases={s['phrases'][:3]}")
            print(f"    source_text: {s['source_text'][:120]}")

    # V3: K_personal distribution
    print(f"\n[V3] K_personal distribution:")
    k_report = report_k_distribution(cluster_data, bank, splits)
    print(f"  n_users_memory_full: {k_report['n_users_memory_full']}")
    print(f"  K=0: {k_report['k0']} ({k_report['k0_rate']*100:.1f}%)  "
          f"[prototype_fallback={k_report['k0_prototype_fallback']}, "
          f"true_cold={k_report['k0_true_cold_default']}]")
    print(f"  K=1: {k_report['k1']} ({k_report['k1_rate']*100:.1f}%)")
    print(f"  K≥2: {k_report['k2plus']} ({k_report['k2plus_rate']*100:.1f}%)")
    print(f"  mean_K: {k_report['mean_k']}")
    print(f"  K distribution: {k_report['k_distribution']}")

    # V4: Splits coherence
    print(f"\n[V4] Splits coherence:")
    coh = check_splits_coherence(bank, splits)
    print(f"  splits_users={coh['splits_users']}, bank_users={coh['bank_users']}")
    print(f"  missing_from_bank={coh['missing_from_bank']}, "
          f"extra_in_bank={coh['extra_in_bank']}")
    print(f"  bank_subset_ok={coh['bank_subset_ok']}")

    # V5: Evidence timestamp check
    print(f"\n[V5] Evidence timestamp leakage:")
    ts_check = check_evidence_timestamps(bank, splits)
    print(f"  timestamps_checked={ts_check['timestamps_checked']}, "
          f"violations={ts_check['violations']}, ok={ts_check['ok']}")

    # === SAVE ===
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    save_bank(bank, BANK_DIR)
    save_prototypes(prototypes, PROTO_PATH)

    elapsed = time.time() - t_start
    report = {
        "build_time_s": round(elapsed, 1),
        "n_records_loaded": len(records),
        "n_unique_users_memory_full": len(set(r["user_id"] for r in records)),
        "n_bank_users_total": len(bank["users"]),
        "n_added_default_intent": n_added,
        "tau_personal": TAU_PERSONAL,
        "tau_global_used": tau_global_used,
        "p_prototypes": len(prototypes),
        "p_min_met": p_min_met,
        "k_distribution": k_report,
        "leakage": leakage_report,
        "splits_coherence": coh,
        "evidence_timestamp_check": ts_check,
        "bank_stats": bank["stats"],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] Report saved → {REPORT_PATH}")
    print(f"[done] Total time: {elapsed/60:.1f} min")
    print(f"[done] Bank: {len(bank['users'])} users → {BANK_DIR}")
    print(f"[done] Prototypes: {len(prototypes)} → {PROTO_PATH}")


if __name__ == "__main__":
    main()
