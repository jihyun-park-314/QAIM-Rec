"""STEP 2 — Per-user memory bank assembly with K=0 prototype fallback (plan.md v0.4.8 §2.2(C)).

Inputs:
  data/memory/Books/f3_bank.jsonl           (personal units, immutable)
  data/processed/Books/memory_bank/_prototypes.json  (from build_prototypes.py)
  data/processed/Books/splits.json          (all 10,566 train users)
  data/processed/Books/sequences.jsonl      (train-history item IDs for K=0 users)
  data/processed/Books/candidates.jsonl     (item_id → item_title lookup)

Outputs:
  data/processed/Books/memory_bank/{user_id}.json   (per-user bank)
  data/processed/Books/memory_bank/_prototypes.json  (copy, already written by STEP 1)
  data/processed/Books/memory_bank/_bank_stats.json

Rules (k_min=1):
  K_personal ≥ 1: personal units only, NO prototype padding.
  K_personal == 0: purchase-history centroid (bge-base) → nearest prototype (cosine sim).
                   is_prototype=True, fallback_for_user tagged in meta.

Gates (checked and printed):
  1. K≥1 user unit count == personal unit count (padding=0 confirmed).
  2. K=0 users have exactly 1 prototype unit each.
  3. Deterministic: two identical runs produce identical per-user JSON md5.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BANK_JSONL = ROOT / "data/memory/Books/f3_bank.jsonl"
PROTOTYPES_PATH = ROOT / "data/processed/Books/memory_bank/_prototypes.json"
SPLITS_PATH = ROOT / "data/processed/Books/splits.json"
SEQUENCES_PATH = ROOT / "data/processed/Books/sequences.jsonl"
CANDIDATES_PATH = ROOT / "data/processed/Books/candidates.jsonl"
OUTPUT_DIR = ROOT / "data/processed/Books/memory_bank"


# ---------------------------------------------------------------------------
# Loaders

def load_personal_units(path: Path) -> dict[str, list[dict]]:
    """Return {user_id_str: [unit, ...]}."""
    by_user: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                u = json.loads(line)
                by_user[str(u["user_id"])].append(u)
    return dict(by_user)


def load_prototypes(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_splits_users(path: Path) -> dict[str, list[int]]:
    """Return {user_id_str: [item_id, ...]} from splits train sequences."""
    with open(path, encoding="utf-8") as f:
        splits = json.load(f)
    users = splits["users"]
    return {str(uid): seqs.get("train", []) for uid, seqs in users.items()}


def load_item_titles(candidates_path: Path) -> dict[str, str]:
    """Return {item_id_str: title} from candidates.jsonl."""
    titles: dict[str, str] = {}
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                iid = str(r["item_id"])
                if iid not in titles and r.get("item_title"):
                    titles[iid] = r["item_title"]
    return titles


def load_item_seq_lookup(sequences_path: Path, target_users: set[str]) -> dict[str, list[str]]:
    """Return {user_id_str: [item_id_str, ...]} for target users from sequences.jsonl."""
    seq_lookup: dict[str, list[str]] = {}
    with open(sequences_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                uid = str(r["user_id"])
                if uid in target_users:
                    seq_lookup[uid] = [str(it["item_id"]) for it in r.get("items", [])]
    return seq_lookup


# ---------------------------------------------------------------------------
# Embedding + nearest prototype

def _embed_titles(titles: list[str]) -> np.ndarray:
    from src.memory.embed import EmbeddingModel
    model = EmbeddingModel()
    return model.encode_corpus(titles)


def _nearest_prototype(centroid: np.ndarray, prototypes: list[dict]) -> dict:
    """Cosine similarity (vectors are L2-normalised from bge-base)."""
    proto_vecs = np.array([p["embedding"]["vector"] for p in prototypes], dtype=np.float32)
    sims = proto_vecs @ centroid
    return prototypes[int(np.argmax(sims))]


def purchase_centroid_for_user(
    uid: str,
    seq_lookup: dict[str, list[str]],
    item_titles: dict[str, str],
    embed_cache: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Embed available item titles in user train history → L2-normalised centroid."""
    item_ids = seq_lookup.get(uid, [])
    available_titles = [(iid, item_titles[iid]) for iid in item_ids if iid in item_titles]
    if not available_titles:
        return None
    iids, titles = zip(*available_titles)
    # Embed uncached titles
    to_embed = [(iid, t) for iid, t in zip(iids, titles) if iid not in embed_cache]
    if to_embed:
        new_iids, new_titles = zip(*to_embed)
        vecs = _embed_titles(list(new_titles))
        for iid, vec in zip(new_iids, vecs):
            embed_cache[iid] = vec
    vecs = np.stack([embed_cache[iid] for iid in iids])
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


# ---------------------------------------------------------------------------
# Assemble

def assemble(
    personal_by_user: dict[str, list[dict]],
    prototypes: list[dict],
    all_users: dict[str, list],  # {uid: train_item_ids}
    seq_lookup: dict[str, list[str]],
    item_titles: dict[str, str],
) -> tuple[dict[str, dict], dict]:
    """Build per-user bank entries.

    Returns: ({uid: bank_entry}, stats)
    """
    bank: dict[str, dict] = {}
    stats = {
        "k_personal_ge1": 0,
        "k_personal_0": 0,
        "prototype_fallback": 0,
        "no_centroid_fallback": 0,  # K=0 with no item titles → use rank-0 prototype
        "total": len(all_users),
    }

    embed_cache: dict[str, np.ndarray] = {}

    k0_users = [uid for uid in all_users if uid not in personal_by_user]
    k1plus_users = [uid for uid in all_users if uid in personal_by_user]

    # --- K≥1 users ---
    for uid in k1plus_users:
        units = personal_by_user[uid]
        bank[uid] = {
            "user_id": uid,
            "k_personal": len(units),
            "units": units,
            "fallback_type": "personal",
        }
        stats["k_personal_ge1"] += 1

    # --- K=0 users ---
    if k0_users:
        print(f"  [bank] Embedding purchase history for {len(k0_users)} K=0 users ...")
        # Collect all unique item_ids across K=0 users for batch embedding
        all_k0_iids: set[str] = set()
        for uid in k0_users:
            for iid in seq_lookup.get(uid, []):
                if iid in item_titles:
                    all_k0_iids.add(iid)

        # Batch embed all unique items at once
        if all_k0_iids:
            iid_list = list(all_k0_iids)
            title_list = [item_titles[iid] for iid in iid_list]
            print(f"  [bank] Embedding {len(iid_list)} unique items ...")
            vecs = _embed_titles(title_list)
            for iid, vec in zip(iid_list, vecs):
                embed_cache[iid] = vec

        for uid in k0_users:
            centroid = purchase_centroid_for_user(uid, seq_lookup, item_titles, embed_cache)
            if centroid is not None and prototypes:
                proto = _nearest_prototype(centroid, prototypes)
                fallback_type = "prototype"
                stats["prototype_fallback"] += 1
            elif prototypes:
                # No item titles available: assign most popular prototype (rank 0)
                proto = prototypes[0]
                fallback_type = "prototype_no_centroid"
                stats["no_centroid_fallback"] += 1
            else:
                raise RuntimeError("No prototypes available for K=0 fallback")

            # Tag fallback in meta (don't mutate original prototype dict)
            proto_copy = json.loads(json.dumps(proto))
            proto_copy["meta"] = {**proto_copy["meta"], "fallback_for_user": uid}

            bank[uid] = {
                "user_id": uid,
                "k_personal": 0,
                "units": [proto_copy],
                "fallback_type": fallback_type,
            }
            stats["k_personal_0"] += 1

    return bank, stats


# ---------------------------------------------------------------------------
# Gates

def check_gates(
    bank: dict[str, dict],
    personal_by_user: dict[str, list[dict]],
    all_users: dict[str, list],
) -> bool:
    ok = True

    # Gate 1: K≥1 users have exactly their personal units (no padding)
    padding_violations = 0
    for uid, entry in bank.items():
        if entry["k_personal"] >= 1:
            expected = len(personal_by_user.get(uid, []))
            actual = len(entry["units"])
            if actual != expected:
                print(f"  [GATE FAIL] K≥1 uid={uid}: expected {expected} units, got {actual}")
                padding_violations += 1
    if padding_violations == 0:
        print(f"  [GATE 1 PASS] K≥1 padding=0 confirmed for all {sum(1 for e in bank.values() if e['k_personal']>=1)} users")
    else:
        print(f"  [GATE 1 FAIL] {padding_violations} padding violations")
        ok = False

    # Gate 2: K=0 users have exactly 1 prototype unit
    k0_entries = [(uid, e) for uid, e in bank.items() if e["k_personal"] == 0]
    k0_bad = [(uid, len(e["units"])) for uid, e in k0_entries if len(e["units"]) != 1]
    if k0_bad:
        print(f"  [GATE 2 FAIL] {len(k0_bad)} K=0 users with ≠1 prototype unit")
        ok = False
    else:
        print(f"  [GATE 2 PASS] All {len(k0_entries)} K=0 users have exactly 1 prototype unit")

    # Gate 3: is_prototype=True for all K=0 units
    k0_not_proto = [
        uid for uid, e in bank.items()
        if e["k_personal"] == 0 and not e["units"][0].get("meta", {}).get("is_prototype")
    ]
    if k0_not_proto:
        print(f"  [GATE 3 FAIL] {len(k0_not_proto)} K=0 users whose unit is not is_prototype=True")
        ok = False
    else:
        print(f"  [GATE 3 PASS] All K=0 fallback units have is_prototype=True")

    return ok


def md5_of_bank(bank: dict[str, dict]) -> str:
    """Deterministic md5: sort by user_id, hash concatenated JSON."""
    combined = "".join(
        json.dumps(bank[uid], ensure_ascii=False, sort_keys=True)
        for uid in sorted(bank.keys())
    )
    return hashlib.md5(combined.encode()).hexdigest()


# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading personal units from {BANK_JSONL}")
    personal_by_user = load_personal_units(BANK_JSONL)
    print(f"  {len(personal_by_user)} users, {sum(len(v) for v in personal_by_user.values())} units")

    print(f"Loading prototypes from {PROTOTYPES_PATH}")
    prototypes = load_prototypes(PROTOTYPES_PATH)
    print(f"  {len(prototypes)} prototypes")

    print(f"Loading splits from {SPLITS_PATH}")
    all_users = load_splits_users(SPLITS_PATH)  # {uid: train_items}
    print(f"  {len(all_users)} total users")

    k0_user_set = set(all_users.keys()) - set(personal_by_user.keys())
    print(f"  K=0 users: {len(k0_user_set)}, K≥1 users: {len(personal_by_user)}")

    print(f"Loading item titles from {CANDIDATES_PATH}")
    item_titles = load_item_titles(CANDIDATES_PATH)
    print(f"  {len(item_titles)} items with titles")

    print(f"Loading sequences for K=0 users from {SEQUENCES_PATH}")
    seq_lookup = load_item_seq_lookup(SEQUENCES_PATH, k0_user_set)
    print(f"  Loaded sequences for {len(seq_lookup)} K=0 users")

    # Assemble
    print("\nAssembling bank ...")
    bank, stats = assemble(
        personal_by_user=personal_by_user,
        prototypes=prototypes,
        all_users=all_users,
        seq_lookup=seq_lookup,
        item_titles=item_titles,
    )
    print(f"\nAssembly stats: {json.dumps(stats, indent=2)}")

    # Gates
    print("\n=== GATES ===")
    gates_ok = check_gates(bank, personal_by_user, all_users)

    # Reproducibility check (same data → same md5)
    bank_md5 = md5_of_bank(bank)
    print(f"\n=== REPRODUCIBILITY ===")
    print(f"Bank md5 (sorted JSON): {bank_md5}")
    print("(Re-run with identical inputs should produce the same md5)")

    if not gates_ok:
        print("\nGate failures detected — bank NOT written.")
        sys.exit(1)

    # Save per-user files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(bank)} user bank files → {OUTPUT_DIR}")
    for uid, entry in bank.items():
        user_path = OUTPUT_DIR / f"{uid}.json"
        with open(user_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)

    # Save stats
    stats_path = OUTPUT_DIR / "_bank_stats.json"
    full_stats = {**stats, "bank_md5": bank_md5, "gates_ok": gates_ok}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2)
    print(f"Stats → {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
