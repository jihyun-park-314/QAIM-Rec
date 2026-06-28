"""Regenerate align_pairs.jsonl with v0.4.17 fixes (no LLM calls).

Changes vs previous:
  A. Add source_item_id (X) field — the train-history item that generated the query.
     Verified: X ∈ positive_memory cluster evidence.item_ids = 100% (provenance).
  B. Hard-neg redesign:
     K≥2 users → hard_negs = same-user other-cluster memories ONLY (intra-user primary).
                  Cross-user negatives handled implicitly by in-batch during InfoNCE.
     K=1 users  → no intra-user neg possible; hard_negs = cross-user sample (fallback).
     K=0 users  → prototype fallback users, skipped (no personal memories to contrast).

Input:  pseudo_queries_train.jsonl  (already has user_id, source_item_id, query, positive_memory_id)
        memory bank (per-user JSON files)
Output: align_pairs.jsonl (new), align_pairs_smoke.jsonl (first 100 rows)

Verification printed at the end:
  - X ∈ evidence rate (must be 100%)
  - intra-user hard-neg rate by K group
  - row count, user count
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders

def load_memory_bank(bank_dir: Path) -> tuple[dict, dict, dict]:
    """Load bank → (user_to_mems, user_to_k, mem_to_evidence).

    user_to_mems:  uid_str → [memory_id, ...]
    user_to_k:     uid_str → k_personal (int)
    mem_to_evidence: memory_id → frozenset of item_id (str)
    """
    user_to_mems: dict[str, list[str]] = {}
    user_to_k: dict[str, int] = {}
    mem_to_evidence: dict[str, frozenset] = {}

    for fpath in sorted(bank_dir.glob("[0-9]*.json")):
        data = json.loads(fpath.read_text(encoding="utf-8"))
        uid = str(data["user_id"])
        k = int(data.get("k_personal", 0))
        user_to_k[uid] = k
        mids = []
        for unit in data.get("units", []):
            mid = unit["memory_id"]
            ev_ids = frozenset(str(i) for i in unit.get("evidence", {}).get("item_ids", []))
            mem_to_evidence[mid] = ev_ids
            mids.append(mid)
        user_to_mems[uid] = mids

    return user_to_mems, user_to_k, mem_to_evidence


def _sample_cross_user(
    uid: str,
    user_to_mems: dict[str, list[str]],
    n: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    pool = [m for u, mids in user_to_mems.items() if u != uid for m in mids]
    return rng.sample(pool, min(n, len(pool))) if pool else []


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pseudo_queries",
                   default="data/processed/Books/pseudo_queries_train.jsonl")
    p.add_argument("--bank_dir",
                   default="data/processed/Books/memory_bank")
    p.add_argument("--out_pairs",
                   default="data/processed/Books/align_pairs.jsonl")
    p.add_argument("--cross_user_negs", type=int, default=10,
                   help="Cross-user negs for K=1 fallback only")
    p.add_argument("--smoke_rows", type=int, default=100,
                   help="First N rows → align_pairs_smoke.jsonl")
    args = p.parse_args()

    bank_dir = Path(args.bank_dir)
    out_pairs = Path(args.out_pairs)
    smoke_path = out_pairs.parent / (out_pairs.stem + "_smoke" + out_pairs.suffix)

    print(f"[regen] Loading memory bank from {bank_dir} …")
    user_to_mems, user_to_k, mem_to_evidence = load_memory_bank(bank_dir)
    print(f"[regen] {len(user_to_mems)} users  {sum(len(v) for v in user_to_mems.values())} memories")

    # Stats for verification
    n_written = 0
    n_skipped_null = 0
    n_skipped_no_pos = 0
    seen_users: set[str] = set()

    # Intra-user neg stats per K
    k_stats: dict[int, dict] = {}  # k → {pairs, intra_negs, cross_negs}

    # Verification: X ∈ evidence
    n_x_in_ev = 0
    n_x_total = 0

    with open(args.pseudo_queries, encoding="utf-8") as fin, \
         open(out_pairs, "w", encoding="utf-8") as fout, \
         open(smoke_path, "w", encoding="utf-8") as fsmoke:

        for line in fin:
            rec = json.loads(line)
            if rec.get("query") is None:
                n_skipped_null += 1
                continue

            uid = str(rec["user_id"])
            src_iid = str(rec["source_item_id"])
            pos_mid = rec["positive_memory_id"]
            query = rec["query"]

            if pos_mid is None:
                n_skipped_no_pos += 1
                continue

            k = user_to_k.get(uid, 0)
            mids = user_to_mems.get(uid, [])
            same_user_negs = [m for m in mids if m != pos_mid]

            # Modification B: intra-user primary for K≥2; cross only for K=1
            if k >= 2:
                hard_negs = same_user_negs  # intra-user only
            elif k == 1:
                seed = (int(uid) * 1_000_003 + int(src_iid)) & 0xFFFFFFFF
                hard_negs = _sample_cross_user(uid, user_to_mems, args.cross_user_negs, seed)
            else:
                # K=0 prototype users — no meaningful hard neg
                hard_negs = []

            # Modification A: add source_item_id
            row = {
                "query": query,
                "positive_memory_id": pos_mid,
                "source_item_id": int(src_iid),
                "hard_negative_memory_ids": hard_negs,
            }

            line_out = json.dumps(row, ensure_ascii=False) + "\n"
            fout.write(line_out)
            if n_written < args.smoke_rows:
                fsmoke.write(line_out)

            seen_users.add(uid)
            n_written += 1

            # Verification: X ∈ evidence
            n_x_total += 1
            if src_iid in mem_to_evidence.get(pos_mid, frozenset()):
                n_x_in_ev += 1

            # K-stats
            ks = k_stats.setdefault(k, {"pairs": 0, "intra_negs": 0, "cross_negs": 0})
            ks["pairs"] += 1
            if k >= 2:
                ks["intra_negs"] += len(hard_negs)
            elif k == 1:
                ks["cross_negs"] += len(hard_negs)

    # ---------------------------------------------------------------------------
    # Verification report

    print("\n" + "=" * 60)
    print("VERIFICATION REPORT — align_pairs v0.4.17")
    print("=" * 60)

    x_rate = n_x_in_ev / max(n_x_total, 1)
    gate_x = x_rate == 1.0
    print(f"\n[A] Target X ∈ positive cluster evidence.item_ids")
    print(f"    {n_x_in_ev}/{n_x_total} = {x_rate:.4f}  → {'PASS ✓' if gate_x else 'FAIL ✗ (must be 1.0)'}")

    print(f"\n[B] Intra-user hard-neg ratio by K group (K≥2 must be >> 8.4% baseline)")
    total_negs = 0
    total_intra = 0
    for k in sorted(k_stats):
        ks = k_stats[k]
        n = ks["pairs"]
        intra = ks["intra_negs"]
        cross = ks["cross_negs"]
        total_neg = intra + cross
        intra_rate = intra / max(total_neg, 1)
        if k >= 2:
            total_intra += intra
            total_negs += total_neg
        print(f"    K={k}: pairs={n}  intra={intra}  cross={cross}  "
              f"intra_rate={intra_rate:.3f}")
    if total_negs > 0:
        overall_kge2 = total_intra / total_negs
        gate_neg = overall_kge2 > 0.5  # should be near 1.0 for K≥2
        print(f"    K≥2 overall intra_rate: {overall_kge2:.4f}  → "
              f"{'PASS ✓' if gate_neg else 'WARN'} (baseline was 0.084)")

    print(f"\n[Summary]")
    print(f"    pairs written: {n_written}")
    print(f"    users covered: {len(seen_users)}")
    print(f"    skipped (null query): {n_skipped_null}")
    print(f"    skipped (no positive): {n_skipped_no_pos}")

    # Manifest
    md5 = _md5(out_pairs)
    manifest = {"path": str(out_pairs), "rows": n_written,
                "users": len(seen_users), "md5": md5,
                "version": "v0.4.17",
                "changes": ["source_item_id_added", "intra_user_hard_neg_primary"]}
    mpath = out_pairs.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\n[manifest] {mpath.name}: rows={n_written} md5={md5[:8]}…")

    smoke_manifest = {"path": str(smoke_path), "rows": min(n_written, args.smoke_rows),
                      "md5": _md5(smoke_path), "version": "v0.4.17"}
    (smoke_path.with_suffix(".manifest.json")).write_text(json.dumps(smoke_manifest, indent=2))

    print("=" * 60)
    print(f"Output: {out_pairs}")
    print(f"Smoke:  {smoke_path}")


if __name__ == "__main__":
    main()
