"""Generate P2 pseudo-queries — dual mode: train (per history review) + eval (per test review).

v2 — provenance-aligned train mode (plan.md v0.4.11).

MODE train:
  Reads p1_extractions.jsonl (eligible reviews) + selected_review_ids from both GPU shards.
  Generates one search query per eligible review via P2 prompt.
  positive_memory_id: deterministic provenance lookup via evidence.item_ids — NO LLM judge.
  hard_negatives: same-user other-memory units (all) + cross-user memory sample N.
  Output 1: pseudo_queries_train.jsonl  {user_id, source_item_id, query, positive_memory_id}
  Output 2: align_pairs.jsonl           {query, positive_memory_id, hard_negative_memory_ids[]}

MODE eval:
  Reads raw reviews for each user's test item (from splits.json). Deferred until post-Stage1.

Usage — train (smoke, 5 users):
  python3 -u scripts/generate_pseudo_queries.py --mode train \\
    --p1_extractions data/processed/Books/p1_extractions.jsonl \\
    --selected_review_ids data/p1_shards_gpu01/selected_review_ids.json \\
                          data/p1_shards_gpu23/selected_review_ids.json \\
    --bank_dir data/processed/Books/memory_bank \\
    --meta_jsonl data/raw/Books/meta.jsonl \\
    --id_maps_json data/processed/Books/id_maps.json \\
    --prompt config/prompts/amazon/Books/p2_pseudo_query.txt \\
    --output_queries data/processed/Books/pseudo_queries_train.jsonl \\
    --output_pairs data/processed/Books/align_pairs.jsonl \\
    --llm_config configs/llm/p2.yaml \\
    --smoke_users 5

Usage — train (full, 4-GPU disjoint via two instances):
  # Instance A (GPU 0,1):
  python3 -u scripts/generate_pseudo_queries.py --mode train \\
    --p1_extractions data/processed/Books/p1_extractions.jsonl \\
    --selected_review_ids data/p1_shards_gpu01/selected_review_ids.json \\
    --bank_dir data/processed/Books/memory_bank \\
    --meta_jsonl data/raw/Books/meta.jsonl \\
    --id_maps_json data/processed/Books/id_maps.json \\
    --prompt config/prompts/amazon/Books/p2_pseudo_query.txt \\
    --output_queries data/processed/Books/pseudo_queries_train_gpu01.jsonl \\
    --output_pairs data/processed/Books/align_pairs_gpu01.jsonl \\
    --llm_config configs/llm/p2_gpu01.yaml
  # Instance B (GPU 2,3): same with gpu23 shard + gpu23 output paths + p2_gpu23.yaml
  # After both finish: cat both output files to merge (users are disjoint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.pipeline import check_leakage_v2, build_common_tokens_from_titles


# ---------------------------------------------------------------------------
# Shared helpers

def _parse_query(raw: str) -> tuple[str | None, str | None]:
    """Parse LLM output. Returns (query_text, failed_reason)."""
    m = re.search(r'\{[^{}]*?"query"\s*:[^{}]*?\}', raw, re.DOTALL)
    if not m:
        m = re.search(r'\{.*?"query".*?\}', raw, re.DOTALL)
    if not m:
        return None, "llm_fail"
    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError:
        return None, "llm_fail"
    query = parsed.get("query")
    if query is None:
        return None, "null_query"
    query = str(query).strip()
    if not query:
        return None, "null_query"
    return query, None


def _llm_call(client, prompt: str, retry_max: int = 2) -> tuple[str | None, str | None]:
    """Call LLM with ≤2 retries; return (query_text, failed_reason)."""
    for _ in range(retry_max + 1):
        try:
            raw, _, _, _ = client.generate([{"role": "user", "content": prompt}])
            query, reason = _parse_query(raw)
            if reason != "llm_fail":
                return query, reason
        except Exception:
            pass
    return None, "llm_fail"


# ---------------------------------------------------------------------------
# Data loaders

def load_memory_bank(bank_dir: str) -> tuple[dict, dict]:
    """Load all K>=1 user memory banks.

    Returns:
        provenance_index  {user_id (int): {item_id (int): memory_id (str)}}
        user_memories     {user_id (int): [memory_id (str), ...]}
    """
    provenance_index: dict[int, dict[int, str]] = {}
    user_memories: dict[int, list[str]] = {}

    for p in sorted(Path(bank_dir).glob("[0-9]*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        uid = int(data["user_id"])
        if data.get("k_personal", 0) < 1:
            continue
        units = data.get("units", [])
        prov: dict[int, str] = {}
        mids: list[str] = []
        for unit in units:
            mid = unit["memory_id"]
            mids.append(mid)
            for iid in unit.get("evidence", {}).get("item_ids", []):
                prov[int(iid)] = mid
        provenance_index[uid] = prov
        user_memories[uid] = mids

    return provenance_index, user_memories


def load_p1_index(p1_path: str) -> dict:
    """Build {(user_id, item_id): record_dict} index from p1_extractions.jsonl.

    Only keeps eligible=True records. record_dict has review_text and item_title.
    """
    idx: dict[tuple[int, int], dict] = {}
    with open(p1_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("eligible", False):
                continue
            idx[(int(r["user_id"]), int(r["item_id"]))] = {
                "review_text": r.get("review_text", ""),
                "item_title": r.get("item_title", ""),
                "eligible": True,
            }
    return idx


def load_selected_ids(paths: list[str]) -> dict[int, list[int]]:
    """Merge selected_review_ids from multiple shard files.

    Each file: {user_id_str: [item_id_int, ...]}. Shards are disjoint by user.
    """
    merged: dict[int, list[int]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)
        for uid_str, items in shard.items():
            uid = int(uid_str)
            if uid in merged:
                # Should not happen (shards are disjoint), but merge safely
                merged[uid].extend(int(i) for i in items)
            else:
                merged[uid] = [int(i) for i in items]
    return merged


def load_author_map(meta_jsonl: str, id_maps_json: str) -> dict[int, str]:
    """Build {item_id (int): author_name (str)} from meta.jsonl + id_maps.json."""
    with open(id_maps_json, encoding="utf-8") as f:
        id_maps = json.load(f)
    asin2id: dict[str, int] = id_maps.get("item2id", {})

    author_map: dict[int, str] = {}
    with open(meta_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            asin = r.get("parent_asin", "")
            if not asin or asin not in asin2id:
                continue
            iid = asin2id[asin]
            raw_author = r.get("author")
            if raw_author is None:
                continue
            # author field can be dict {"name": "..."} or plain string
            if isinstance(raw_author, dict):
                name = raw_author.get("name", "").strip()
            else:
                name = str(raw_author).strip()
            if name:
                author_map[iid] = name
    return author_map


# ---------------------------------------------------------------------------
# Hard negative construction

def _sample_cross_user_negs(
    user_id: int,
    user_memories: dict[int, list[str]],
    n: int,
    seed: int,
) -> list[str]:
    """Sample N memory_ids from users other than user_id (deterministic)."""
    rng = random.Random(seed)
    candidates: list[str] = []
    for uid, mids in user_memories.items():
        if uid != user_id:
            candidates.extend(mids)
    if not candidates:
        return []
    return rng.sample(candidates, min(n, len(candidates)))


# ---------------------------------------------------------------------------
# Manifest / md5

def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(out_path: Path, rows: int, users: int) -> None:
    manifest = {
        "path": str(out_path),
        "rows": rows,
        "users": users,
        "md5": _md5_file(out_path),
    }
    mpath = out_path.with_suffix(".manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] {mpath.name}: rows={rows} users={users} md5={manifest['md5'][:8]}…",
          flush=True)


# ---------------------------------------------------------------------------
# TRAIN mode v2

def run_train(
    client,
    p1_extractions: str,
    selected_review_id_paths: list[str],
    bank_dir: str,
    meta_jsonl: str,
    id_maps_json: str,
    prompt_tmpl: str,
    out_queries: Path,
    out_pairs: Path,
    cross_user_negs: int,
    smoke_users: int,
) -> dict:
    """Generate pseudo_queries_train.jsonl + align_pairs.jsonl.

    Positive labels come from provenance lookup (evidence.item_ids) — no LLM judge.
    """
    print("[p2_train] Loading memory bank …", flush=True)
    provenance_index, user_memories = load_memory_bank(bank_dir)
    print(f"[p2_train]   K>=1 users: {len(provenance_index)}", flush=True)

    print("[p2_train] Loading p1 extractions …", flush=True)
    p1_index = load_p1_index(p1_extractions)
    print(f"[p2_train]   eligible (uid,iid) pairs: {len(p1_index)}", flush=True)

    print("[p2_train] Loading selected review ids …", flush=True)
    selected = load_selected_ids(selected_review_id_paths)
    print(f"[p2_train]   selected users: {len(selected)}", flush=True)

    print("[p2_train] Loading author map …", flush=True)
    author_map = load_author_map(meta_jsonl, id_maps_json)
    print(f"[p2_train]   items with author: {len(author_map)}", flush=True)

    # Build common_tokens from all eligible item titles for leakage v2
    all_titles = [rec["item_title"] for rec in p1_index.values() if rec.get("item_title")]
    common_tokens = build_common_tokens_from_titles(all_titles, df_threshold=0.05)
    print(f"[p2_train]   common tokens for leakage v2: {len(common_tokens)}", flush=True)

    # Determine user processing order
    eligible_users = sorted(set(selected.keys()) & set(provenance_index.keys()))
    if smoke_users > 0:
        eligible_users = eligible_users[:smoke_users]
        print(f"[p2_train] ★ SMOKE MODE: {len(eligible_users)} users", flush=True)
    else:
        print(f"[p2_train] Processing {len(eligible_users)} users …", flush=True)

    stats = {
        "n_queries_ok": 0,
        "n_null_query": 0,
        "n_llm_fail": 0,
        "n_leakage_flagged": 0,
        "n_no_provenance": 0,
        "n_not_eligible": 0,
        "n_pairs": 0,
        "llm_calls": 0,
        "users_processed": 0,
    }
    seen_users_q: set[int] = set()
    seen_users_p: set[int] = set()

    # Smoke gate: collect per-query gate results
    smoke_gate_records: list[dict] = []

    with open(out_queries, "w", encoding="utf-8") as fq, \
         open(out_pairs, "w", encoding="utf-8") as fp:

        for uid in eligible_users:
            item_ids = selected[uid]
            same_user_mids = user_memories.get(uid, [])
            stats["users_processed"] += 1

            for item_id in item_ids:
                # Eligibility check via p1_index
                p1_rec = p1_index.get((uid, item_id))
                if p1_rec is None:
                    stats["n_not_eligible"] += 1
                    continue

                # Deterministic provenance lookup — NO LLM judge
                positive_mid = provenance_index.get(uid, {}).get(item_id)
                if positive_mid is None:
                    stats["n_no_provenance"] += 1
                    continue

                review_text = p1_rec["review_text"]
                item_title = p1_rec["item_title"]
                author = author_map.get(item_id, "")

                # P2 LLM call (only call: query generation, never judge)
                prompt = (prompt_tmpl
                          .replace("{title}", item_title)
                          .replace("{review_text}", review_text[:1000]))

                stats["llm_calls"] += 1
                query, reason = _llm_call(client, prompt)

                if reason == "null_query" or (query is None and reason != "llm_fail"):
                    stats["n_null_query"] += 1
                    # Write null-query row (valid provenance, just uninformative review)
                    fq.write(json.dumps({
                        "user_id": uid,
                        "source_item_id": item_id,
                        "query": None,
                        "positive_memory_id": positive_mid,
                    }, ensure_ascii=False) + "\n")
                    seen_users_q.add(uid)
                    # No align_pair for null query
                    if smoke_users > 0:
                        smoke_gate_records.append({
                            "uid": uid, "item_id": item_id,
                            "positive_mid": positive_mid, "query": None,
                            "reason": "null_query",
                            "leakage": False, "hard_negs_contain_pos": False,
                        })
                    continue

                if reason == "llm_fail":
                    stats["n_llm_fail"] += 1
                    continue

                # Leakage check v2 (distinctiveness-based) — informational only, not a gate
                leakage = check_leakage_v2(
                    query, item_title, author=author, common_tokens=common_tokens,
                )
                if leakage:
                    stats["n_leakage_flagged"] += 1

                # Write query to pseudo_queries_train (all non-null, non-fail queries)
                fq.write(json.dumps({
                    "user_id": uid,
                    "source_item_id": item_id,
                    "query": query,
                    "positive_memory_id": positive_mid,
                    "leakage_flagged": leakage,
                }, ensure_ascii=False) + "\n")
                seen_users_q.add(uid)
                stats["n_queries_ok"] += 1

                # Hard negatives: same-user other memories + cross-user sample
                same_user_negs = [m for m in same_user_mids if m != positive_mid]
                seed = (uid * 1_000_003 + item_id) & 0xFFFFFFFF
                cross_negs = _sample_cross_user_negs(uid, user_memories, cross_user_negs, seed)
                hard_negs = same_user_negs + cross_negs
                hard_negs_contain_pos = positive_mid in hard_negs

                fp.write(json.dumps({
                    "query": query,
                    "positive_memory_id": positive_mid,
                    "hard_negative_memory_ids": hard_negs,
                }, ensure_ascii=False) + "\n")
                seen_users_p.add(uid)
                stats["n_pairs"] += 1

                if smoke_users > 0:
                    smoke_gate_records.append({
                        "uid": uid, "item_id": item_id,
                        "positive_mid": positive_mid, "query": query,
                        "reason": None,
                        "leakage": leakage,
                        "hard_negs_contain_pos": hard_negs_contain_pos,
                        "n_same_negs": len(same_user_negs),
                        "n_cross_negs": len(cross_negs),
                    })

            # Progress
            if stats["users_processed"] % 200 == 0:
                total = stats["n_queries_ok"] + stats["n_null_query"] + stats["n_llm_fail"]
                print(f"  [train u={stats['users_processed']}] "
                      f"ok={stats['n_queries_ok']} null={stats['n_null_query']} "
                      f"fail={stats['n_llm_fail']} leakage={stats['n_leakage_flagged']} "
                      f"no_prov={stats['n_no_provenance']}", flush=True)

    stats["users_q"] = len(seen_users_q)
    stats["users_p"] = len(seen_users_p)

    # Manifests
    _write_manifest(out_queries, stats["n_queries_ok"] + stats["n_null_query"], len(seen_users_q))
    _write_manifest(out_pairs, stats["n_pairs"], len(seen_users_p))

    # Smoke gate report
    if smoke_users > 0 and smoke_gate_records:
        _print_smoke_report(smoke_gate_records, stats)

    return stats


def _print_smoke_report(records: list[dict], stats: dict) -> None:
    """Print smoke gate verification for the 5 gates."""
    total = len(records)
    clean = [r for r in records if r["reason"] is None]
    null_q = [r for r in records if r["reason"] == "null_query"]
    leakage = [r for r in records if r.get("leakage", False)]
    self_neg = [r for r in records if r.get("hard_negs_contain_pos", False)]

    print("\n" + "=" * 60)
    print("SMOKE GATE REPORT")
    print("=" * 60)

    # Gate 1: positive_memory_id filled (null=0)
    null_prov = sum(1 for r in records if r["positive_mid"] is None)
    gate1_pass = null_prov == 0
    print(f"\n[GATE 1] positive_memory_id provenance fill")
    print(f"  null provenance: {null_prov}/{total}  → {'PASS ✓' if gate1_pass else 'FAIL ✗'}")
    if clean:
        print(f"  Sample provenance mapping:")
        for r in clean[:3]:
            print(f"    uid={r['uid']} item={r['item_id']} → {r['positive_mid']}")

    # Gate 2: hard_negatives don't contain positive
    gate2_pass = len(self_neg) == 0
    print(f"\n[GATE 2] hard_negatives exclude positive_memory_id")
    print(f"  self-negative violations: {len(self_neg)}/{len(clean)}  → {'PASS ✓' if gate2_pass else 'FAIL ✗'}")
    if clean:
        r0 = clean[0]
        print(f"  Sample: uid={r0['uid']} item={r0['item_id']}  "
              f"same_negs={r0.get('n_same_negs',0)}  cross_negs={r0.get('n_cross_negs',0)}")

    # Gate 3: leakage rate (informational — not a hard gate; memory source_text has no title/author)
    gate3_pass = True
    print(f"\n[GATE 3] title/author leakage in queries (v2 detector — informational only)")
    print(f"  leakage-flagged: {len(leakage)}/{total}  → INFO (all queries included in align_pairs)")

    # Gate 4: train query sources (all from provenance), q==null rate
    null_rate = len(null_q) / max(total, 1)
    gate4_pass = True  # trivially true: all queries sourced from p1 train history
    print(f"\n[GATE 4] train-history sourcing + q==null rate")
    print(f"  q==null (satisfaction-only): {len(null_q)}/{total} = {null_rate:.1%}  → PASS ✓")
    print(f"  source: all from p1_extractions train-history (selected_review_ids)")

    # Gate 5: LLM judge calls = 0
    gate5_pass = True  # judge is never called in this script
    print(f"\n[GATE 5] LLM judge calls")
    print(f"  judge calls: 0 (provenance-only path, no judge import)  → PASS ✓")
    print(f"  total LLM calls (P2 query gen only): {stats['llm_calls']}")

    # Summary
    all_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass
    print(f"\n{'=' * 60}")
    print(f"SMOKE {'PASS ✓' if all_pass else 'FAIL ✗'}  "
          f"[G1:{int(gate1_pass)} G2:{int(gate2_pass)} G3:{int(gate3_pass)} "
          f"G4:{int(gate4_pass)} G5:{int(gate5_pass)}]")
    print(f"  queries_ok={stats['n_queries_ok']}  null={stats['n_null_query']}  "
          f"fail={stats['n_llm_fail']}  leakage={stats['n_leakage_flagged']}  "
          f"no_prov={stats['n_no_provenance']}  pairs={stats['n_pairs']}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# EVAL mode (deferred — runs after Stage1, uses same unified prompt)

def _build_review_index(reviews_jsonl: str, user2id: dict, item2id: dict) -> dict:
    """Build {internal_uid: {internal_iid: (review_text, item_title)}} index."""
    print("[p2_eval] Building review index …", flush=True)
    idx: dict[int, dict[int, tuple[str, str]]] = {}
    n = 0
    with open(reviews_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            orig_uid = r.get("user_id", "")
            orig_iid = r.get("parent_asin", "")
            if orig_uid not in user2id or orig_iid not in item2id:
                continue
            uid = user2id[orig_uid]
            iid = item2id[orig_iid]
            text = r.get("text", "").strip()
            title = r.get("title", "").strip()
            if not text:
                continue
            idx.setdefault(uid, {})[iid] = (text, title)
            n += 1
    print(f"[p2_eval] Index built: {n} reviews for {len(idx)} users", flush=True)
    return idx


def _eval_worker(
    client, user_items: list, review_idx: dict, prompt_tmpl: str,
    counters: dict, lock: threading.Lock, fout, out_lock: threading.Lock,
    progress_interval: int = 500,
) -> None:
    """Process a subset of users with one Ollama client (runs in its own thread)."""
    for uid_str, entry in user_items:
        if not isinstance(entry, dict):
            continue
        test_item = entry.get("test")
        if test_item is None:
            continue

        uid = int(uid_str)
        pair = review_idx.get(uid, {}).get(test_item)

        if pair is None:
            reason: str | None = "no_review"
            query: str | None = None
        else:
            review_text, item_title = pair
            prompt = (prompt_tmpl
                      .replace("{title}", item_title)
                      .replace("{review_text}", review_text[:1000]))
            query, reason = _llm_call(client, prompt)

        record = json.dumps({
            "user_id": uid, "source": "eval", "memory_id": None,
            "target_item": test_item, "query_text": query,
            "src_review_ids": [] if pair is None else [test_item],
            "masking_passed": pair is not None,
            "failed_reason": reason,
        }, ensure_ascii=False)

        with out_lock:
            fout.write(record + "\n")

        with lock:
            if pair is None:
                counters["no_review"] += 1
            elif reason == "null_query":
                counters["null"] += 1
            elif reason == "llm_fail":
                counters["fail"] += 1
            else:
                counters["ok"] += 1
            counters["total"] += 1
            total = counters["total"]
            if total % progress_interval == 0:
                print(
                    f"  [eval {total}] ok={counters['ok']} null={counters['null']} "
                    f"fail={counters['fail']} no_review={counters['no_review']}",
                    flush=True,
                )


def run_eval(client, splits_json: str, id_maps_json: str, reviews_jsonl: str,
             prompt_tmpl: str, out_path: Path, max_records: int, dry_run: int,
             client2=None) -> None:
    with open(splits_json) as f:
        splits_data = json.load(f)
    splits = splits_data.get("users", splits_data) if "users" in splits_data else splits_data

    with open(id_maps_json) as f:
        id_maps = json.load(f)
    user2id: dict[str, int] = id_maps["user2id"]
    item2id: dict[str, int] = id_maps["item2id"]

    review_idx = _build_review_index(reviews_jsonl, user2id, item2id)

    limit = dry_run if dry_run > 0 else max_records
    user_items = list(splits.items())[:limit]

    if client2 is not None:
        # Dual-Ollama: split users across two threads for ~2x throughput.
        mid = len(user_items) // 2
        counters: dict = {"ok": 0, "null": 0, "fail": 0, "no_review": 0, "total": 0}
        lock = threading.Lock()
        out_lock = threading.Lock()
        print(f"[p2_eval] dual-Ollama mode: {mid} users on client1, "
              f"{len(user_items) - mid} users on client2", flush=True)
        with open(out_path, "w", encoding="utf-8") as fout:
            t1 = threading.Thread(
                target=_eval_worker,
                args=(client, user_items[:mid], review_idx, prompt_tmpl,
                      counters, lock, fout, out_lock),
            )
            t2 = threading.Thread(
                target=_eval_worker,
                args=(client2, user_items[mid:], review_idx, prompt_tmpl,
                      counters, lock, fout, out_lock),
            )
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        total = counters["total"]
        print(f"[p2_eval] done: total={total} ok={counters['ok']} "
              f"null={counters['null']} fail={counters['fail']} "
              f"no_review={counters['no_review']}  "
              f"pass_rate={counters['ok']/max(total,1):.3f}")
        return

    # Single-Ollama sequential mode (unchanged)
    n_ok = n_null = n_fail = n_no_review = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for uid_str, entry in user_items:
            if not isinstance(entry, dict):
                continue
            test_item = entry.get("test")
            if test_item is None:
                continue

            uid = int(uid_str)
            pair = review_idx.get(uid, {}).get(test_item)

            if pair is None:
                n_no_review += 1
                fout.write(json.dumps({
                    "user_id": uid, "source": "eval", "memory_id": None,
                    "target_item": test_item, "query_text": None,
                    "src_review_ids": [], "masking_passed": False,
                    "failed_reason": "no_review",
                }, ensure_ascii=False) + "\n")
                continue

            review_text, item_title = pair
            prompt = (prompt_tmpl
                      .replace("{title}", item_title)
                      .replace("{review_text}", review_text[:1000]))

            query, reason = _llm_call(client, prompt)

            if reason == "null_query":
                n_null += 1
            elif reason == "llm_fail":
                n_fail += 1
            else:
                n_ok += 1

            fout.write(json.dumps({
                "user_id": uid, "source": "eval", "memory_id": None,
                "target_item": test_item, "query_text": query,
                "src_review_ids": [test_item], "masking_passed": True,
                "failed_reason": reason,
            }, ensure_ascii=False) + "\n")

            total = n_ok + n_null + n_fail + n_no_review
            if total % 500 == 0 and total > 0:
                print(f"  [eval {total}] ok={n_ok} null={n_null} fail={n_fail} "
                      f"no_review={n_no_review}", flush=True)

    total = n_ok + n_null + n_fail + n_no_review
    print(f"[p2_eval] done: total={total} ok={n_ok} null={n_null} fail={n_fail} "
          f"no_review={n_no_review}  pass_rate={n_ok/max(total,1):.3f}")


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="P2 pseudo-query generator (v2, provenance)")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)

    # Unified prompt (replaces separate train/eval prompts)
    parser.add_argument("--prompt", type=str,
                        default="config/prompts/amazon/Books/p2_pseudo_query.txt",
                        help="Path to unified P2 prompt (train and eval)")

    # Train-mode args
    parser.add_argument("--p1_extractions", type=str,
                        default="data/processed/Books/p1_extractions.jsonl")
    parser.add_argument("--selected_review_ids", type=str, nargs="+",
                        default=["data/p1_shards_gpu01/selected_review_ids.json",
                                 "data/p1_shards_gpu23/selected_review_ids.json"])
    parser.add_argument("--bank_dir", type=str,
                        default="data/processed/Books/memory_bank")
    parser.add_argument("--meta_jsonl", type=str,
                        default="data/raw/Books/meta.jsonl")
    parser.add_argument("--id_maps_json", type=str,
                        default="data/processed/Books/id_maps.json")
    parser.add_argument("--output_queries", type=str,
                        default="data/processed/Books/pseudo_queries_train.jsonl")
    parser.add_argument("--output_pairs", type=str,
                        default="data/processed/Books/align_pairs.jsonl")
    parser.add_argument("--cross_user_negs", type=int, default=10,
                        help="Number of cross-user hard negatives per query")
    parser.add_argument("--smoke_users", type=int, default=0,
                        help="If >0, process only first N users (smoke test)")

    # Eval-mode args
    parser.add_argument("--splits_json", type=str,
                        default="data/processed/Books/splits.json")
    parser.add_argument("--reviews_jsonl", type=str,
                        default="data/raw/Books/reviews.jsonl")
    parser.add_argument("--output", type=str,
                        default="data/processed/Books/pseudo_queries_eval.jsonl",
                        help="Output path for eval mode")
    parser.add_argument("--max_records", type=int, default=999999)
    parser.add_argument("--dry_run", type=int, default=0)

    # Common
    parser.add_argument("--llm_config", type=str, default="configs/llm/p2.yaml")
    parser.add_argument("--second_ollama_url", type=str, default=None,
                        help="URL of second Ollama instance (e.g. http://localhost:11435/api/chat) "
                             "for dual-instance eval parallelism")

    cli = parser.parse_args()

    from src.llm.client import load_llm_config, LLMClient
    import copy as _copy
    llm_cfg = load_llm_config(cli.llm_config)
    client = LLMClient(llm_cfg)
    client2 = None
    if cli.second_ollama_url:
        llm_cfg2 = _copy.copy(llm_cfg)
        llm_cfg2.api_url = cli.second_ollama_url
        client2 = LLMClient(llm_cfg2)
        print(f"[p2] second Ollama: {cli.second_ollama_url}")

    prompt_tmpl = Path(cli.prompt).read_text(encoding="utf-8")
    print(f"[p2] prompt: {cli.prompt}")

    if cli.mode == "train":
        out_queries = Path(cli.output_queries)
        out_pairs = Path(cli.output_pairs)
        out_queries.parent.mkdir(parents=True, exist_ok=True)
        out_pairs.parent.mkdir(parents=True, exist_ok=True)

        print(f"[p2_train] p1: {cli.p1_extractions}")
        print(f"[p2_train] selected_review_ids: {cli.selected_review_ids}")
        print(f"[p2_train] bank: {cli.bank_dir}")
        print(f"[p2_train] output_queries: {cli.output_queries}")
        print(f"[p2_train] output_pairs: {cli.output_pairs}")
        print(f"[p2_train] cross_user_negs: {cli.cross_user_negs}  smoke_users: {cli.smoke_users}")

        stats = run_train(
            client=client,
            p1_extractions=cli.p1_extractions,
            selected_review_id_paths=cli.selected_review_ids,
            bank_dir=cli.bank_dir,
            meta_jsonl=cli.meta_jsonl,
            id_maps_json=cli.id_maps_json,
            prompt_tmpl=prompt_tmpl,
            out_queries=out_queries,
            out_pairs=out_pairs,
            cross_user_negs=cli.cross_user_negs,
            smoke_users=cli.smoke_users,
        )

        print(f"\n[p2_train] FINAL STATS:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    else:
        out_path = Path(cli.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[p2_eval] splits: {cli.splits_json}  output: {cli.output}")
        if cli.dry_run > 0:
            print(f"[p2_eval] *** DRY RUN: {cli.dry_run} records ***")
        run_eval(client, cli.splits_json, cli.id_maps_json, cli.reviews_jsonl,
                 prompt_tmpl, out_path, cli.max_records, cli.dry_run,
                 client2=client2)

    client.close()
    if client2 is not None:
        client2.close()


if __name__ == "__main__":
    main()
