"""STEP 2 — LLM synthesis of cluster_assignments → IntentMemoryUnit JSONL.

Reads data/processed/Books/cluster_assignments.jsonl (from scripts/cluster.py).

Per-cluster LLM call (gemma4:26b, strict-JSON):
  intent_description + preference_signal.summary  ← 1 combined LLM call
  attributes.feature_priorities                   ← algorithmic (keyword_union[:5])
  attributes.avoid                                ← None (algorithmic)
  persona                                         ← {tag: None, description: ""}
                                                     (all disposition_note are null → 0 LLM calls)

★ evidence_map fix: item_ids and timestamps come DIRECTLY from cluster data (never None).

Usage:
  # Smoke test first (5 users):
  python scripts/synth.py --smoke

  # Full run — single GPU:
  python scripts/synth.py --output data/memory/Books/f3_bank.jsonl

  # Full run — 2-instance split (exclusive user ranges):
  python scripts/synth.py --shard 0 2 --output data/memory/Books/f3_bank_s0.jsonl
  python scripts/synth.py --shard 1 2 --output data/memory/Books/f3_bank_s1.jsonl

Smoke gate checks (must ALL pass before reporting success):
  1. evidence.item_ids 빈 [] 0건 — every unit has ≥1 item_id
  2. member set match — unit's item_ids == cluster's item_ids
  3. cluster_size == len(item_ids) — size consistency
  4. persona.tag == null — no LLM persona call (no disposition_note)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient, load_llm_config
from src.memory.synth import make_intent_memory_unit, make_memory_id

# ---------------------------------------------------------------------------
# Paths / constants

CLUSTER_PATH = ROOT / "data/processed/Books/cluster_assignments.jsonl"
DEFAULT_OUT = ROOT / "data/memory/Books/f3_bank.jsonl"
LLM_CONFIG_PATH = ROOT / "configs/llm/p1.yaml"
TAU_PERSONAL = 0.30

# ---------------------------------------------------------------------------
# LLM prompt

SYNTH_SYSTEM = (
    "You are synthesizing a cluster of book reading records into a coherent reader intent. "
    "Output strict JSON only — no markdown, no prose outside the JSON object."
)

SYNTH_USER_TMPL = """You have {n} reading records from the same reader, grouped by shared intent.

Records (intent | preference_keywords):
{members}

Write ONE concise:
1. intent_description: A short phrase (≤15 words) capturing the shared reading goal.
   - Start with a noun phrase or gerund (not "I" or "The reader")
   - No book titles, author names, or series names
2. preference_signal_summary: 1-2 sentences (≤40 words) capturing what this reader values.
   - Describe reading preferences and patterns, not specific books

Output JSON exactly:
{{"intent_description": "...", "preference_signal_summary": "..."}}"""


def build_synth_prompt(cluster: dict) -> list[dict]:
    """Build messages list for one LLM call."""
    n = cluster["size"]
    lines = []
    for i in range(n):
        intent = cluster["intents"][i] if i < len(cluster["intents"]) else ""
        pref_raw = cluster["pref_summaries"][i] if i < len(cluster["pref_summaries"]) else ""
        kw = " ".join(pref_raw.split()[:8])
        lines.append(f"  {i+1}. {intent} | {kw}")
    user_msg = SYNTH_USER_TMPL.format(n=n, members="\n".join(lines))
    return [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


REQUIRED_KEYS = {"intent_description", "preference_signal_summary"}


def parse_synth_response(text: str) -> dict | None:
    """Parse LLM JSON response.  Returns None on failure."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or not REQUIRED_KEYS.issubset(obj.keys()):
        return None
    if not obj.get("intent_description") or not obj.get("preference_signal_summary"):
        return None
    return obj


# ---------------------------------------------------------------------------
# Synthesize one cluster

def synth_one_cluster(
    user_id: Any,
    cluster: dict,
    k_personal: int,
    llm: LLMClient,
    tau: float,
) -> dict:
    """Build one IntentMemoryUnit with LLM intent+summary, algorithmic rest.

    evidence.item_ids / evidence.timestamps come directly from cluster['item_ids']
    / cluster['timestamps'] — never None, never [].
    """
    mid = make_memory_id(user_id, cluster["label"], tau)
    messages = build_synth_prompt(cluster)

    # Try generate (may be a cache hit or live call)
    raw, _, _, cache_hit = llm.generate(messages)
    llm_result = parse_synth_response(raw)

    if llm_result is None and not cache_hit:
        # Retry with larger budget
        for _ in range(llm.config.retry_max):
            raw2, _, _, _ = llm.generate(
                messages, max_new_tokens=llm.config.retry_max_new_tokens
            )
            llm_result = parse_synth_response(raw2)
            if llm_result:
                llm.store_parsed(messages, json.dumps(llm_result))
                break

    # centroid stored as list in JSONL; need ndarray for make_intent_memory_unit
    cluster_copy = dict(cluster)
    cluster_copy["_k_personal"] = k_personal
    if isinstance(cluster_copy.get("centroid"), list):
        cluster_copy["centroid"] = np.array(cluster_copy["centroid"], dtype=np.float32)

    unit = make_intent_memory_unit(
        memory_id=mid,
        user_id=user_id,
        cluster=cluster_copy,
        tau=tau,
        is_prototype=False,
        evidence_item_ids=cluster["item_ids"],       # ★ full provenance, never None
        evidence_timestamps=cluster["timestamps"],   # ★ full provenance, never None
        evidence_snippets=cluster["review_snippets"],
    )

    # Patch LLM-generated fields (override algorithmic defaults from make_intent_memory_unit)
    if llm_result:
        unit["intent_description"] = llm_result["intent_description"]
        unit["preference_signal"]["summary"] = llm_result["preference_signal_summary"]
        # Keep embedding.source_text in sync with new intent
        kw_union = " ".join(cluster.get("keyword_union", []))
        intent_desc = llm_result["intent_description"]
        unit["embedding"]["source_text"] = (
            f"{intent_desc} {kw_union}".strip() if kw_union else intent_desc
        )

    unit["meta"]["llm_synth"] = llm_result is not None
    unit["meta"]["created_by"] = "synth_v1"
    return unit


# ---------------------------------------------------------------------------
# Load cluster assignments

def load_cluster_assignments(path: Path, shard: int = 0, n_shards: int = 1) -> list[dict]:
    users = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            users.append(json.loads(line))
    users.sort(key=lambda u: str(u["user_id"]))
    if n_shards > 1:
        users = [u for i, u in enumerate(users) if i % n_shards == shard]
    return users


# ---------------------------------------------------------------------------
# Smoke gate

def run_smoke_gate(units: list[dict], clusters_by_mid: dict[str, dict]) -> dict:
    empty_item_ids = [u["memory_id"] for u in units if not u["evidence"]["item_ids"]]
    member_mismatches, size_mismatches = [], []
    bad_persona = [u["memory_id"] for u in units if u["persona"]["tag"] is not None]

    for unit in units:
        mid = unit["memory_id"]
        cluster = clusters_by_mid.get(mid)
        if cluster is None:
            continue
        expected_set = set(cluster["item_ids"])
        actual_set = set(unit["evidence"]["item_ids"])
        if actual_set != expected_set:
            member_mismatches.append({"mid": mid, "diff": list(expected_set ^ actual_set)[:5]})
        if len(unit["evidence"]["item_ids"]) != cluster["size"]:
            size_mismatches.append({
                "mid": mid,
                "expected": cluster["size"],
                "actual": len(unit["evidence"]["item_ids"]),
            })

    all_pass = not any([empty_item_ids, member_mismatches, size_mismatches, bad_persona])
    return {
        "n_units": len(units),
        "g1_empty_item_ids": {"count": len(empty_item_ids), "examples": empty_item_ids[:5], "pass": not empty_item_ids},
        "g2_member_mismatch": {"count": len(member_mismatches), "examples": member_mismatches[:3], "pass": not member_mismatches},
        "g3_size_mismatch": {"count": len(size_mismatches), "examples": size_mismatches[:3], "pass": not size_mismatches},
        "g4_persona_tag_none": {"bad_count": len(bad_persona), "pass": not bad_persona},
        "all_pass": all_pass,
    }


# ---------------------------------------------------------------------------
# Main

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run on N_SMOKE users, gate-check, then print full cmd")
    parser.add_argument("--n_smoke", type=int, default=5)
    parser.add_argument("--shard", type=int, nargs=2, default=[0, 1],
                        metavar=("SHARD_ID", "N_SHARDS"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tau", type=float, default=TAU_PERSONAL)
    args = parser.parse_args(argv)

    shard_id, n_shards = args.shard
    t_start = time.time()
    mode = "SMOKE" if args.smoke else f"FULL shard={shard_id}/{n_shards}"
    print("=" * 60)
    print(f"STEP 2 — Synth  mode={mode}  τ={args.tau}")
    print("=" * 60)

    print(f"\n[1/3] Loading cluster assignments ...")
    users = load_cluster_assignments(CLUSTER_PATH, shard=shard_id, n_shards=n_shards)
    total_clusters = sum(len(u["clusters"]) for u in users)
    print(f"  {len(users)} users, {total_clusters} clusters")

    if args.smoke:
        multi_k = [u for u in users if u["k_personal"] >= 2]
        single_k = [u for u in users if u["k_personal"] == 1]
        users = (multi_k[:max(args.n_smoke - 2, 1)] + single_k[:2])[:args.n_smoke]
        n_smoke_clusters = sum(len(u["clusters"]) for u in users)
        print(f"  [smoke] {len(users)} users, {n_smoke_clusters} clusters selected")

    print(f"\n[2/3] Loading LLM config ...")
    llm_cfg = load_llm_config(str(LLM_CONFIG_PATH))
    llm_cfg = dataclasses.replace(llm_cfg, prompt_version="synth_v1")
    llm = LLMClient(llm_cfg)
    print(f"  model={llm_cfg.model_id}  cache={llm_cfg.cache_db}")

    print(f"\n[3/3] Synthesizing ...")
    all_units: list[dict] = []
    clusters_by_mid: dict[str, dict] = {}
    n_live, n_cached = 0, 0

    for u_idx, user in enumerate(users):
        uid = user["user_id"]
        k = user["k_personal"]
        for cluster in user["clusters"]:
            mid = make_memory_id(uid, cluster["label"], args.tau)
            # Quick cache check before full synth_one_cluster call
            msgs = build_synth_prompt(cluster)
            _, _, _, is_cached = llm.generate(msgs)
            if is_cached:
                n_cached += 1
            else:
                n_live += 1

            unit = synth_one_cluster(uid, cluster, k, llm, args.tau)
            all_units.append(unit)
            if args.smoke:
                clusters_by_mid[mid] = cluster

        if (u_idx + 1) % 200 == 0:
            print(f"  [{u_idx+1}/{len(users)}]  units={len(all_units)}  "
                  f"live={n_live}  cached={n_cached}  "
                  f"elapsed={time.time()-t_start:.0f}s")

    llm.close()
    print(f"\n  Done: {len(all_units)} units  live={n_live}  cached={n_cached}  "
          f"elapsed={time.time()-t_start:.1f}s")

    if args.smoke:
        print("\n" + "=" * 60)
        print("SMOKE GATE CHECKS")
        print("=" * 60)
        gate = run_smoke_gate(all_units, clusters_by_mid)
        for gk, gv in gate.items():
            if gk in ("n_units", "all_pass"):
                continue
            status = "PASS" if gv.get("pass") else "FAIL"
            print(f"  [{status}] {gk}: {gv}")
        print(f"\n  ALL GATES PASS: {gate['all_pass']}")
        if gate["all_pass"]:
            _print_full_commands(args.output)
        else:
            sys.exit(1)

    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for unit in all_units:
                f.write(json.dumps(unit, ensure_ascii=False) + "\n")
        print(f"[save] {args.output}  ({args.output.stat().st_size // 1024} KB)")
        print(f"  {len(all_units)} IntentMemoryUnits written")


def _print_full_commands(default_out: Path) -> None:
    print(f"""
{"=" * 60}
FULL SYNTH COMMANDS
{"=" * 60}

# Single GPU:
docker exec qaim-rec bash -c "cd /qaim-rec && python3 scripts/synth.py \\
    --output {default_out}"

# 2-instance GPU split (run concurrently on separate containers/devices):
# Instance 0 (first half of users):
docker exec qaim-rec bash -c "cd /qaim-rec && CUDA_VISIBLE_DEVICES=0 python3 scripts/synth.py \\
    --shard 0 2 \\
    --output data/memory/Books/f3_bank_s0.jsonl"

# Instance 1 (second half of users):
docker exec qaim-rec bash -c "cd /qaim-rec && CUDA_VISIBLE_DEVICES=1 python3 scripts/synth.py \\
    --shard 1 2 \\
    --output data/memory/Books/f3_bank_s1.jsonl"

# Merge shards after both finish:
cat data/memory/Books/f3_bank_s0.jsonl \\
    data/memory/Books/f3_bank_s1.jsonl \\
    > {default_out}
""")


if __name__ == "__main__":
    main()
