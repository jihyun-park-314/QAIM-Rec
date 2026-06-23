# -*- coding: utf-8 -*-
"""Task [2] validation: 18-case labeled mini-set for v0.4.4 prompt.

Groups:
  A — 10 prior mismatches (old prompt said generic, v0.4.4 should say TRUE)
  B — 5 pure-approval (expected: is_discriminative=false, grounding review_only)
  C — 3 metadata-dominant (expected: is_discriminative=false, grounding metadata_dominant)

Run:
  python scripts/validate_miniset.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.llm.client import LLMClient, LLMConfig
from src.llm.prompts.p1_books import run_p1_books

LLM_CONFIG = "configs/llm/p1.yaml"
CANDIDATES = "data/processed/Books/books_memory_candidates.jsonl"

# ---- Mini-set definition ---------------------------------------------------
# (user_id, item_id, expected_is_discriminative, expected_grounding, group, note)
MINI_SET = [
    # Group A: prior mismatches — old prompt labeled FALSE, v0.4.4 should be TRUE
    (297,   7720, True,  None,                "A", "pacing+character"),
    (1068,  8070, True,  None,                "A", "series continuation"),
    (1419,  7370, True,  None,                "A", "suspense/tension"),
    (3002,  3716, True,  None,                "A", "immersion+situational (was metadata_dom)"),
    (3828,  7270, True,  None,                "A", "pacing+series"),
    (4194,  7771, True,  None,                "A", "pacing+character"),
    (4646,  4654, True,  None,                "A", "series continuation"),
    (7681,  5126, True,  None,                "A", "pacing"),
    (8902,  5614, True,  None,                "A", "structure+character (was metadata_dom)"),
    (10143, 5452, True,  None,                "A", "immersion+series"),
    # Group B: pure-approval — expected FALSE, grounding review_only
    (4619,  5331, False, "review_only",       "B", "series name approval only"),
    (10,    5839, False, "review_only",       "B", "author/series name approval"),
    (22,    5371, False, "review_only",       "B", "pure approval, no axis"),
    (36,    5415, False, "review_only",       "B", "author approval only"),
    (2,     4697, False, "review_only",       "B", "author approval only"),
    # Group C: metadata-dominant — expected FALSE, grounding metadata_dominant
    (1105,  987,  False, "metadata_dominant", "C", "subject from metadata"),
    (7603,  5910, False, "metadata_dominant", "C", "type from metadata"),
    (6486,  6954, False, "metadata_dominant", "C", "kind from metadata"),
]


def load_candidates(path: str) -> dict:
    lookup = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                lookup[(c["user_id"], c["item_id"])] = c
    return lookup


def main():
    with open(LLM_CONFIG) as f:
        raw = yaml.safe_load(f)
    cfg = LLMConfig(
        model_id=raw["model_id"],
        api_url=raw["api_url"],
        prompt_version=f"miniset_v044_{int(time.time())}",  # unique = no cache reuse
        temperature=raw.get("temperature", 0.0),
        max_new_tokens=raw.get("max_new_tokens", 512),
        cache_db=raw.get("cache_db", "data/llm_cache.db"),
    )
    client = LLMClient(cfg)
    lookup = load_candidates(CANDIDATES)

    print(f"=== v0.4.4 Mini-set Validation ({len(MINI_SET)} cases) ===\n")
    header = f"{'#':>2}  {'Grp':3}  {'u':>5} {'i':>5}  {'Exp':5}  {'Got':5}  {'ExpGnd':16}  {'GotGnd':20}  {'OK?':4}  {'CI (truncated)':40}  Note"
    print(header)
    print("-" * len(header))

    correct = 0
    total = 0
    boundary_cases = []

    for idx, (uid, iid, exp_disc, exp_grnd, group, note) in enumerate(MINI_SET, 1):
        item = lookup.get((uid, iid))
        if item is None:
            print(f"{idx:>2}  {group}  {uid:>5} {iid:>5}  {'?':5}  {'?':5}  {'MISSING':16}  {'':20}  SKIP")
            continue

        result = run_p1_books(client, item, variant="B")

        if result.parsed is None:
            got_disc = None
            got_grnd = "PARSE_FAIL"
            got_ci = ""
            ok = False
        else:
            got_disc = result.parsed.get("is_discriminative")
            got_grnd = result.parsed.get("grounding_level", "")
            got_ci = result.parsed.get("contextual_intent", "")

            disc_ok = (got_disc == exp_disc)
            grnd_ok = (exp_grnd is None) or (got_grnd == exp_grnd)
            # For false cases: contextual_intent must be empty
            ci_ok = True
            if exp_disc is False:
                ci_ok = (got_ci == "")
            ok = disc_ok and grnd_ok and ci_ok

        if ok:
            correct += 1
        total += 1

        exp_disc_str = str(exp_disc)
        got_disc_str = str(got_disc) if got_disc is not None else "FAIL"
        exp_grnd_str = exp_grnd or "(any)"
        ci_short = got_ci[:40] if got_ci else ""
        ok_str = "OK" if ok else "FAIL"
        lat = f"{result.latency_s:.1f}s" if not result.cache_hit else "cache"

        print(f"{idx:>2}  {group}  {uid:>5} {iid:>5}  {exp_disc_str:5}  {got_disc_str:5}  "
              f"{exp_grnd_str:16}  {got_grnd:20}  {ok_str:4}  {ci_short:40}  [{lat}] {note}")

        # Collect boundary for qualitative review
        rev = item.get("review_text", "").lower()
        for kw in ["engrossing", "immersive", "interesting", "compelling", "good read"]:
            if kw in rev and group == "A":
                boundary_cases.append((uid, iid, kw, got_disc, got_ci[:80]))

    print()
    print(f"Accuracy: {correct}/{total} = {correct/total:.3f}")

    if boundary_cases:
        print()
        print("=== Boundary keyword cases (from Group A, for qualitative review) ===")
        for uid, iid, kw, got_disc, ci in boundary_cases:
            item = lookup.get((uid, iid), {})
            print(f"  u={uid} i={iid} kw='{kw}' → disc={got_disc}  CI: {ci}")
            print(f"    review: {repr(item.get('review_text','')[:120])}")


if __name__ == "__main__":
    main()
