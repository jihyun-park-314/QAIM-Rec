"""Spot-check: verify 4 sentiment-only mismatches flip to is_discriminative=false
after adding negative examples to p1_books_b.txt.

Target reviews (A=false, B=true in original smoke; expected to flip with fixed prompt):
  - user=7681, item=5126: "hard to put down as it was so interesting plot"
  - user=8902, item=5614: "plenty of twists and turns, keep you reading..."
  - user=10143, item=5452: "Can't wait to read the next book"
  - user=3160, item=7303: "told with feeling"
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.client import LLMClient, load_llm_config
from src.llm.prompts.p1_books import run_p1_books

TARGETS = {
    (7681, 5126), (8902, 5614), (10143, 5452), (3160, 7303)
}

LLM_CONFIG = "configs/llm/p1.yaml"
CANDIDATES = "data/processed/Books/books_memory_candidates.jsonl"


def main():
    # Collect target reviews
    target_items = []
    with open(CANDIDATES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("user_id"), r.get("item_id"))
            if key in TARGETS:
                target_items.append(r)
                if len(target_items) == len(TARGETS):
                    break

    print(f"Found {len(target_items)}/{len(TARGETS)} target reviews")

    llm_cfg = load_llm_config(LLM_CONFIG)
    client = LLMClient(llm_cfg)

    results = []
    for item in target_items:
        key = (item.get("user_id"), item.get("item_id"))
        res = run_p1_books(client, item, variant="B", retry_max=2)
        disc = res.parsed.get("is_discriminative") if res.parsed else None
        ev = res.parsed.get("evidence_span") if res.parsed else []
        print(f"  user={key[0]} item={key[1]}  is_discriminative={disc}  "
              f"cache_hit={res.cache_hit}  lat={res.latency_s:.1f}s")
        print(f"    evidence: {ev}")
        results.append({"key": key, "is_discriminative": disc, "cache_hit": res.cache_hit})

    client.close()

    n_flipped = sum(1 for r in results if r["is_discriminative"] is False)
    print(f"\n{n_flipped}/{len(results)} flipped to is_discriminative=false")
    if n_flipped == len(TARGETS):
        print("PASS — all 4 sentiment-only cases correctly rejected")
    else:
        print("PARTIAL — check remaining cases manually")


if __name__ == "__main__":
    main()
