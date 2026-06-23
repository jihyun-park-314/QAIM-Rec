"""Extract leakage cases from the pilot run for manual inspection.

All LLM calls are cache hits so this runs in seconds.

Usage:
  python3 -m src.memory.dump_leakage \
      --pilot_jsonl data/memory_pilot/pilot_b_u300_seed42.jsonl \
      --variant B \
      --output reports/leakage_cases_b_u300.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from src.llm.client import LLMClient, load_llm_config
from src.memory.pipeline import extract_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_jsonl", default="data/memory_pilot/pilot_b_u300_seed42.jsonl")
    parser.add_argument("--variant", default="B")
    parser.add_argument("--candidates", default="data/processed/Books/books_memory_candidates.jsonl")
    parser.add_argument("--llm_config_path", default="configs/llm/p1.yaml")
    parser.add_argument("--output", default="reports/leakage_cases_b_u300.json")
    args = parser.parse_args()

    # Load pilot summary to find users with leakage
    pilot_rows = [json.loads(l) for l in open(args.pilot_jsonl, encoding="utf-8")]
    leaky_user_ids = {r["user_id"] for r in pilot_rows if r["n_leakage"] > 0}
    print(f"Users with leakage: {len(leaky_user_ids)}")

    # Load candidates for those users
    candidates = defaultdict(list)
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["user_id"] in leaky_user_ids:
                candidates[row["user_id"]].append(row)

    # Re-extract (all cache hits)
    llm_cfg = load_llm_config(args.llm_config_path)
    client = LLMClient(llm_cfg)

    leakage_cases = []
    for uid, items in candidates.items():
        for item in items[:12]:
            rec = extract_record(client, item, variant=args.variant)
            if rec.leakage_detected:
                leakage_cases.append({
                    "user_id": uid,
                    "item_id": rec.item_id,
                    "item_title": rec.item_title,
                    "contextual_intent": rec.contextual_intent,
                    "preference_summary": rec.preference_summary,
                    "source_text": rec.source_text,
                    "evidence_span": rec.evidence_span,
                    "grounding_level": rec.grounding_level,
                    "review_excerpt": rec.review_text_original[:200],
                })

    client.close()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(leakage_cases, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(leakage_cases)} leakage cases → {args.output}")


if __name__ == "__main__":
    main()
