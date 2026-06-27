"""C4+JSON variant with no null option — every review must produce a query.

Prompt: C4 original intro + strict JSON {"query": str} only (no null branch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.pipeline import check_leakage_v2, build_common_tokens_from_titles
from scripts.generate_pseudo_queries import (
    load_p1_index, load_selected_ids, load_author_map, _llm_call,
)
from scripts.ablate_p2_prompts import _build_pairs

_PROMPT = (
    'Given a review from Amazon, can you rephrase it with a first-person tone '
    'as if you are the customer looking for a product? '
    'Give me the rephrased output without saying anything else. '
    'Note that the name of the product must not show in the output since you are looking for it. '
    'You should ignore irrelevant information that doesn\'t help with the rewriting.\n\n'
    'Output strict JSON: {"query": str}\n'
    '- str: the rephrased query in first-person tone.\n\n'
    'Review: """__REVIEW__"""'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1_extractions", required=True)
    ap.add_argument("--selected_review_ids", nargs="+", required=True)
    ap.add_argument("--meta_jsonl", required=True)
    ap.add_argument("--id_maps_json", required=True)
    ap.add_argument("--llm_config", required=True)
    ap.add_argument("--smoke_users", type=int, default=5)
    ap.add_argument("--output", default="data/processed/Books/c4_no_null.json")
    args = ap.parse_args()

    p1_index = load_p1_index(args.p1_extractions)
    selected_ids = load_selected_ids(args.selected_review_ids)
    author_map = load_author_map(args.meta_jsonl, args.id_maps_json)
    all_titles = [rec["item_title"] for rec in p1_index.values() if rec.get("item_title")]
    common_tokens = build_common_tokens_from_titles(all_titles)
    pairs = _build_pairs(p1_index, selected_ids, args.smoke_users)

    from src.llm.client import load_llm_config, LLMClient
    client = LLMClient(load_llm_config(args.llm_config))

    print(f"Reviews: {len(pairs)}", flush=True)
    records = []
    for idx, (uid, iid, review_text, item_title) in enumerate(pairs):
        prompt = _PROMPT.replace("__REVIEW__", review_text)
        query, reason = _llm_call(client, prompt)
        author = author_map.get(iid, "")
        title_leak = check_leakage_v2(query, item_title or "", author="", common_tokens=common_tokens) if query else False
        author_leak = check_leakage_v2(query, title="", author=author, common_tokens=common_tokens) if (query and author) else False
        records.append({"uid": uid, "iid": iid, "title": item_title or "", "author": author,
                         "query": query, "reason": reason, "title_leak": title_leak, "author_leak": author_leak})
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(pairs)}", flush=True)

    n = len(records)
    n_null = sum(1 for r in records if r["reason"] == "null_query")
    n_fail = sum(1 for r in records if r["reason"] == "llm_fail")
    n_tl = sum(1 for r in records if r["title_leak"])
    n_al = sum(1 for r in records if r["author_leak"])

    result = {
        "code": "C4J",
        "n": n, "null_rate": round(n_null/max(n,1),4), "fail_rate": round(n_fail/max(n,1),4),
        "title_leak_rate": round(n_tl/max(n,1),4), "author_leak_rate": round(n_al/max(n,1),4),
        "records": records,
    }
    print(f"\nnull={result['null_rate']*100:.1f}%  fail={result['fail_rate']*100:.1f}%  "
          f"title_lk={result['title_leak_rate']*100:.1f}%  author_lk={result['author_leak_rate']*100:.1f}%")

    # Sample queries
    print("\n--- sample queries ---")
    for r in records[:5]:
        print(f"  [{r['uid']},{r['iid']}] {r['query']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
