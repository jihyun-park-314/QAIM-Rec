"""Full 2^4 ablation of P2 prompt factors.

Fixed across all variants: C4 base text, null-case, strict-JSON schema.
Variable (O = C4-like/v1, X = enhanced):
  L: length constraint      O=none (~40-50 words)        X=4-15 words
  A: author guard           O=product name only          X=product + author
  G: genre vocabulary       O=none (1st-person only)     X=explicit genre/tone/structure examples
  T: title context          O=not provided               X=provided with do-NOT-copy guard

16 variants × N reviews = full ablation.

Usage:
  python3 scripts/ablate_p2_prompts.py \\
    --p1_extractions data/processed/Books/p1_extractions.jsonl \\
    --selected_review_ids data/p1_shards_gpu01/selected_review_ids.json \\
                          data/p1_shards_gpu23/selected_review_ids.json \\
    --meta_jsonl data/raw/Books/meta.jsonl \\
    --id_maps_json data/processed/Books/id_maps.json \\
    --llm_config /tmp/p2_smoke.yaml \\
    --smoke_users 5 \\
    --output ablation_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.pipeline import check_leakage_v2, build_common_tokens_from_titles
from scripts.generate_pseudo_queries import (
    load_p1_index,
    load_selected_ids,
    load_author_map,
    _llm_call,
)

# ---------------------------------------------------------------------------
# Prompt builder — uses str.replace() to avoid issues with literal {} in JSON schema

_C4_INTRO_O = (
    "Given a review from Amazon, can you rephrase it with a first-person tone "
    "as if you are the customer looking for a product? "
    "Give me the rephrased output without saying anything else. "
    "Note that the name of the product must not show in the output since you are looking for it. "
    "You should ignore irrelevant information that doesn't help with the rewriting."
)

_C4_INTRO_X = (
    "Given a review from Amazon, can you rephrase it with a first-person tone "
    "as if you are the customer looking for a product? "
    "Give me the rephrased output without saying anything else. "
    "Note that the name of the product and the author must not show in the output "
    "since you are looking for it. "
    "You should ignore irrelevant information that doesn't help with the rewriting."
)

_GENRE_BLOCK = """\
  Express it in generalized terms — genre/subgenre, pacing (page-turner, slow-burn),
  emotional tone (cozy, dark, uplifting), themes, narrative structure (unreliable narrator,
  dual timeline), reader level (YA, literary fiction), series/standalone preference.
  Examples:
    "slow-burn psychological thriller with unreliable narrator"
    "heartwarming historical fiction family saga easy read"
    "hard sci-fi space opera with detailed world-building"
    "cozy mystery series small town amateur sleuth"
"""


def build_prompt_template(L: bool, A: bool, G: bool, T: bool) -> str:
    """Build prompt template using __TITLE__ and __REVIEW__ as substitution markers."""
    intro = _C4_INTRO_X if A else _C4_INTRO_O

    str_line = "the rephrased query in first-person tone, 4–15 words." if L \
        else "the rephrased query in first-person tone."

    genre_section = "\n" + _GENRE_BLOCK if G else ""

    title_block = (
        "Item title (context only — do NOT copy title, author, or series tokens into output): __TITLE__\n"
        "Category: Books\n"
    ) if T else ""

    # Literal braces in JSON schema are fine here since we use replace(), not format()
    prompt = (
        f"{intro}\n\n"
        'Output strict JSON: {"query": str | null}\n'
        f"- str: {str_line}{genre_section}"
        '- null: only if the review expresses nothing but satisfaction ("loved it", "great book",\n'
        '  "highly recommend") without revealing any reading need.\n\n'
        f"{title_block}"
        'Review: """__REVIEW__"""'
    )
    return prompt


def format_prompt(template: str, title: str, review_text: str) -> str:
    return template.replace("__TITLE__", title).replace("__REVIEW__", review_text)


# ---------------------------------------------------------------------------
# Build all 16 variants

def all_variants() -> list[tuple[str, str]]:
    """Return [(code, template), ...] for all 16 LAGT combinations."""
    variants = []
    for L, A, G, T in product([False, True], repeat=4):
        code = (
            ("X" if L else "O")
            + ("X" if A else "O")
            + ("X" if G else "O")
            + ("X" if T else "O")
        )
        tmpl = build_prompt_template(L, A, G, T)
        variants.append((code, tmpl))
    return variants


# ---------------------------------------------------------------------------
# Data loading

def _build_pairs(p1_index, selected_ids, smoke_users):
    uids = sorted(selected_ids.keys())
    if smoke_users:
        uids = uids[:smoke_users]
    pairs = []
    for uid in uids:
        for iid in selected_ids.get(uid, []):
            key = (uid, iid)
            if key in p1_index:
                rec = p1_index[key]
                pairs.append((uid, iid, rec["review_text"], rec["item_title"]))
    return pairs


# ---------------------------------------------------------------------------
# Run one variant

def run_variant(
    code: str,
    template: str,
    pairs: list,
    author_map: dict,
    common_tokens: frozenset,
    client,
) -> dict:
    records = []
    for idx, (uid, iid, review_text, item_title) in enumerate(pairs):
        prompt = format_prompt(template, item_title or "", review_text)
        query, reason = _llm_call(client, prompt)
        author = author_map.get(iid, "")

        title_leak = False
        author_leak = False
        if query:
            title_leak = check_leakage_v2(
                query, item_title or "", author="", common_tokens=common_tokens
            )
            if author:
                author_leak = check_leakage_v2(
                    query, title="", author=author, common_tokens=common_tokens
                )

        records.append({
            "uid": uid,
            "iid": iid,
            "title": item_title or "",
            "author": author,
            "query": query,
            "reason": reason,
            "title_leak": title_leak,
            "author_leak": author_leak,
        })

        if (idx + 1) % 10 == 0:
            print(f"    [{code}] {idx+1}/{len(pairs)}", flush=True)

    n = len(records)
    n_null = sum(1 for r in records if r["reason"] == "null_query")
    n_fail = sum(1 for r in records if r["reason"] == "llm_fail")
    n_tl = sum(1 for r in records if r["title_leak"])
    n_al = sum(1 for r in records if r["author_leak"])

    return {
        "code": code,
        "n": n,
        "null_rate": round(n_null / max(n, 1), 4),
        "fail_rate": round(n_fail / max(n, 1), 4),
        "title_leak_rate": round(n_tl / max(n, 1), 4),
        "author_leak_rate": round(n_al / max(n, 1), 4),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Reporting

def print_summary(results: list[dict]) -> None:
    header = f"{'code':<6} {'null%':>6} {'fail%':>6} {'title_lk%':>10} {'author_lk%':>11}"
    print("\n" + "=" * 55)
    print("  Ablation summary (L A G T — O=C4-like, X=enhanced)")
    print("=" * 55)
    print(f"  {header}")
    print("  " + "-" * 50)
    for r in sorted(results, key=lambda x: x["code"]):
        print(
            f"  {r['code']:<6} "
            f"{r['null_rate']*100:>6.1f} "
            f"{r['fail_rate']*100:>6.1f} "
            f"{r['title_leak_rate']*100:>10.1f} "
            f"{r['author_leak_rate']*100:>11.1f}"
        )
    print("=" * 55)
    print("  Columns: L=length  A=author-guard  G=genre-vocab  T=title-context")


def print_samples(results: list[dict], n_samples: int = 3) -> None:
    # Find pairs where both OOOO and XXXX produced non-null queries — good contrast cases
    oooo = next((r for r in results if r["code"] == "OOOO"), None)
    xxxx = next((r for r in results if r["code"] == "XXXX"), None)
    if not oooo or not xxxx:
        return

    print("\n" + "=" * 55)
    print("  OOOO vs XXXX query pairs (first non-null)")
    print("=" * 55)
    shown = 0
    for ra, rb in zip(oooo["records"], xxxx["records"]):
        if shown >= n_samples:
            break
        if ra["reason"] or rb["reason"]:
            continue
        print(f"\n  uid={ra['uid']}  iid={ra['iid']}")
        print(f"  title  : {ra['title'][:60]}")
        print(f"  author : {ra['author'] or '(none)'}")
        print(f"  OOOO   : {ra['query']}")
        print(f"  XXXX   : {rb['query']}")
        shown += 1


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1_extractions", required=True)
    ap.add_argument("--selected_review_ids", nargs="+", required=True)
    ap.add_argument("--meta_jsonl", required=True)
    ap.add_argument("--id_maps_json", required=True)
    ap.add_argument("--llm_config", required=True)
    ap.add_argument("--smoke_users", type=int, default=5)
    ap.add_argument("--output", default="data/processed/Books/ablation_p2.json")
    args = ap.parse_args()

    print("Loading data ...", flush=True)
    p1_index = load_p1_index(args.p1_extractions)
    selected_ids = load_selected_ids(args.selected_review_ids)
    author_map = load_author_map(args.meta_jsonl, args.id_maps_json)
    all_titles = [rec["item_title"] for rec in p1_index.values() if rec.get("item_title")]
    common_tokens = build_common_tokens_from_titles(all_titles)

    pairs = _build_pairs(p1_index, selected_ids, args.smoke_users)
    print(f"Reviews: {len(pairs)}  users: {args.smoke_users}", flush=True)

    from src.llm.client import load_llm_config, LLMClient
    client = LLMClient(load_llm_config(args.llm_config))

    variants = all_variants()
    print(f"Variants: {len(variants)}  Total LLM calls: {len(variants) * len(pairs)}", flush=True)

    results = []
    for i, (code, tmpl) in enumerate(variants):
        print(f"\n[{i+1:02d}/16] Variant {code} ...", flush=True)
        result = run_variant(code, tmpl, pairs, author_map, common_tokens, client)
        results.append(result)
        print(
            f"         null={result['null_rate']*100:.1f}%  "
            f"title_leak={result['title_leak_rate']*100:.1f}%  "
            f"author_leak={result['author_leak_rate']*100:.1f}%",
            flush=True,
        )

    print_summary(results)
    print_samples(results)

    # Save full results (without per-record prompt text to keep file small)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
