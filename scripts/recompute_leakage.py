"""Recompute leakage_detected field using check_leakage_v2 (plan.md v0.4.7).

Usage:
    python3 scripts/recompute_leakage.py \
        --input data/processed/Books/p1_extractions.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.memory.pipeline import recompute_leakage_field, build_common_tokens_from_titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="p1_extractions.jsonl path")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print(f"[leakage_v2] Loading {input_path} ...")
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[leakage_v2] {len(records)} records loaded")

    titles = [r.get("item_title", "") for r in records]
    common_tokens = build_common_tokens_from_titles(titles)
    print(f"[leakage_v2] common_tokens built: {len(common_tokens)} tokens")

    updated, stats = recompute_leakage_field(records, author_map=None, common_tokens=common_tokens)

    print(f"[leakage_v2] old_fire_rate : {stats['old_fire_rate']:.4f}  ({int(stats['old_fire_rate']*stats['n_records'])} / {stats['n_records']})")
    print(f"[leakage_v2] new_fire_rate : {stats['new_fire_rate']:.4f}  ({int(stats['new_fire_rate']*stats['n_records'])} / {stats['n_records']})")
    print(f"[leakage_v2] flipped_off   : {stats['flipped_off']}  (FP removed)")
    print(f"[leakage_v2] flipped_on    : {stats['flipped_on']}  (previously missed)")

    print(f"[leakage_v2] Writing to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in updated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[leakage_v2] Done.")


if __name__ == "__main__":
    main()
