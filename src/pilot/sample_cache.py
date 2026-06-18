"""Pilot 1 sample cache: sample once per (category, seed), reuse for A/B.

Cache: data/processed/{category}/pilot1_sample_n20_seed{seed}.jsonl
Both p1_base and p1_aspect read from the same file → guaranteed identical reviews.
Smoke test (n_samples=3) reads the first 3 entries of the N=20 cache.
"""
from __future__ import annotations

import json
import os

from src.data.sample import sample_review_pairs

CACHE_N = 20


def cache_path(category: str, seed: int, cache_dir: str = "data/processed") -> str:
    return os.path.join(cache_dir, category, f"pilot1_sample_n{CACHE_N}_seed{seed}.jsonl")


def get_pilot1_sample(
    category: str,
    seed: int,
    n_samples: int,
    data_dir: str = "data/raw",
    cache_dir: str = "data/processed",
) -> list[dict]:
    """Return first n_samples items from the N=20 cache (builds cache if missing).

    Call with identical (category, seed) from both p1_base and p1_aspect
    to guarantee they operate on the exact same reviews.
    """
    path = cache_path(category, seed, cache_dir)
    if not os.path.isfile(path):
        full = sample_review_pairs(category, CACHE_N, seed, data_dir=data_dir)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in full:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[cache] Sampled {len(full)} → {path}")
    else:
        print(f"[cache] Hit → {path}")

    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if n_samples > len(items):
        import warnings
        warnings.warn(
            f"n_samples={n_samples} > cache size {len(items)}; returning all.",
            stacklevel=2,
        )
    return items[:n_samples]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-build pilot1 sample cache")
    parser.add_argument("--category", required=True,
                        help="e.g. Amazon_Fashion, Beauty_and_Personal_Care, Books")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--cache_dir", default="data/processed")
    args = parser.parse_args()

    items = get_pilot1_sample(
        args.category, args.seed, CACHE_N,
        data_dir=args.data_dir, cache_dir=args.cache_dir,
    )
    asins = [it["parent_asin"] for it in items]
    print(f"parent_asins ({len(asins)}): {asins}")
