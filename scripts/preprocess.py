"""CLI entry point for M0/F1 preprocessing.

Usage:
    # Full run (top-20K heavy users + re-5-core):
    python scripts/preprocess.py --category Beauty_and_Personal_Care --n_users 20000
    python scripts/preprocess.py --category Books --n_users 20000

    # Smoke test (top-200 heavy users + re-5-core):
    python scripts/preprocess.py --category Beauty_and_Personal_Care --n_users 200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import run_preprocessing


def main():
    parser = argparse.ArgumentParser(description="M0/F1 preprocessing pipeline")
    parser.add_argument("--category", required=True,
                        help="Category name (e.g. Books, Beauty_and_Personal_Care)")
    parser.add_argument("--n_users", type=int, default=20_000,
                        help="Top-N heaviest users to subsample (default: 20000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data_dir", default="data/raw",
                        help="Raw data directory (default: data/raw)")
    parser.add_argument("--out_dir", default="data/processed",
                        help="Output directory (default: data/processed)")
    args = parser.parse_args()

    run_preprocessing(
        category=args.category,
        n_users=args.n_users,
        seed=args.seed,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
