"""CLI entry point for M0/F1 preprocessing.

Usage:
    # Full run (EXP3RT-style random 50K → 5-core cascade → users≈items):
    python scripts/preprocess.py --category Beauty_and_Personal_Care --n_users 50000
    python scripts/preprocess.py --category Books --n_users 50000

    # Full run (legacy heavy-user mode for ablation):
    python scripts/preprocess.py --category Beauty_and_Personal_Care --n_users 20000 --sampling heavy

    # Smoke test (random 500 users + re-5-core):
    python scripts/preprocess.py --category Beauty_and_Personal_Care --n_users 500
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
    parser.add_argument("--n_users", type=int, default=50_000,
                        help="Users to subsample before re-5-core (default: 50000). "
                             "With --sampling random, 5-core cascade shrinks this to the "
                             "final balanced count (target: users≈items).")
    parser.add_argument("--sampling", default="random", choices=["random", "heavy"],
                        help="Sampling strategy: 'random' (default, EXP3RT-style, "
                             "produces users≈items) or 'heavy' (top-N by activity)")
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
        sampling_strategy=args.sampling,
    )


if __name__ == "__main__":
    main()
