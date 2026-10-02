from __future__ import annotations

import argparse
from pathlib import Path

from realwaste.data.split import create_stratified_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic stratified splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/RealWaste"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = create_stratified_splits(
        args.data_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Created splits: {metadata['counts']}")
    print(f"Split fingerprint: {metadata['fingerprint']}")


if __name__ == "__main__":
    main()
