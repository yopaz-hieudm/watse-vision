from __future__ import annotations

import argparse
from pathlib import Path

from realwaste.evaluation.comparison import compare_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mandatory experiments E1-E4")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked = compare_experiments(args.input_dir, args.output_dir)
    print("Experiments ranked by Test Macro-F1:")
    for row in ranked:
        print(f"{row['experiment_id']}: {row['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
