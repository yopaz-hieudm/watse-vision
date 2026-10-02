from __future__ import annotations

import argparse
from pathlib import Path

from realwaste.evaluation.ensemble import evaluate_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a weighted prediction ensemble")
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("WEIGHT", "PREDICTIONS_CSV"),
        required=True,
        help="Repeat for each member; weights must sum to 1",
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="weighted_ensemble")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = [float(item[0]) for item in args.input]
    prediction_paths = [Path(item[1]) for item in args.input]
    metrics = evaluate_ensemble(
        prediction_paths,
        weights,
        args.output_dir,
        split=args.split,
        name=args.name,
    )
    print(metrics)


if __name__ == "__main__":
    main()
