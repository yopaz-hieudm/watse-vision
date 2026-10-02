from __future__ import annotations

import argparse
from pathlib import Path

from realwaste.data.inspection import write_eda_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect RealWaste and produce EDA artifacts")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/RealWaste"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_eda_outputs(
        args.data_dir,
        args.output_dir,
        seed=args.seed,
        strict=not args.no_strict,
    )
    print(
        f"EDA complete: {summary['valid_images']} valid, {summary['invalid_images']} invalid images"
    )


if __name__ == "__main__":
    main()
