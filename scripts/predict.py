from __future__ import annotations

import argparse
import json
from pathlib import Path

from realwaste.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single RealWaste image")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor(args.checkpoint, args.device)
    print(json.dumps(predictor.predict(args.image, args.top_k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
