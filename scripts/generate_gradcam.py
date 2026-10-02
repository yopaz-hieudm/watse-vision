from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from realwaste.evaluation.gradcam import GradCAM, save_gradcam_figure
from realwaste.inference import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for correct and wrong samples")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def select_gradcam_samples(rows: list[dict[str, str]], sample_count: int) -> list[tuple[str, dict]]:
    """Select high/low-confidence correct cases and high-confidence mistakes."""

    if sample_count < 3:
        raise ValueError("--samples must be at least 3 to cover all Grad-CAM groups")
    correct = sorted(
        (row for row in rows if _as_bool(row["correct"])),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )
    wrong = sorted(
        (row for row in rows if not _as_bool(row["correct"])),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )
    if len(correct) < 2 or not wrong:
        raise ValueError("Grad-CAM analysis requires at least two correct and one wrong prediction")

    group_size = max(1, sample_count // 3)
    high_count = min(group_size, len(correct) - 1)
    high_correct = correct[:high_count]
    remaining_correct = correct[high_count:]
    low_correct = list(reversed(remaining_correct))[: min(group_size, len(remaining_correct))]
    wrong_count = max(1, sample_count - len(high_correct) - len(low_correct))
    high_wrong = wrong[:wrong_count]
    return (
        [("correct_high", row) for row in high_correct]
        + [("correct_low", row) for row in low_correct]
        + [("wrong_high", row) for row in high_wrong]
    )


def main() -> None:
    args = parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be at least 3 to cover all Grad-CAM groups")
    predictions_path = args.predictions or args.checkpoint.parent / "predictions.csv"
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_gradcam_samples(rows, args.samples)

    predictor = Predictor(args.checkpoint, args.device)
    dataset_root = Path(predictor.config["data"]["root"])
    output_dir = args.checkpoint.parent / "figures" / "gradcam"
    metadata = []
    with GradCAM(predictor.model) as gradcam:
        for index, (kind, row) in enumerate(selected, start=1):
            image_path = dataset_root / row["path"]
            tensor = predictor.prepare_image(image_path)
            heatmap, predicted, probabilities = gradcam.generate(tensor)
            output_path = output_dir / f"{index:02d}_{kind}_{Path(row['path']).stem}.png"
            save_gradcam_figure(
                image_path,
                heatmap,
                output_path,
                title=f"{kind}: true={row['true_class']} predicted={row['predicted_class']}",
            )
            metadata.append(
                {
                    "kind": kind,
                    "path": row["path"],
                    "true_class": row["true_class"],
                    "predicted_class": predictor.class_names[predicted],
                    "confidence": float(probabilities[predicted]),
                    "figure": str(output_path),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(metadata)} Grad-CAM figures in {output_dir}")


if __name__ == "__main__":
    main()
