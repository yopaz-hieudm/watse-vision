from __future__ import annotations

import csv
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from realwaste.constants import CLASS_NAMES, EXPECTED_IMAGE_COUNT, IMAGE_EXTENSIONS


def iter_class_images(dataset_root: str | Path):
    root = Path(dataset_root)
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path, label, class_name


def inspect_dataset(dataset_root: str | Path) -> tuple[list[dict], list[dict]]:
    root = Path(dataset_root)
    valid: list[dict] = []
    invalid: list[dict] = []
    for path, label, class_name in iter_class_images(root):
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("image has non-positive dimensions")
            valid.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label": label,
                    "class_name": class_name,
                    "width": width,
                    "height": height,
                }
            )
        except (OSError, ValueError, UnidentifiedImageError) as error:
            invalid.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "class_name": class_name,
                    "error": str(error),
                }
            )
    return valid, invalid


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_distribution(counts: Counter, output_dir: Path) -> None:
    rows = [{"class_name": name, "count": counts[name]} for name in CLASS_NAMES]
    _write_csv(output_dir / "class_distribution.csv", rows, ["class_name", "count"])

    figure, axis = plt.subplots(figsize=(11, 5.5))
    values = [counts[name] for name in CLASS_NAMES]
    axis.bar(range(len(CLASS_NAMES)), values, color="#1F4E78")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    axis.set_ylabel("Number of images")
    axis.set_title("RealWaste class distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(figure)


def _save_sample_grid(
    dataset_root: Path,
    valid_rows: list[dict],
    output_path: Path,
    *,
    samples_per_class: int = 4,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    by_class = {name: [] for name in CLASS_NAMES}
    for row in valid_rows:
        by_class[row["class_name"]].append(row)

    figure, axes = plt.subplots(
        len(CLASS_NAMES), samples_per_class, figsize=(3 * samples_per_class, 2.6 * len(CLASS_NAMES))
    )
    for row_index, class_name in enumerate(CLASS_NAMES):
        candidates = by_class[class_name]
        chosen = rng.sample(candidates, k=min(samples_per_class, len(candidates)))
        for column_index in range(samples_per_class):
            axis = axes[row_index, column_index]
            axis.axis("off")
            if column_index < len(chosen):
                with Image.open(dataset_root / chosen[column_index]["path"]) as image:
                    axis.imshow(image.convert("RGB"))
            if column_index == 0:
                axis.set_title(class_name, loc="left", fontsize=10, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_eda_outputs(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    strict: bool = True,
) -> dict:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid, invalid = inspect_dataset(dataset_root)
    counts = Counter(row["class_name"] for row in valid)

    _save_distribution(counts, output_dir)
    _save_sample_grid(dataset_root, valid, output_dir / "sample_grid.png", seed=seed)
    _write_csv(
        output_dir / "invalid_images.csv",
        invalid,
        ["path", "class_name", "error"],
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset_root),
        "total_files": len(valid) + len(invalid),
        "valid_images": len(valid),
        "invalid_images": len(invalid),
        "number_of_classes": sum(1 for name in CLASS_NAMES if counts[name] > 0),
        "class_counts": {name: counts[name] for name in CLASS_NAMES},
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if strict:
        if summary["number_of_classes"] != len(CLASS_NAMES):
            raise ValueError(f"Expected 9 non-empty classes, got {summary['number_of_classes']}")
        if summary["valid_images"] != EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_IMAGE_COUNT} valid images, got {summary['valid_images']}"
            )
    return summary
