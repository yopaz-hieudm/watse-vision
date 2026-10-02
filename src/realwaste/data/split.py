from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from realwaste.data.inspection import inspect_dataset


def compute_split_fingerprint(paths: list[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def create_stratified_splits(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("split ratios must all be positive")

    valid, invalid = inspect_dataset(dataset_root)
    if invalid:
        raise ValueError("dataset contains invalid images; inspect outputs/eda/invalid_images.csv")
    frame = pd.DataFrame(valid)[["path", "label", "class_name"]]
    train_val, test = train_test_split(
        frame,
        test_size=test_ratio,
        random_state=seed,
        stratify=frame["label"],
    )
    validation_fraction_of_train_val = validation_ratio / (train_ratio + validation_ratio)
    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction_of_train_val,
        random_state=seed,
        stratify=train_val["label"],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    for name, split_frame in (("train", train), ("validation", validation), ("test", test)):
        split_frame.sample(frac=1.0, random_state=seed).to_csv(paths[name], index=False)

    sets = {name: set(pd.read_csv(path)["path"]) for name, path in paths.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"]:
        raise RuntimeError("train split overlaps validation or test")
    if sets["validation"] & sets["test"]:
        raise RuntimeError("validation split overlaps test")

    fingerprint = compute_split_fingerprint(list(paths.values()))
    metadata = {
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "counts": {name: len(values) for name, values in sets.items()},
        "fingerprint": fingerprint,
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
