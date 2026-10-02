from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realwaste.evaluation.metrics import compute_classification_metrics

REQUIRED_COLUMNS = {"path", "true_label", "true_class"}


def load_and_combine_predictions(
    prediction_paths: Sequence[str | Path],
    weights: Sequence[float],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load aligned prediction files and return their weighted probabilities."""
    if len(prediction_paths) < 2:
        raise ValueError("at least two prediction files are required")
    if len(prediction_paths) != len(weights):
        raise ValueError("prediction_paths and weights must have the same length")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must sum to 1.0")

    frames = [pd.read_csv(path) for path in prediction_paths]
    reference = frames[0]
    missing = REQUIRED_COLUMNS - set(reference.columns)
    if missing:
        raise ValueError(f"prediction file is missing columns: {sorted(missing)}")
    probability_columns = [
        column for column in reference.columns if column.startswith("probability_")
    ]
    if not probability_columns:
        raise ValueError("prediction file has no probability columns")

    probabilities = np.zeros((len(reference), len(probability_columns)), dtype=float)
    for frame, weight in zip(frames, weights, strict=True):
        if list(frame.columns) != list(reference.columns):
            raise ValueError("prediction files must have identical columns")
        if not frame["path"].equals(reference["path"]):
            raise ValueError("prediction files must contain paths in the same order")
        if not frame["true_label"].equals(reference["true_label"]):
            raise ValueError("prediction files must contain the same labels")
        probabilities += weight * frame[probability_columns].to_numpy(dtype=float)
    return reference, probabilities, probability_columns


def evaluate_ensemble(
    prediction_paths: Sequence[str | Path],
    weights: Sequence[float],
    output_dir: str | Path,
    *,
    split: str,
    name: str = "weighted_ensemble",
) -> dict[str, Any]:
    reference, probabilities, probability_columns = load_and_combine_predictions(
        prediction_paths, weights
    )
    predictions = probabilities.argmax(axis=1)
    y_true = reference["true_label"].astype(int).to_numpy()
    metrics = {
        "experiment_id": name,
        "experiment_name": name,
        "model": "weighted_ensemble",
        "split": split,
        "members": [str(path) for path in prediction_paths],
        "weights": list(weights),
        **compute_classification_metrics(y_true.tolist(), predictions.tolist()),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    result = reference[["path", "true_label", "true_class"]].copy()
    result["predicted_label"] = predictions
    class_names = [column.removeprefix("probability_") for column in probability_columns]
    result["predicted_class"] = [class_names[index] for index in predictions]
    result["confidence"] = probabilities[np.arange(len(predictions)), predictions]
    result["correct"] = predictions == y_true
    for index, column in enumerate(probability_columns):
        result[column] = probabilities[:, index]
    result.to_csv(output / "predictions.csv", index=False)
    return metrics
