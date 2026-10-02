from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from realwaste.evaluation.metrics import (
    classification_report_rows,
    compute_classification_metrics,
)


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, labels, batch_paths in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
            sample_count += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            probabilities.extend(probs.cpu().numpy())
            paths.extend(batch_paths)
    average_loss = total_loss / sample_count if criterion is not None and sample_count else None
    return {
        "metrics": compute_classification_metrics(y_true, y_pred, loss=average_loss),
        "y_true": y_true,
        "y_pred": y_pred,
        "paths": paths,
        "probabilities": np.asarray(probabilities),
    }


def write_evaluation_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
    class_names: Sequence[str],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = {**metadata, **result["metrics"]}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_rows = classification_report_rows(result["y_true"], result["y_pred"], class_names)
    with (output_dir / "classification_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    probability_columns = [f"probability_{name}" for name in class_names]
    fieldnames = [
        "path",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "confidence",
        "correct",
        *probability_columns,
    ]
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path, truth, prediction, probs in zip(
            result["paths"],
            result["y_true"],
            result["y_pred"],
            result["probabilities"],
            strict=True,
        ):
            row = {
                "path": path,
                "true_label": truth,
                "true_class": class_names[truth],
                "predicted_label": prediction,
                "predicted_class": class_names[prediction],
                "confidence": float(probs[prediction]),
                "correct": truth == prediction,
            }
            row.update(dict(zip(probability_columns, map(float, probs), strict=True)))
            writer.writerow(row)
    return metrics
