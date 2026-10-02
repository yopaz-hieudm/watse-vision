from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    loss: float | None = None,
) -> dict[str, float | int | None]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "loss": None if loss is None else float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "sample_count": int(len(y_true)),
    }


def classification_report_rows(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> list[dict]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict] = []
    for class_name in class_names:
        values = report[class_name]
        rows.append(
            {
                "class_name": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return rows


def probabilities_to_predictions(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    return probabilities.argmax(axis=1)
