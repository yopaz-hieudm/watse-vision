from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def top_confusions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    *,
    limit: int = 20,
) -> list[dict]:
    counts = Counter(
        (truth, prediction)
        for truth, prediction in zip(y_true, y_pred, strict=True)
        if truth != prediction
    )
    return [
        {
            "true_label": truth,
            "true_class": class_names[truth],
            "predicted_label": prediction,
            "predicted_class": class_names[prediction],
            "count": count,
        }
        for (truth, prediction), count in counts.most_common(limit)
    ]


def save_top_confusions(rows: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["true_label", "true_class", "predicted_label", "predicted_class", "count"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
