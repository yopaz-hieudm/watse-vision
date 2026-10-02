from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_training_curves(history_path: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(history_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, label in zip(
        axes,
        ("loss", "accuracy", "macro_f1"),
        ("Loss", "Accuracy", "Macro-F1"),
        strict=True,
    ):
        axis.plot(frame["epoch"], frame[f"train_{metric}"], label="Train")
        axis.plot(frame["epoch"], frame[f"val_{metric}"], label="Validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.set_title(f"Train and validation {label}")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
