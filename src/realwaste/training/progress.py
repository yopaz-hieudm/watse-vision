from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_PROGRESS_FIELDS = (
    "epoch",
    "total_epochs",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "learning_rate",
    "best_metric",
)


def _validate_progress(progress: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_PROGRESS_FIELDS if field not in progress]
    if missing:
        raise ValueError(f"missing training progress fields: {', '.join(missing)}")

    epoch = progress["epoch"]
    total_epochs = progress["total_epochs"]
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
        raise TypeError("total_epochs must be an integer")
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be between 1 and total_epochs")

    for field in REQUIRED_PROGRESS_FIELDS[2:]:
        value = progress[field]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{field} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON using an atomic replace so readers never see partial content."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


class ProgressReporter:
    """Persist the most recent training epoch for UI and external consumers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def report(self, progress: Mapping[str, Any]) -> Path:
        _validate_progress(progress)
        payload = dict(progress)
        payload["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return atomic_write_json(self.path, payload)

    def __call__(self, progress: Mapping[str, Any]) -> None:
        self.report(progress)
