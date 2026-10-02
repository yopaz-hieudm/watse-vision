from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from realwaste.training.progress import ProgressReporter
from realwaste.training.trainer import train_model


def _progress(epoch: int = 1) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "total_epochs": 2,
        "train_loss": 0.82,
        "train_accuracy": 0.71,
        "train_macro_f1": 0.69,
        "val_loss": 0.76,
        "val_accuracy": 0.74,
        "val_macro_f1": 0.72,
        "learning_rate": 0.001,
        "best_metric": 0.72,
    }


def test_progress_reporter_atomically_replaces_json(tmp_path: Path) -> None:
    path = tmp_path / "job" / "progress.json"
    reporter = ProgressReporter(path)
    reporter(_progress())
    reporter(_progress(epoch=2))

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["epoch"] == 2
    assert saved["total_epochs"] == 2
    assert saved["updated_at"].endswith("Z")
    assert list(path.parent.glob("*.tmp")) == []


def test_progress_reporter_rejects_incomplete_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing training progress fields"):
        ProgressReporter(tmp_path / "progress.json")({"epoch": 1})


def test_train_model_calls_optional_progress_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = iter(
        (
            {
                "loss": 0.8,
                "accuracy": 0.7,
                "macro_precision": 0.7,
                "macro_recall": 0.7,
                "macro_f1": 0.68,
                "sample_count": 10,
            },
            {
                "loss": 0.7,
                "accuracy": 0.75,
                "macro_precision": 0.72,
                "macro_recall": 0.71,
                "macro_f1": 0.70,
                "sample_count": 4,
            },
            {
                "loss": 0.6,
                "accuracy": 0.8,
                "macro_precision": 0.78,
                "macro_recall": 0.77,
                "macro_f1": 0.76,
                "sample_count": 10,
            },
            {
                "loss": 0.65,
                "accuracy": 0.79,
                "macro_precision": 0.77,
                "macro_recall": 0.76,
                "macro_f1": 0.75,
                "sample_count": 4,
            },
        )
    )

    def fake_run_epoch(*_args: Any, **_kwargs: Any) -> dict[str, float | int | None]:
        return next(metrics)

    monkeypatch.setattr("realwaste.training.trainer._run_epoch", fake_run_epoch)
    model = nn.Linear(2, 2)
    model.realwaste_model_name = "test_model"  # type: ignore[attr-defined]
    config = {
        "model": {"name": "test_model", "weights": None},
        "training": {
            "epochs": 2,
            "mixed_precision": False,
            "optimizer": {"name": "adamw", "learning_rate": 0.001},
            "loss": {"name": "cross_entropy"},
            "scheduler": {"name": "reduce_lr_on_plateau"},
            "early_stopping": {"patience": 2, "mode": "max"},
        },
        "output": {"directory": str(tmp_path / "experiment")},
    }
    reported: list[dict[str, Any]] = []

    result = train_model(
        model,
        {"train": object(), "validation": object()},
        config,
        torch.device("cpu"),
        class_names=["a", "b"],
        split_fingerprint="fixture-split",
        progress_callback=reported.append,
    )

    assert result == {"best_metric": 0.75, "last_epoch": 1}
    assert reported == [
        {
            "epoch": 1,
            "total_epochs": 2,
            "train_loss": 0.8,
            "train_accuracy": 0.7,
            "train_macro_f1": 0.68,
            "val_loss": 0.7,
            "val_accuracy": 0.75,
            "val_macro_f1": 0.7,
            "learning_rate": 0.001,
            "best_metric": 0.7,
        },
        {
            "epoch": 2,
            "total_epochs": 2,
            "train_loss": 0.6,
            "train_accuracy": 0.8,
            "train_macro_f1": 0.76,
            "val_loss": 0.65,
            "val_accuracy": 0.79,
            "val_macro_f1": 0.75,
            "learning_rate": 0.001,
            "best_metric": 0.75,
        },
    ]


def test_train_model_keeps_callback_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = {
        "loss": 0.7,
        "accuracy": 0.75,
        "macro_precision": 0.72,
        "macro_recall": 0.71,
        "macro_f1": 0.70,
        "sample_count": 4,
    }
    monkeypatch.setattr("realwaste.training.trainer._run_epoch", lambda *_a, **_k: metrics)
    model = nn.Linear(2, 2)
    config = {
        "model": {"name": "test_model", "weights": None},
        "training": {
            "epochs": 1,
            "mixed_precision": False,
            "optimizer": {"name": "adamw", "learning_rate": 0.001},
            "loss": {"name": "cross_entropy"},
            "scheduler": {"name": "reduce_lr_on_plateau"},
            "early_stopping": {"patience": 2, "mode": "max"},
        },
        "output": {"directory": str(tmp_path / "experiment")},
    }
    result = train_model(
        model,
        {"train": object(), "validation": object()},
        config,
        torch.device("cpu"),
        class_names=["a", "b"],
        split_fingerprint="fixture-split",
    )
    assert result["last_epoch"] == 0
