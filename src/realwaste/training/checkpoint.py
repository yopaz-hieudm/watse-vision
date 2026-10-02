from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: dict,
    class_names: list[str] | tuple[str, ...],
    split_fingerprint: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "class_names": list(class_names),
        "model_name": getattr(model, "realwaste_model_name", config["model"]["name"]),
        "weights_id": getattr(model, "realwaste_weights_id", config["model"].get("weights")),
        "split_fingerprint": split_fingerprint,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def validate_resume_checkpoint(
    path: str | Path,
    *,
    expected_split_fingerprint: str,
    expected_experiment_id: str,
    expected_model_name: str,
) -> dict[str, Any]:
    """Validate checkpoint metadata before a background resume job is started."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError(f"resume checkpoint cannot be loaded: {checkpoint_path}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("resume checkpoint must contain a mapping")

    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_metric",
        "config",
        "class_names",
        "model_name",
        "weights_id",
        "split_fingerprint",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"resume checkpoint is missing fields: {', '.join(missing)}")

    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("resume checkpoint config must be a mapping")
    if str(config.get("experiment_id", "")).upper() != expected_experiment_id.upper():
        raise ValueError("resume checkpoint belongs to a different experiment")
    if str(checkpoint["model_name"]) != expected_model_name:
        raise ValueError("resume checkpoint uses a different model")
    if str(checkpoint["split_fingerprint"]) != expected_split_fingerprint:
        raise ValueError("resume checkpoint uses a different split fingerprint")
    if not isinstance(checkpoint["class_names"], list | tuple) or not checkpoint["class_names"]:
        raise ValueError("resume checkpoint has invalid class names")
    if isinstance(checkpoint["epoch"], bool) or not isinstance(checkpoint["epoch"], int):
        raise ValueError("resume checkpoint epoch must be an integer")
    return dict(checkpoint)
