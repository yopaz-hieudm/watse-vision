from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MODELS = {"mobilenet_v3_small", "resnet18"}


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _require(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        current = current[part]
    return current


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "experiment_id",
        "experiment_name",
        "seed",
        "device",
        "data.root",
        "data.train_csv",
        "data.val_csv",
        "data.test_csv",
        "data.image_size",
        "data.batch_size",
        "data.num_workers",
        "data.split.train",
        "data.split.validation",
        "data.split.test",
        "data.augmentation.enabled",
        "model.name",
        "model.weights",
        "model.num_classes",
        "model.train_backbone",
        "training.epochs",
        "training.optimizer.name",
        "training.optimizer.learning_rate",
        "training.loss.name",
        "training.scheduler.name",
        "training.early_stopping.monitor",
        "checkpoint.monitor",
        "output.directory",
    )
    for key in required:
        _require(config, key)

    model_name = str(_require(config, "model.name"))
    if model_name not in SUPPORTED_MODELS:
        raise ConfigError(
            f"Unsupported model {model_name!r}; expected one of {sorted(SUPPORTED_MODELS)}"
        )

    if int(_require(config, "model.num_classes")) != 9:
        raise ConfigError("model.num_classes must be 9 for RealWaste")

    split = _require(config, "data.split")
    split_sum = float(split["train"]) + float(split["validation"]) + float(split["test"])
    if abs(split_sum - 1.0) > 1e-9:
        raise ConfigError(f"data.split ratios must sum to 1.0, got {split_sum}")
    if any(float(split[name]) <= 0 for name in ("train", "validation", "test")):
        raise ConfigError("all data.split ratios must be positive")

    if int(_require(config, "data.image_size")) <= 0:
        raise ConfigError("data.image_size must be positive")
    if int(_require(config, "data.batch_size")) <= 0:
        raise ConfigError("data.batch_size must be positive")
    if int(_require(config, "training.epochs")) <= 0:
        raise ConfigError("training.epochs must be positive")

    if str(_require(config, "training.optimizer.name")).lower() != "adamw":
        raise ConfigError("the required optimizer for E1-E4 is adamw")
    if str(_require(config, "training.loss.name")).lower() != "cross_entropy":
        raise ConfigError("the required loss for E1-E4 is cross_entropy")
    if str(_require(config, "checkpoint.monitor")) != "val_macro_f1":
        raise ConfigError("checkpoint.monitor must be val_macro_f1")

    return config


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    return validate_config(loaded)


def save_resolved_config(config: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(copy.deepcopy(config), handle, sort_keys=False, allow_unicode=True)
    return output_path
