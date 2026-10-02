from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from realwaste.config import ConfigError, load_config, validate_config


@pytest.mark.parametrize(
    "name",
    [
        "mobilenet_v3_small_baseline.yaml",
        "mobilenet_v3_small_aug.yaml",
        "mobilenet_v3_small_finetune.yaml",
        "resnet18_baseline.yaml",
        "resnet18_aug.yaml",
    ],
)
def test_shipped_configs_are_valid(name: str) -> None:
    config = load_config(Path("configs") / name)
    assert config["model"]["num_classes"] == 9
    assert config["checkpoint"]["monitor"] == "val_macro_f1"


def test_rejects_invalid_split() -> None:
    config = yaml.safe_load(Path("configs/mobilenet_v3_small_baseline.yaml").read_text())
    config["data"]["split"]["test"] = 0.5
    with pytest.raises(ConfigError, match="sum to 1.0"):
        validate_config(config)
