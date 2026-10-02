from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from realwaste.ui.artifact_reader import ArtifactReader, UnsafeArtifactPathError
from realwaste.ui.experiment_registry import (
    EXPERIMENTS,
    ExperimentRegistry,
    ExperimentRegistryError,
    UnknownExperimentError,
)
from realwaste.ui.view_models import build_project_overview


def _valid_config(experiment_id: str, name: str) -> dict:
    return {
        "experiment_id": experiment_id,
        "experiment_name": name,
        "seed": 42,
        "device": "auto",
        "data": {
            "root": "data/raw/RealWaste",
            "train_csv": "data/splits/train.csv",
            "val_csv": "data/splits/val.csv",
            "test_csv": "data/splits/test.csv",
            "image_size": 224,
            "batch_size": 32,
            "num_workers": 0,
            "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "augmentation": {"enabled": experiment_id in {"E3", "E4"}},
        },
        "model": {
            "name": ("mobilenet_v3_small" if experiment_id in {"E1", "E3"} else "resnet18"),
            "weights": "IMAGENET1K_V1",
            "num_classes": 9,
            "train_backbone": False,
        },
        "training": {
            "epochs": 30,
            "optimizer": {"name": "adamw", "learning_rate": 0.001},
            "loss": {"name": "cross_entropy"},
            "scheduler": {"name": "reduce_lr_on_plateau"},
            "early_stopping": {"monitor": "val_macro_f1"},
        },
        "checkpoint": {"monitor": "val_macro_f1"},
        "output": {"directory": f"outputs/experiments/{name}"},
    }


def _make_project(tmp_path: Path) -> Path:
    names = {
        "E1": "mobilenet_v3_small_baseline",
        "E2": "resnet18_baseline",
        "E3": "mobilenet_v3_small_aug",
        "E4": "resnet18_aug",
    }
    configs = tmp_path / "configs"
    configs.mkdir()
    for experiment_id, relative_path in EXPERIMENTS.items():
        path = tmp_path / relative_path
        path.write_text(
            yaml.safe_dump(_valid_config(experiment_id, names[experiment_id])),
            encoding="utf-8",
        )
    return tmp_path


def test_registry_loads_only_fixed_e1_to_e4(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    registry = ExperimentRegistry(project)

    assert registry.experiment_ids == ("E1", "E2", "E3", "E4")
    assert [item.experiment_id for item in registry.list()] == ["E1", "E2", "E3", "E4"]
    e3 = registry.get("e3")
    assert e3.model_name == "mobilenet_v3_small"
    assert e3.augmentation_enabled is True
    assert e3.output_dir == (project / "outputs/experiments/mobilenet_v3_small_aug").resolve()
    with pytest.raises(UnknownExperimentError, match="chỉ cho phép E1"):
        registry.get("E5")


def test_registry_rejects_config_id_mismatch(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    e1_path = project / EXPERIMENTS["E1"]
    config = yaml.safe_load(e1_path.read_text(encoding="utf-8"))
    config["experiment_id"] = "E2"
    e1_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ExperimentRegistryError, match="phải là E1"):
        ExperimentRegistry(project).get("E1")


def test_registry_rejects_output_outside_project(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    e1_path = project / EXPERIMENTS["E1"]
    config = yaml.safe_load(e1_path.read_text(encoding="utf-8"))
    config["output"]["directory"] = "../outside"
    e1_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(UnsafeArtifactPathError, match="ngoài thư mục dự án"):
        ExperimentRegistry(project).get("E1")


def test_record_finds_checkpoint_history_and_test_metrics(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    record = ExperimentRegistry(project).get("E1")
    record.output_dir.mkdir(parents=True)
    record.last_checkpoint.write_bytes(b"checkpoint")
    record.history_path.write_text("epoch,val_macro_f1\n1,0.5\n", encoding="utf-8")
    metrics = record.metrics_path("test")
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"macro_f1": 0.8}', encoding="utf-8")

    assert record.available_checkpoint() == record.last_checkpoint
    assert record.existing_metrics_path("test") == metrics
    overview = build_project_overview(ExperimentRegistry(project), ArtifactReader(project))
    e1 = overview.experiments[0]
    assert e1.has_checkpoint
    assert e1.history_path == record.history_path
    assert e1.test_metrics == {"macro_f1": 0.8}
    assert e1.next_action == "Sẵn sàng dự đoán"


def test_registry_mapping_cannot_add_arbitrary_experiment(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    arbitrary = deepcopy(dict(EXPERIMENTS))
    arbitrary["E5"] = "configs/evil.yaml"

    with pytest.raises(ExperimentRegistryError, match="exactly E1"):
        ExperimentRegistry(project, experiments=arbitrary)
