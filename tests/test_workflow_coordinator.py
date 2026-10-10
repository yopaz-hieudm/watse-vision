from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from realwaste.constants import CLASS_NAMES, EXPECTED_IMAGE_COUNT
from realwaste.ui.artifact_reader import ArtifactReader
from realwaste.ui.experiment_registry import EXPERIMENTS, ExperimentRegistry
from realwaste.ui.job_manager import JobRecord
from realwaste.workflow import (
    WorkflowAction,
    WorkflowActionBlockedError,
    WorkflowCoordinator,
)


def _config(experiment_id: str, name: str) -> dict[str, Any]:
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
            "name": "mobilenet_v3_small" if experiment_id in {"E1", "E3"} else "resnet18",
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


def _project(tmp_path: Path, *, ready: bool = False) -> Path:
    names = {
        "E1": "mobilenet_v3_small_baseline",
        "E2": "resnet18_baseline",
        "E3": "mobilenet_v3_small_aug",
        "E4": "resnet18_aug",
    }
    for experiment_id, relative in EXPERIMENTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(_config(experiment_id, names[experiment_id])),
            encoding="utf-8",
        )
    if ready:
        _make_data_ready(tmp_path)
    return tmp_path


def _make_data_ready(root: Path, fingerprint: str = "split-v1") -> None:
    dataset = root / "data/raw/RealWaste"
    for class_name in CLASS_NAMES:
        (dataset / class_name).mkdir(parents=True, exist_ok=True)

    summary = root / "outputs/eda/dataset_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        (f'{{"valid_images": {EXPECTED_IMAGE_COUNT}, "number_of_classes": {len(CLASS_NAMES)}}}'),
        encoding="utf-8",
    )

    splits = root / "data/splits"
    splits.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (splits / f"{name}.csv").write_text("path,label,class_name\n", encoding="utf-8")
    (splits / "split_metadata.json").write_text(
        f'{{"fingerprint": "{fingerprint}", "counts": {{}}}}',
        encoding="utf-8",
    )


@dataclass
class _FakeJobManager:
    active: JobRecord | None = None
    calls: list[tuple[str, str | None, dict[str, Any]]] = field(default_factory=list)

    def get_active_job(self) -> JobRecord | None:
        return self.active

    def start_job(
        self,
        kind: str,
        experiment_id: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> JobRecord:
        self.calls.append((kind, experiment_id, dict(args or {})))
        return _job(kind=kind, experiment_id=experiment_id)

    def stop_job(self, job_id: str, *, force: bool = False) -> JobRecord:
        del job_id, force
        return _job(status="stopped")


def _job(
    *,
    kind: str = "train",
    experiment_id: str | None = "E1",
    status: str = "running",
) -> JobRecord:
    return JobRecord(
        job_id="20261009_120000_E1_train",
        kind=kind,
        experiment_id=experiment_id,
        command=("python", "script.py"),
        pid=123,
        status=status,
        created_at="2026-10-09T12:00:00+00:00",
        started_at="2026-10-09T12:00:00+00:00",
        finished_at=None,
        exit_code=None,
        cwd="/tmp/project",
    )


def _coordinator(root: Path, manager: _FakeJobManager | None = None) -> WorkflowCoordinator:
    registry = ExperimentRegistry(root)
    return WorkflowCoordinator(
        root,
        registry=registry,
        reader=ArtifactReader(root),
        job_manager=manager or _FakeJobManager(),  # type: ignore[arg-type]
    )


def _checkpoint(root: Path, experiment_id: str, *, fingerprint: str = "split-v1") -> Path:
    record = ExperimentRegistry(root).get(experiment_id)
    record.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = record.best_checkpoint
    torch.save(
        {
            "config": record.config,
            "model_name": record.model_name,
            "split_fingerprint": fingerprint,
        },
        checkpoint,
    )
    return checkpoint


def _test_metrics(root: Path, experiment_id: str, checkpoint: Path) -> None:
    record = ExperimentRegistry(root).get(experiment_id)
    metrics = record.metrics_path("test")
    metrics.parent.mkdir(parents=True, exist_ok=True)
    relative_checkpoint = checkpoint.relative_to(root).as_posix()
    metrics.write_text(
        (
            "{"
            f'"experiment_id": "{experiment_id}", '
            '"split": "test", '
            '"split_fingerprint": "split-v1", '
            f'"checkpoint_path": "{relative_checkpoint}", '
            '"macro_f1": 0.8'
            "}"
        ),
        encoding="utf-8",
    )
    record.predictions_path("test").write_text(
        "path,true_label,predicted_label\n", encoding="utf-8"
    )


def test_missing_dataset_recommends_download_and_blocks_training(tmp_path: Path) -> None:
    coordinator = _coordinator(_project(tmp_path))

    state = coordinator.get_state("E1")

    assert not state.data_ready
    assert state.next_action is WorkflowAction.DOWNLOAD_DATA
    assert WorkflowAction.DOWNLOAD_DATA in state.allowed_actions
    assert WorkflowAction.START_TRAINING not in state.allowed_actions


def test_ready_project_recommends_training_for_selected_experiment(tmp_path: Path) -> None:
    coordinator = _coordinator(_project(tmp_path, ready=True))

    state = coordinator.get_state("E2")

    assert state.data_ready and state.eda_ready and state.split_ready
    assert state.selected_experiment_id == "E2"
    assert state.next_action is WorkflowAction.START_TRAINING
    assert WorkflowAction.START_TRAINING in state.allowed_actions


def test_current_checkpoint_enables_evaluation_and_inference(tmp_path: Path) -> None:
    root = _project(tmp_path, ready=True)
    checkpoint = _checkpoint(root, "E1")
    manager = _FakeJobManager()
    coordinator = _coordinator(root, manager)

    state = coordinator.get_state("E1")

    assert state.selected_experiment.best_checkpoint_compatible
    assert state.inference_checkpoint == checkpoint
    assert WorkflowAction.EVALUATE_TEST in state.allowed_actions

    coordinator.execute(WorkflowAction.EVALUATE_TEST, experiment_id="E1")

    assert manager.calls == [("evaluate", "E1", {"checkpoint": "best", "split": "test"})]


def test_changed_split_marks_checkpoint_stale_and_blocks_evaluation(tmp_path: Path) -> None:
    root = _project(tmp_path, ready=True)
    _checkpoint(root, "E1", fingerprint="old-split")
    coordinator = _coordinator(root)

    state = coordinator.get_state("E1")

    assert not state.selected_experiment.best_checkpoint_compatible
    assert state.inference_checkpoint is None
    assert WorkflowAction.EVALUATE_TEST not in state.allowed_actions
    with pytest.raises(WorkflowActionBlockedError, match="Không có checkpoint"):
        coordinator.execute(WorkflowAction.EVALUATE_TEST, experiment_id="E1")


def test_active_job_blocks_new_heavy_action(tmp_path: Path) -> None:
    root = _project(tmp_path, ready=True)
    manager = _FakeJobManager(active=_job())
    coordinator = _coordinator(root, manager)

    state = coordinator.get_state("E1")

    assert not state.allowed_actions
    assert state.next_action is None
    assert "đang chạy" in state.blockers_for(WorkflowAction.START_TRAINING)[0]


def test_compare_requires_four_current_test_results(tmp_path: Path) -> None:
    root = _project(tmp_path, ready=True)
    for experiment_id in EXPERIMENTS:
        checkpoint = _checkpoint(root, experiment_id)
        _test_metrics(root, experiment_id, checkpoint)
    coordinator = _coordinator(root)

    state = coordinator.get_state("E1")

    assert state.evaluated_experiment_count == 4
    assert WorkflowAction.COMPARE_EXPERIMENTS in state.allowed_actions
    assert state.next_action is WorkflowAction.COMPARE_EXPERIMENTS


def test_inference_never_switches_to_another_experiment(tmp_path: Path) -> None:
    root = _project(tmp_path, ready=True)
    checkpoint = _checkpoint(root, "E1")
    _test_metrics(root, "E1", checkpoint)
    coordinator = _coordinator(root)
    assert coordinator.get_state("E2").inference_checkpoint is None
    second = _checkpoint(root, "E2")
    assert coordinator.get_state("E2").inference_checkpoint == second


def test_changed_config_is_detected_without_restarting_application(tmp_path: Path) -> None:
    root = _project(tmp_path, ready=True)
    _checkpoint(root, "E1")
    coordinator = _coordinator(root)
    assert coordinator.get_state("E1").inference_checkpoint is not None
    path = root / EXPERIMENTS["E1"]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["training"]["epochs"] = 31
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    state = coordinator.get_state("E1")
    assert state.inference_checkpoint is None
    assert state.selected_experiment.has_training_artifacts
    assert WorkflowAction.RESUME_TRAINING not in state.allowed_actions
