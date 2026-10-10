from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realwaste.ui.job_manager import JobRecord
from realwaste.workflow.actions import WorkflowAction


@dataclass(frozen=True, slots=True)
class ExperimentState:
    experiment_id: str
    experiment_name: str
    model_name: str
    augmentation_enabled: bool
    output_dir: Path
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    history_path: Path | None
    best_checkpoint_compatible: bool
    last_checkpoint_compatible: bool
    validation_metrics: Mapping[str, Any] | None
    test_metrics: Mapping[str, Any] | None
    validation_stale: bool
    test_stale: bool
    resume_ready: bool
    test_predictions_ready: bool
    blockers: tuple[str, ...] = ()

    @property
    def has_checkpoint(self) -> bool:
        return self.best_checkpoint is not None or self.last_checkpoint is not None

    @property
    def has_training_artifacts(self) -> bool:
        return (
            self.has_checkpoint
            or self.history_path is not None
            or (self.output_dir / "config.yaml").is_file()
        )

    @property
    def compatible_checkpoint(self) -> tuple[str, Path] | None:
        if self.best_checkpoint is not None and self.best_checkpoint_compatible:
            return "best", self.best_checkpoint
        if self.last_checkpoint is not None and self.last_checkpoint_compatible:
            return "last", self.last_checkpoint
        return None

    @property
    def next_action_label(self) -> str:
        if not self.has_checkpoint:
            return "Huấn luyện"
        if self.compatible_checkpoint is None:
            return "Kiểm tra checkpoint"
        if self.test_metrics is None:
            return "Đánh giá Test"
        return "Sẵn sàng dự đoán"


@dataclass(frozen=True, slots=True)
class WorkflowState:
    data_ready: bool
    eda_ready: bool
    split_ready: bool
    split_fingerprint: str | None
    dataset_summary: Mapping[str, Any] | None
    split_metadata: Mapping[str, Any] | None
    selected_experiment_id: str
    experiments: Mapping[str, ExperimentState]
    active_job: JobRecord | None
    comparison_ready: bool
    inference_checkpoint: Path | None
    allowed_actions: frozenset[WorkflowAction]
    action_blockers: Mapping[WorkflowAction, tuple[str, ...]]
    next_action: WorkflowAction | None
    issues: tuple[str, ...] = ()

    @property
    def selected_experiment(self) -> ExperimentState:
        return self.experiments[self.selected_experiment_id]

    @property
    def trained_experiment_count(self) -> int:
        return sum(
            experiment.compatible_checkpoint is not None for experiment in self.experiments.values()
        )

    @property
    def evaluated_experiment_count(self) -> int:
        return sum(experiment.test_metrics is not None for experiment in self.experiments.values())

    def blockers_for(self, action: WorkflowAction) -> tuple[str, ...]:
        return self.action_blockers.get(action, ())

    def workflow_statuses(self) -> dict[str, str]:
        trained = self.trained_experiment_count
        evaluated = self.evaluated_experiment_count
        return {
            "data": (
                "complete"
                if self.eda_ready and self.split_ready
                else "partial"
                if self.data_ready
                else "pending"
            ),
            "train": (
                "complete"
                if trained == len(self.experiments)
                else "partial"
                if trained
                else "pending"
            ),
            "evaluate": (
                "complete"
                if evaluated == len(self.experiments)
                else "partial"
                if evaluated
                else "pending"
            ),
            "compare": "complete" if self.comparison_ready else "pending",
            "predict": "complete" if self.inference_checkpoint is not None else "pending",
        }
