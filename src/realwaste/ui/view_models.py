from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realwaste.ui.artifact_reader import ArtifactError, ArtifactReader
from realwaste.ui.experiment_registry import ExperimentRecord, ExperimentRegistry


@dataclass(frozen=True, slots=True)
class ExperimentView:
    experiment_id: str
    experiment_name: str
    model_name: str
    augmentation_enabled: bool
    epochs: int
    batch_size: int
    optimizer_name: str
    loss_name: str
    scheduler_name: str
    device: str
    checkpoint_monitor: str
    output_dir: Path
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    history_path: Path | None
    validation_metrics: dict[str, Any] | None
    test_metrics: dict[str, Any] | None
    artifact_error: str | None

    @property
    def has_checkpoint(self) -> bool:
        return self.best_checkpoint is not None or self.last_checkpoint is not None

    @property
    def next_action(self) -> str:
        if not self.has_checkpoint:
            return "Huấn luyện"
        if self.test_metrics is None:
            return "Đánh giá Test"
        return "Sẵn sàng dự đoán"


@dataclass(frozen=True, slots=True)
class ProjectOverview:
    dataset_summary: dict[str, Any] | None
    split_metadata: dict[str, Any] | None
    experiments: tuple[ExperimentView, ...]
    warnings: tuple[str, ...]

    @property
    def trained_experiment_count(self) -> int:
        return sum(view.has_checkpoint for view in self.experiments)

    @property
    def evaluated_experiment_count(self) -> int:
        return sum(view.test_metrics is not None for view in self.experiments)


def build_experiment_view(
    experiment: ExperimentRecord,
    reader: ArtifactReader,
) -> ExperimentView:
    validation_metrics, test_metrics = None, None
    errors: list[str] = []
    for split in ("validation", "test"):
        metrics_path = experiment.existing_metrics_path(split)
        if metrics_path is None:
            continue
        try:
            metrics = reader.read_json_object(metrics_path)
        except ArtifactError as error:
            errors.append(str(error))
            continue
        if split == "validation":
            validation_metrics = metrics
        else:
            test_metrics = metrics

    config = experiment.config
    return ExperimentView(
        experiment_id=experiment.experiment_id,
        experiment_name=experiment.experiment_name,
        model_name=experiment.model_name,
        augmentation_enabled=experiment.augmentation_enabled,
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["data"]["batch_size"]),
        optimizer_name=str(config["training"]["optimizer"]["name"]),
        loss_name=str(config["training"]["loss"]["name"]),
        scheduler_name=str(config["training"]["scheduler"]["name"]),
        device=str(config["device"]),
        checkpoint_monitor=str(config["checkpoint"]["monitor"]),
        output_dir=experiment.output_dir,
        best_checkpoint=(
            experiment.best_checkpoint if experiment.best_checkpoint.is_file() else None
        ),
        last_checkpoint=(
            experiment.last_checkpoint if experiment.last_checkpoint.is_file() else None
        ),
        history_path=experiment.history_path if experiment.history_path.is_file() else None,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        artifact_error="; ".join(errors) or None,
    )


def build_experiment_views(
    registry: ExperimentRegistry,
    reader: ArtifactReader | None = None,
) -> tuple[ExperimentView, ...]:
    artifact_reader = reader or ArtifactReader(registry.project_root)
    return tuple(build_experiment_view(record, artifact_reader) for record in registry)


def build_project_overview(
    registry: ExperimentRegistry,
    reader: ArtifactReader | None = None,
) -> ProjectOverview:
    artifact_reader = reader or ArtifactReader(registry.project_root)
    warnings: list[str] = []

    dataset_summary = _read_optional_object(
        artifact_reader,
        "outputs/eda/dataset_summary.json",
        "Chưa có kết quả kiểm tra dữ liệu.",
        warnings,
    )
    split_metadata = _read_optional_object(
        artifact_reader,
        "data/splits/split_metadata.json",
        "Chưa có thông tin train/validation/test split.",
        warnings,
    )
    experiments = build_experiment_views(registry, artifact_reader)
    warnings.extend(
        f"{view.experiment_id}: {view.artifact_error}"
        for view in experiments
        if view.artifact_error
    )
    return ProjectOverview(
        dataset_summary=dataset_summary,
        split_metadata=split_metadata,
        experiments=experiments,
        warnings=tuple(warnings),
    )


def _read_optional_object(
    reader: ArtifactReader,
    path: str,
    missing_message: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    try:
        result = reader.read_optional_json_object(path)
    except ArtifactError as error:
        warnings.append(str(error))
        return None
    if result is None:
        warnings.append(missing_message)
    return result
