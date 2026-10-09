from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from realwaste.constants import CLASS_NAMES, EXPECTED_IMAGE_COUNT
from realwaste.ui.artifact_reader import ArtifactError, ArtifactReader
from realwaste.ui.experiment_registry import ExperimentRecord, ExperimentRegistry
from realwaste.ui.job_manager import JobManager, JobRecord
from realwaste.workflow.actions import WorkflowAction
from realwaste.workflow.errors import WorkflowActionBlockedError, WorkflowStateError
from realwaste.workflow.policies import WorkflowFacts, evaluate_policy
from realwaste.workflow.state import ExperimentState, WorkflowState


@dataclass(frozen=True, slots=True)
class _CheckpointIdentity:
    experiment_id: str
    model_name: str
    split_fingerprint: str
    config_signature: str


class WorkflowCoordinator:
    """Coordinate project-level actions while leaving process execution to JobManager."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        registry: ExperimentRegistry | None = None,
        reader: ArtifactReader | None = None,
        job_manager: JobManager | None = None,
    ) -> None:
        default_root = Path(__file__).resolve().parents[3]
        self.project_root = Path(project_root or default_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"Project root does not exist: {self.project_root}")
        self.registry = registry or ExperimentRegistry(self.project_root)
        self.reader = reader or ArtifactReader(self.project_root)
        self.job_manager = job_manager or JobManager(self.project_root)

    def get_state(self, selected_experiment_id: str | None = None) -> WorkflowState:
        selected = self.registry.get(selected_experiment_id or self.registry.experiment_ids[0])
        issues: list[str] = []

        dataset_summary = self._read_optional_object(
            "outputs/eda/dataset_summary.json",
            issues,
        )
        split_metadata = self._read_optional_object(
            "data/splits/split_metadata.json",
            issues,
        )
        data_ready = self._dataset_ready()
        eda_ready = self._eda_ready(dataset_summary, data_ready)
        split_fingerprint = self._split_fingerprint(split_metadata)
        split_ready = self._split_ready(split_metadata, eda_ready)

        experiment_states = {
            record.experiment_id: self._build_experiment_state(
                record,
                split_fingerprint if split_ready else None,
                issues,
            )
            for record in self.registry
        }

        try:
            active_job = self.job_manager.get_active_job()
        except Exception as error:
            raise WorkflowStateError(f"Không thể đọc trạng thái job: {error}") from error

        comparison_ready = self._comparison_ready(
            tuple(experiment_states.values()),
            split_fingerprint,
        )
        inference_checkpoint = self._choose_inference_checkpoint(
            selected.experiment_id,
            tuple(experiment_states.values()),
        )
        facts = WorkflowFacts(
            data_ready=data_ready,
            eda_ready=eda_ready,
            split_ready=split_ready,
            selected_experiment=experiment_states[selected.experiment_id],
            experiments=tuple(experiment_states.values()),
            active_job=active_job,
            comparison_ready=comparison_ready,
        )
        decision = evaluate_policy(facts)

        return WorkflowState(
            data_ready=data_ready,
            eda_ready=eda_ready,
            split_ready=split_ready,
            split_fingerprint=split_fingerprint,
            dataset_summary=dataset_summary,
            split_metadata=split_metadata,
            selected_experiment_id=selected.experiment_id,
            experiments=experiment_states,
            active_job=active_job,
            comparison_ready=comparison_ready,
            inference_checkpoint=inference_checkpoint,
            allowed_actions=decision.allowed_actions,
            action_blockers=decision.blockers,
            next_action=decision.next_action,
            issues=tuple(dict.fromkeys(issues)),
        )

    def execute(
        self,
        action: WorkflowAction | str,
        *,
        experiment_id: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        normalized = self._normalize_action(action)
        state = self.get_state(experiment_id)
        if normalized not in state.allowed_actions:
            raise WorkflowActionBlockedError(normalized, state.blockers_for(normalized))

        supplied_options = dict(options or {})
        selected_id = state.selected_experiment_id if normalized.requires_experiment else None

        if normalized is WorkflowAction.DOWNLOAD_DATA:
            return self.job_manager.start_job("download", args={"verify_only": False})
        if normalized is WorkflowAction.VERIFY_DATA:
            return self.job_manager.start_job("download", args={"verify_only": True})
        if normalized is WorkflowAction.RUN_EDA:
            return self.job_manager.start_job("eda", args=supplied_options)
        if normalized is WorkflowAction.CREATE_SPLIT:
            return self.job_manager.start_job("split", args=supplied_options)
        if normalized is WorkflowAction.START_TRAINING:
            return self.job_manager.start_job("train", selected_id)
        if normalized is WorkflowAction.RESUME_TRAINING:
            return self.job_manager.start_job("train", selected_id, {"resume": True})
        if normalized in {
            WorkflowAction.EVALUATE_VALIDATION,
            WorkflowAction.EVALUATE_TEST,
        }:
            split = "validation" if normalized is WorkflowAction.EVALUATE_VALIDATION else "test"
            checkpoint_kind = self._checkpoint_kind(state, supplied_options)
            return self.job_manager.start_job(
                "evaluate",
                selected_id,
                {"checkpoint": checkpoint_kind, "split": split},
            )
        if normalized is WorkflowAction.GENERATE_GRADCAM:
            checkpoint_kind = self._checkpoint_kind(state, supplied_options)
            samples = supplied_options.get("samples", 20)
            return self.job_manager.start_job(
                "gradcam",
                selected_id,
                {"checkpoint": checkpoint_kind, "samples": samples},
            )
        if normalized is WorkflowAction.COMPARE_EXPERIMENTS:
            return self.job_manager.start_job("compare")
        raise AssertionError(f"Unhandled workflow action: {normalized}")

    def get_recent_job(self) -> JobRecord | None:
        jobs = self.job_manager.list_jobs(limit=1)
        return jobs[0] if jobs else None

    def stop_job(self, job_id: str, *, force: bool = False) -> JobRecord:
        active = self.get_state().active_job
        if active is None:
            raise WorkflowStateError("Không có job đang chạy để dừng")
        if active.job_id != job_id:
            raise WorkflowStateError(
                f"Job {job_id} không phải job đang chạy hiện tại ({active.job_id})"
            )
        return self.job_manager.stop_job(job_id, force=force)

    def stop_active_job(self, *, force: bool = False) -> JobRecord:
        active = self.get_state().active_job
        if active is None:
            raise WorkflowStateError("Không có job đang chạy để dừng")
        return self.stop_job(active.job_id, force=force)

    @staticmethod
    def _normalize_action(action: WorkflowAction | str) -> WorkflowAction:
        if isinstance(action, WorkflowAction):
            return action
        try:
            return WorkflowAction(str(action))
        except ValueError as error:
            raise ValueError(f"Workflow action không hợp lệ: {action!r}") from error

    @staticmethod
    def _checkpoint_kind(
        state: WorkflowState,
        options: Mapping[str, Any],
    ) -> str:
        selected = state.selected_experiment
        requested = options.get("checkpoint")
        compatible = selected.compatible_checkpoint
        if requested is None:
            if compatible is None:
                raise WorkflowActionBlockedError(
                    WorkflowAction.EVALUATE_TEST,
                    ("Không có checkpoint tương thích",),
                )
            return compatible[0]
        if requested not in {"best", "last"}:
            raise ValueError("checkpoint phải là best hoặc last")
        is_compatible = (
            selected.best_checkpoint_compatible
            if requested == "best"
            else selected.last_checkpoint_compatible
        )
        if not is_compatible:
            raise WorkflowActionBlockedError(
                WorkflowAction.EVALUATE_TEST,
                (f"{requested}.pt không tương thích với workflow hiện tại",),
            )
        return str(requested)

    def _build_experiment_state(
        self,
        record: ExperimentRecord,
        split_fingerprint: str | None,
        issues: list[str],
    ) -> ExperimentState:
        current_config_signature = _config_signature(record.config)
        best_path = record.best_checkpoint if record.best_checkpoint.is_file() else None
        last_path = record.last_checkpoint if record.last_checkpoint.is_file() else None
        best_compatible, best_error = self._checkpoint_compatible(
            best_path,
            record,
            split_fingerprint,
            current_config_signature,
        )
        last_compatible, last_error = self._checkpoint_compatible(
            last_path,
            record,
            split_fingerprint,
            current_config_signature,
        )
        experiment_blockers = tuple(
            message for message in (last_error, best_error) if message is not None
        )
        issues.extend(f"{record.experiment_id}: {message}" for message in experiment_blockers)

        compatible_paths = {
            path.resolve()
            for path, compatible in (
                (best_path, best_compatible),
                (last_path, last_compatible),
            )
            if path is not None and compatible
        }
        validation_metrics, validation_stale = self._current_metrics(
            record,
            "validation",
            split_fingerprint,
            compatible_paths,
            issues,
        )
        test_metrics, test_stale = self._current_metrics(
            record,
            "test",
            split_fingerprint,
            compatible_paths,
            issues,
        )

        return ExperimentState(
            experiment_id=record.experiment_id,
            experiment_name=record.experiment_name,
            model_name=record.model_name,
            augmentation_enabled=record.augmentation_enabled,
            output_dir=record.output_dir,
            best_checkpoint=best_path,
            last_checkpoint=last_path,
            history_path=record.history_path if record.history_path.is_file() else None,
            best_checkpoint_compatible=best_compatible,
            last_checkpoint_compatible=last_compatible,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            validation_stale=validation_stale,
            test_stale=test_stale,
            resume_ready=last_path is not None and last_compatible,
            test_predictions_ready=(
                test_metrics is not None and record.predictions_path("test").is_file()
            ),
            blockers=experiment_blockers,
        )

    def _checkpoint_compatible(
        self,
        path: Path | None,
        record: ExperimentRecord,
        split_fingerprint: str | None,
        config_signature: str,
    ) -> tuple[bool, str | None]:
        if path is None:
            return False, None
        if split_fingerprint is None:
            return False, f"{path.name} chưa thể xác minh vì split chưa sẵn sàng"
        try:
            stat = path.stat()
            identity = _read_checkpoint_identity(
                str(path.resolve()),
                stat.st_mtime_ns,
                stat.st_size,
            )
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
            return False, f"{path.name} không đọc được: {error}"

        if identity.experiment_id.upper() != record.experiment_id:
            return False, f"{path.name} thuộc experiment khác"
        if identity.model_name != record.model_name:
            return False, f"{path.name} dùng model khác config hiện tại"
        if identity.split_fingerprint != split_fingerprint:
            return False, f"{path.name} dùng split fingerprint cũ"
        if identity.config_signature != config_signature:
            return False, f"{path.name} không khớp config hiện tại"
        return True, None

    def _current_metrics(
        self,
        record: ExperimentRecord,
        split: str,
        split_fingerprint: str | None,
        compatible_checkpoints: set[Path],
        issues: list[str],
    ) -> tuple[dict[str, Any] | None, bool]:
        metrics_path = record.existing_metrics_path(split)
        if metrics_path is None:
            return None, False
        try:
            metrics = self.reader.read_json_object(metrics_path)
        except ArtifactError as error:
            issues.append(f"{record.experiment_id} {split}: {error}")
            return None, True

        reason: str | None = None
        if split_fingerprint is None:
            reason = "split hiện tại chưa sẵn sàng"
        elif str(metrics.get("split", "")).lower() != split:
            reason = f"metrics không khai báo split {split}"
        elif str(metrics.get("split_fingerprint", "")) != split_fingerprint:
            reason = "metrics dùng split fingerprint cũ"
        else:
            checkpoint_value = metrics.get("checkpoint_path")
            if not isinstance(checkpoint_value, str) or not checkpoint_value:
                reason = "metrics thiếu checkpoint_path"
            else:
                checkpoint = Path(checkpoint_value).expanduser()
                if not checkpoint.is_absolute():
                    checkpoint = self.project_root / checkpoint
                checkpoint = checkpoint.resolve()
                if checkpoint not in compatible_checkpoints:
                    reason = "metrics tham chiếu checkpoint không tương thích"
                elif metrics_path.stat().st_mtime_ns < checkpoint.stat().st_mtime_ns:
                    reason = "metrics cũ hơn checkpoint"

        if reason is not None:
            issues.append(f"{record.experiment_id} {split}: {reason}")
            return None, True
        return metrics, False

    def _read_optional_object(
        self,
        path: str,
        issues: list[str],
    ) -> dict[str, Any] | None:
        try:
            return self.reader.read_optional_json_object(path)
        except ArtifactError as error:
            issues.append(str(error))
            return None

    def _dataset_ready(self) -> bool:
        root = self.project_root / "data/raw/RealWaste"
        return root.is_dir() and all((root / class_name).is_dir() for class_name in CLASS_NAMES)

    @staticmethod
    def _eda_ready(
        summary: Mapping[str, Any] | None,
        data_ready: bool,
    ) -> bool:
        if not data_ready or summary is None:
            return False
        return summary.get("valid_images") == EXPECTED_IMAGE_COUNT and summary.get(
            "number_of_classes"
        ) == len(CLASS_NAMES)

    def _split_ready(
        self,
        metadata: Mapping[str, Any] | None,
        prerequisite_ready: bool,
    ) -> bool:
        if not prerequisite_ready or self._split_fingerprint(metadata) is None:
            return False
        return all(
            (self.project_root / f"data/splits/{name}.csv").is_file()
            for name in ("train", "val", "test")
        )

    @staticmethod
    def _split_fingerprint(metadata: Mapping[str, Any] | None) -> str | None:
        if metadata is None:
            return None
        value = metadata.get("fingerprint")
        return value if isinstance(value, str) and value.strip() else None

    def _comparison_ready(
        self,
        experiments: tuple[ExperimentState, ...],
        split_fingerprint: str | None,
    ) -> bool:
        if split_fingerprint is None or not all(
            item.test_metrics is not None for item in experiments
        ):
            return False
        comparison_dir = self.project_root / "outputs/comparison"
        required = (
            comparison_dir / "experiment_summary.csv",
            comparison_dir / "experiment_summary.md",
            comparison_dir / "metrics_comparison.png",
            comparison_dir / "augmentation_impact.csv",
        )
        if not all(path.is_file() for path in required):
            return False
        comparison_mtime = min(path.stat().st_mtime_ns for path in required)
        metrics_paths = [
            self.registry.get(item.experiment_id).existing_metrics_path("test")
            for item in experiments
        ]
        return all(
            path is not None and path.stat().st_mtime_ns <= comparison_mtime
            for path in metrics_paths
        )

    @staticmethod
    def _choose_inference_checkpoint(
        selected_experiment_id: str,
        experiments: tuple[ExperimentState, ...],
    ) -> Path | None:
        ranked: list[tuple[float, bool, int, Path]] = []
        for experiment in experiments:
            compatible = experiment.compatible_checkpoint
            if compatible is None:
                continue
            kind, path = compatible
            score = float("-inf")
            if experiment.test_metrics is not None:
                try:
                    score = float(experiment.test_metrics.get("macro_f1", float("-inf")))
                except (TypeError, ValueError):
                    score = float("-inf")
            ranked.append(
                (
                    score,
                    experiment.experiment_id == selected_experiment_id,
                    kind == "best",
                    path,
                )
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return ranked[0][3]


def _config_signature(config: Mapping[str, Any]) -> str:
    relevant = {
        key: config.get(key)
        for key in (
            "experiment_id",
            "experiment_name",
            "seed",
            "data",
            "model",
            "training",
            "checkpoint",
        )
    }
    return json.dumps(
        relevant,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


@lru_cache(maxsize=32)
def _read_checkpoint_identity(
    path_value: str,
    modified_ns: int,
    size: int,
) -> _CheckpointIdentity:
    del modified_ns, size
    checkpoint = torch.load(path_value, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint phải chứa một mapping")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint thiếu config hợp lệ")
    required = ("model_name", "split_fingerprint")
    missing = [name for name in required if not checkpoint.get(name)]
    if missing:
        raise ValueError(f"checkpoint thiếu metadata: {', '.join(missing)}")
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("checkpoint thiếu experiment_id")
    return _CheckpointIdentity(
        experiment_id=experiment_id.upper(),
        model_name=str(checkpoint["model_name"]),
        split_fingerprint=str(checkpoint["split_fingerprint"]),
        config_signature=_config_signature(config),
    )
