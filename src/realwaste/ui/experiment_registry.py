from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from realwaste.config import ConfigError, load_config
from realwaste.ui.artifact_reader import resolve_within_root

EXPERIMENTS: Mapping[str, str] = MappingProxyType(
    {
        "E1": "configs/mobilenet_v3_small_baseline.yaml",
        "E2": "configs/resnet18_baseline.yaml",
        "E3": "configs/mobilenet_v3_small_aug.yaml",
        "E4": "configs/resnet18_aug.yaml",
    }
)

VALID_EVALUATION_SPLITS = frozenset({"validation", "test"})


class ExperimentRegistryError(RuntimeError):
    """Raised when the fixed experiment registry or one of its configs is invalid."""


class UnknownExperimentError(ExperimentRegistryError, KeyError):
    """Raised when the UI requests an experiment outside the E1-E4 allowlist."""


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    config_path: Path
    config: dict[str, Any]
    output_dir: Path

    @property
    def experiment_name(self) -> str:
        return str(self.config["experiment_name"])

    @property
    def model_name(self) -> str:
        return str(self.config["model"]["name"])

    @property
    def augmentation_enabled(self) -> bool:
        return bool(self.config["data"]["augmentation"]["enabled"])

    @property
    def best_checkpoint(self) -> Path:
        return self.output_dir / "best.pt"

    @property
    def last_checkpoint(self) -> Path:
        return self.output_dir / "last.pt"

    @property
    def history_path(self) -> Path:
        return self.output_dir / "history.csv"

    @property
    def progress_path(self) -> Path:
        return self.output_dir / "progress.json"

    def evaluation_dir(self, split: str = "test") -> Path:
        normalized = split.lower()
        if normalized not in VALID_EVALUATION_SPLITS:
            raise ValueError(f"Unknown evaluation split {split!r}; expected validation or test")
        return self.output_dir / "evaluation" / normalized

    def metrics_path(self, split: str = "test") -> Path:
        return self.evaluation_dir(split) / "metrics.json"

    def classification_report_path(self, split: str = "test") -> Path:
        return self.evaluation_dir(split) / "classification_report.csv"

    def predictions_path(self, split: str = "test") -> Path:
        return self.evaluation_dir(split) / "predictions.csv"

    def figure_path(self, name: str, split: str = "test") -> Path:
        if Path(name).name != name or Path(name).suffix.lower() != ".png":
            raise ValueError("Figure name must be a plain PNG filename")
        return self.evaluation_dir(split) / "figures" / name

    def available_checkpoint(self, *, prefer_best: bool = True) -> Path | None:
        candidates = (
            (self.best_checkpoint, self.last_checkpoint)
            if prefer_best
            else (self.last_checkpoint, self.best_checkpoint)
        )
        return next((path for path in candidates if path.is_file()), None)

    def existing_metrics_path(self, split: str = "test") -> Path | None:
        path = self.metrics_path(split)
        if path.is_file():
            return path

        # Compatibility with artifacts produced before evaluation/<split>/ was added.
        legacy = self.output_dir / "metrics.json"
        if split == "test" and legacy.is_file():
            return legacy
        return None


class ExperimentRegistry:
    """Load the immutable E1-E4 experiment allowlist within one project root."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        experiments: Mapping[str, str] | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"Project root does not exist: {self.project_root}")
        mapping = EXPERIMENTS if experiments is None else experiments
        if tuple(mapping) != tuple(EXPERIMENTS):
            raise ExperimentRegistryError("Registry must contain exactly E1, E2, E3 and E4")
        self._config_paths = MappingProxyType(dict(mapping))
        self._records: dict[str, ExperimentRecord] = {}

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        return tuple(EXPERIMENTS)

    def resolve_project_path(self, path: str | Path) -> Path:
        return resolve_within_root(self.project_root, path)

    def get(self, experiment_id: str) -> ExperimentRecord:
        normalized = str(experiment_id).upper()
        if normalized not in self._config_paths:
            raise UnknownExperimentError(
                f"Thí nghiệm {experiment_id!r} không hợp lệ; chỉ cho phép E1, E2, E3 hoặc E4"
            )
        if normalized not in self._records:
            self._records[normalized] = self._load(normalized)
        return self._records[normalized]

    def list(self) -> tuple[ExperimentRecord, ...]:
        return tuple(self.get(experiment_id) for experiment_id in self.experiment_ids)

    def __iter__(self) -> Iterator[ExperimentRecord]:
        return iter(self.list())

    def __len__(self) -> int:
        return len(EXPERIMENTS)

    def refresh(self, experiment_id: str | None = None) -> None:
        if experiment_id is None:
            self._records.clear()
            return
        normalized = str(experiment_id).upper()
        if normalized not in self._config_paths:
            raise UnknownExperimentError(
                f"Thí nghiệm {experiment_id!r} không hợp lệ; chỉ cho phép E1, E2, E3 hoặc E4"
            )
        self._records.pop(normalized, None)

    def _load(self, experiment_id: str) -> ExperimentRecord:
        config_path = self.resolve_project_path(self._config_paths[experiment_id])
        try:
            config = load_config(config_path)
        except (ConfigError, FileNotFoundError, OSError) as error:
            raise ExperimentRegistryError(
                f"Không thể tải cấu hình {experiment_id} từ {config_path}: {error}"
            ) from error
        configured_id = str(config["experiment_id"]).upper()
        if configured_id != experiment_id:
            raise ExperimentRegistryError(
                f"Cấu hình {config_path} khai báo {configured_id}, phải là {experiment_id}"
            )
        output_dir = self.resolve_project_path(config["output"]["directory"])
        return ExperimentRecord(
            experiment_id=experiment_id,
            config_path=config_path,
            config=config,
            output_dir=output_dir,
        )
