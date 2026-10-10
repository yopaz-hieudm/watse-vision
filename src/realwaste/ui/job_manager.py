from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

import psutil

from realwaste.training.checkpoint import validate_resume_checkpoint
from realwaste.training.progress import atomic_write_json
from realwaste.ui.artifact_reader import resolve_within_root
from realwaste.ui.experiment_registry import ExperimentRegistry

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses msvcrt below.
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]


JOB_KINDS = frozenset({"download", "eda", "split", "train", "evaluate", "compare", "gradcam"})
JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "stopping", "stopped"})
ACTIVE_JOB_STATUSES = frozenset({"queued", "running", "stopping"})
_KIND_ALIASES = {"inspect": "eda"}
_EXPERIMENT_JOB_KINDS = frozenset({"train", "evaluate", "gradcam"})

_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class JobManagerError(RuntimeError):
    """Base error raised by the local background-job manager."""


class UnknownJobKindError(JobManagerError, ValueError):
    """Raised when a caller requests a command outside the fixed allowlist."""


class InvalidJobArgumentsError(JobManagerError, ValueError):
    """Raised when allowlisted job arguments have an invalid shape or value."""


class JobAlreadyRunningError(JobManagerError):
    """Raised when a second heavyweight job is requested."""


class JobNotFoundError(JobManagerError, FileNotFoundError):
    """Raised when persisted job metadata cannot be found."""


class JobMetadataError(JobManagerError):
    """Raised when a persisted job record is malformed."""


class UnsafeJobOperationError(JobManagerError):
    """Raised when a PID no longer belongs to the recorded runner process."""


class JobStopTimeoutError(JobManagerError, TimeoutError):
    """Raised when graceful termination exceeds the configured timeout."""


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Serializable state for one background pipeline command."""

    job_id: str
    kind: str
    experiment_id: str | None
    command: tuple[str, ...]
    pid: int | None
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    cwd: str
    runner_command: tuple[str, ...] = ()
    process_create_time: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "experiment_id": self.experiment_id,
            "command": list(self.command),
            "pid": self.pid,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "cwd": self.cwd,
            "runner_command": list(self.runner_command),
            "process_create_time": self.process_create_time,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobRecord:
        try:
            job_id = value["job_id"]
            kind = value["kind"]
            command = value["command"]
            status = value["status"]
            created_at = value["created_at"]
        except KeyError as error:
            raise JobMetadataError(f"Thiếu trường job bắt buộc: {error.args[0]}") from error

        if not isinstance(job_id, str) or not _is_safe_job_id(job_id):
            raise JobMetadataError("job_id trong metadata không hợp lệ")
        if not isinstance(kind, str) or kind not in JOB_KINDS:
            raise JobMetadataError(f"Loại job trong metadata không hợp lệ: {kind!r}")
        if not isinstance(command, list | tuple) or not all(
            isinstance(item, str) for item in command
        ):
            raise JobMetadataError("command trong metadata phải là một danh sách chuỗi")
        if not isinstance(status, str) or status not in JOB_STATUSES:
            raise JobMetadataError(f"Trạng thái job không hợp lệ: {status!r}")
        if not isinstance(created_at, str):
            raise JobMetadataError("created_at trong metadata phải là chuỗi")

        runner_command = value.get("runner_command", ())
        if not isinstance(runner_command, list | tuple) or not all(
            isinstance(item, str) for item in runner_command
        ):
            raise JobMetadataError("runner_command trong metadata phải là một danh sách chuỗi")

        pid = value.get("pid")
        if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0):
            raise JobMetadataError("pid trong metadata phải là số nguyên dương hoặc null")
        exit_code = value.get("exit_code")
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            raise JobMetadataError("exit_code trong metadata phải là số nguyên hoặc null")
        process_create_time = value.get("process_create_time")
        if process_create_time is not None and (
            isinstance(process_create_time, bool)
            or not isinstance(process_create_time, int | float)
        ):
            raise JobMetadataError("process_create_time trong metadata phải là số hoặc null")

        experiment_id = value.get("experiment_id")
        if experiment_id is not None and not isinstance(experiment_id, str):
            raise JobMetadataError("experiment_id trong metadata phải là chuỗi hoặc null")
        cwd = value.get("cwd", "")
        if not isinstance(cwd, str):
            raise JobMetadataError("cwd trong metadata phải là chuỗi")
        for field in ("started_at", "finished_at", "error"):
            field_value = value.get(field)
            if field_value is not None and not isinstance(field_value, str):
                raise JobMetadataError(f"{field} trong metadata phải là chuỗi hoặc null")

        return cls(
            job_id=job_id,
            kind=kind,
            experiment_id=experiment_id,
            command=tuple(command),
            pid=pid,
            status=status,
            created_at=created_at,
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
            exit_code=exit_code,
            cwd=cwd,
            runner_command=tuple(runner_command),
            process_create_time=(
                float(process_create_time) if process_create_time is not None else None
            ),
            error=value.get("error"),
        )


class JobManager:
    """Launch and recover one allowlisted heavyweight job at a time."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        jobs_dir: str | Path | None = None,
        stop_timeout: float = 10,
    ) -> None:
        default_root = Path(__file__).resolve().parents[3]
        self.project_root = Path(project_root or default_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"Project root does not exist: {self.project_root}")

        if jobs_dir is None:
            self.jobs_dir = self.project_root / "outputs" / "jobs"
        else:
            candidate = Path(jobs_dir).expanduser()
            self.jobs_dir = (
                candidate if candidate.is_absolute() else self.project_root / candidate
            ).resolve()
        if isinstance(stop_timeout, bool) or stop_timeout <= 0:
            raise ValueError("stop_timeout must be a positive number")
        self.stop_timeout = float(stop_timeout)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.jobs_dir / ".manager.lock"
        self._registry = ExperimentRegistry(self.project_root)
        self._local_processes: dict[str, subprocess.Popen[str]] = {}

    def start_job(
        self,
        kind: str,
        experiment_id: str | None = None,
        args: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        canonical_kind = self._normalize_kind(kind)
        normalized_args = self._normalize_args(args)
        normalized_experiment = self._normalize_experiment(canonical_kind, experiment_id)

        with self._manager_lock():
            active = self._get_active_job_unlocked()
            if active is not None:
                raise JobAlreadyRunningError(
                    f"Job {active.job_id} ({active.kind}) đang ở trạng thái {active.status}"
                )

            job_id = self._new_job_id(canonical_kind, normalized_experiment)
            job_dir = self.jobs_dir / job_id
            command = self._build_command(
                canonical_kind,
                normalized_experiment,
                normalized_args,
                job_dir,
            )
            job_file = job_dir / "job.json"
            runner_command = (
                sys.executable,
                "scripts/run_job.py",
                "--job-file",
                self._command_path(job_file),
                "--cwd",
                str(self.project_root),
                "--",
                *command,
            )
            record = JobRecord(
                job_id=job_id,
                kind=canonical_kind,
                experiment_id=normalized_experiment,
                command=command,
                pid=None,
                status="queued",
                created_at=_utc_now(),
                started_at=None,
                finished_at=None,
                exit_code=None,
                cwd=str(self.project_root),
                runner_command=runner_command,
            )

            job_dir.mkdir(parents=True, exist_ok=False)
            self._write_record(record)
            stdout_handle: TextIO | None = None
            stderr_handle: TextIO | None = None
            process: subprocess.Popen[str] | None = None
            try:
                stdout_handle = (job_dir / "stdout.log").open("a", encoding="utf-8", buffering=1)
                stderr_handle = (job_dir / "stderr.log").open("a", encoding="utf-8", buffering=1)
                process = subprocess.Popen(
                    list(runner_command),
                    cwd=self.project_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    start_new_session=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                process_create_time = self._read_process_create_time(process.pid)
                record = replace(
                    record,
                    pid=process.pid,
                    status="running",
                    started_at=_utc_now(),
                    process_create_time=process_create_time,
                )
                self._write_record(record)
                self._local_processes[job_id] = process
            except BaseException as error:
                if process is not None:
                    with suppress(OSError):
                        process.terminate()
                failed = replace(
                    record,
                    pid=process.pid if process is not None else None,
                    status="failed",
                    finished_at=_utc_now(),
                    error=f"Không thể khởi động job: {error}",
                )
                self._write_record(failed)
                raise JobManagerError(failed.error) from error
            finally:
                if stdout_handle is not None:
                    stdout_handle.close()
                if stderr_handle is not None:
                    stderr_handle.close()
            return record

    def get_job(self, job_id: str) -> JobRecord:
        return self._read_record(self._job_file(job_id))

    def archive_experiment(self, experiment_id: str) -> Path:
        """Move a whole run into a unique archive under the same job lock."""
        with self._manager_lock():
            active = self._get_active_job_unlocked()
            if active is not None:
                raise JobAlreadyRunningError(f"Hãy đợi hoặc dừng job {active.job_id} trước")
            self._registry.refresh(experiment_id)
            experiment = self._registry.get(experiment_id)
            source = resolve_within_root(self.project_root, experiment.output_dir)
            experiments_root = self.project_root / "outputs/experiments"
            if source.parent != experiments_root.resolve() or source == experiments_root:
                raise UnsafeJobOperationError(
                    "Chỉ lưu trữ thư mục experiment trong outputs/experiments"
                )
            if not source.is_dir() or not any(source.iterdir()):
                raise FileNotFoundError("Experiment chưa có kết quả để lưu trữ")
            archive_root = resolve_within_root(
                self.project_root, f"outputs/archive/{experiment.experiment_id}"
            )
            destination = archive_root / f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:12]}"
            resolve_within_root(self.project_root, destination)
            archive_root.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            return destination

    def refresh_job(self, job_id: str) -> JobRecord:
        with self._manager_lock():
            return self._refresh_record(self.get_job(job_id))

    def get_active_job(self) -> JobRecord | None:
        with self._manager_lock():
            return self._get_active_job_unlocked()

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        records = self._list_records_unlocked()
        records.sort(key=lambda record: (record.created_at, record.job_id), reverse=True)
        return records[:limit]

    def stop_job(self, job_id: str, *, force: bool = False) -> JobRecord:
        with self._manager_lock():
            record = self.get_job(job_id)
            if record.status in {"succeeded", "failed", "stopped"}:
                return record
            if record.pid is None:
                stopped = replace(
                    record,
                    status="stopped",
                    finished_at=_utc_now(),
                    error="Job đã bị dừng trước khi tiến trình được khởi tạo",
                )
                self._write_record(stopped)
                return stopped

            try:
                process = self._owned_process(record)
            except psutil.NoSuchProcess:
                stopped = replace(
                    record,
                    status="stopped",
                    finished_at=_utc_now(),
                    error=record.error or "Tiến trình đã kết thúc trước yêu cầu dừng",
                )
                self._local_processes.pop(job_id, None)
                self._write_record(stopped)
                return stopped
            if force and record.status != "stopping":
                raise InvalidJobArgumentsError(
                    "Chỉ được force kill sau khi job đã ở trạng thái stopping"
                )
            stopping = replace(record, status="stopping")
            self._write_record(stopping)

        processes = [*process.children(recursive=True), process]
        if force:
            for item in processes:
                with suppress(psutil.NoSuchProcess):
                    item.kill()
        else:
            for item in processes:
                with suppress(psutil.NoSuchProcess):
                    item.terminate()

        _gone, alive = psutil.wait_procs(processes, timeout=self.stop_timeout)
        if alive and not force:
            raise JobStopTimeoutError(
                f"Job {job_id} chưa dừng sau {self.stop_timeout:g} giây; "
                "gọi lại stop_job(..., force=True) để buộc dừng"
            )
        if alive:
            raise JobStopTimeoutError(f"Không thể buộc dừng job {job_id}")

        with self._manager_lock():
            latest = self.get_job(job_id)
            self._local_processes.pop(job_id, None)
            if latest.status in {"succeeded", "failed", "stopped"}:
                return latest
            stopped = replace(latest, status="stopped", finished_at=_utc_now())
            self._write_record(stopped)
            return stopped

    def tail_log(self, job_id: str, stream: str = "stdout", lines: int = 200) -> str:
        self.get_job(job_id)  # Validate both the identifier and the persisted job.
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        if isinstance(lines, bool) or not isinstance(lines, int) or lines <= 0:
            raise ValueError("lines must be a positive integer")
        path = self.jobs_dir / job_id / f"{stream}.log"
        if not path.is_file():
            return ""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return "".join(deque(handle, maxlen=lines))
        except OSError as error:
            raise JobManagerError(f"Không thể đọc log {path}: {error}") from error

    def _normalize_kind(self, kind: str) -> str:
        if not isinstance(kind, str):
            raise UnknownJobKindError("Loại job phải là chuỗi")
        normalized = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
        if normalized not in JOB_KINDS:
            allowed = ", ".join(sorted(JOB_KINDS | set(_KIND_ALIASES)))
            raise UnknownJobKindError(f"Loại job {kind!r} không hợp lệ; cho phép: {allowed}")
        return normalized

    @staticmethod
    def _normalize_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
        if args is None:
            return {}
        if not isinstance(args, Mapping):
            raise InvalidJobArgumentsError("args phải là mapping hoặc null")
        if not all(isinstance(key, str) for key in args):
            raise InvalidJobArgumentsError("Tên tham số job phải là chuỗi")
        return dict(args)

    def _normalize_experiment(self, kind: str, experiment_id: str | None) -> str | None:
        if kind in _EXPERIMENT_JOB_KINDS:
            if not experiment_id:
                raise InvalidJobArgumentsError(f"Job {kind} yêu cầu experiment_id")
            self._registry.refresh(str(experiment_id))
            return self._registry.get(str(experiment_id)).experiment_id
        if experiment_id is not None:
            raise InvalidJobArgumentsError(f"Job {kind} không nhận experiment_id")
        return None

    def _build_command(
        self,
        kind: str,
        experiment_id: str | None,
        args: dict[str, Any],
        job_dir: Path,
    ) -> tuple[str, ...]:
        if kind == "download":
            self._require_keys(args, {"verify_only"})
            command = [sys.executable, "scripts/download_data.py"]
            if self._bool_arg(args, "verify_only", default=False):
                command.append("--verify-only")
            return tuple(command)

        if kind == "eda":
            self._require_keys(args, {"seed", "strict"})
            seed = self._int_arg(args, "seed", default=42)
            command = [
                sys.executable,
                "scripts/inspect_data.py",
                "--data-dir",
                "data/raw/RealWaste",
                "--output-dir",
                "outputs/eda",
                "--seed",
                str(seed),
            ]
            if not self._bool_arg(args, "strict", default=True):
                command.append("--no-strict")
            return tuple(command)

        if kind == "split":
            keys = {"train_ratio", "val_ratio", "test_ratio", "seed"}
            self._require_keys(args, keys)
            ratios = [
                self._ratio_arg(args, "train_ratio", 0.70),
                self._ratio_arg(args, "val_ratio", 0.15),
                self._ratio_arg(args, "test_ratio", 0.15),
            ]
            if abs(sum(ratios) - 1.0) > 1e-9:
                raise InvalidJobArgumentsError("Tổng train/validation/test ratio phải bằng 1")
            seed = self._int_arg(args, "seed", default=42)
            return (
                sys.executable,
                "scripts/create_splits.py",
                "--data-dir",
                "data/raw/RealWaste",
                "--output-dir",
                "data/splits",
                "--train-ratio",
                str(ratios[0]),
                "--val-ratio",
                str(ratios[1]),
                "--test-ratio",
                str(ratios[2]),
                "--seed",
                str(seed),
            )

        if kind == "compare":
            self._require_keys(args, set())
            return (
                sys.executable,
                "scripts/compare_experiments.py",
                "--input-dir",
                "outputs/experiments",
                "--output-dir",
                "outputs/comparison",
            )

        if experiment_id is None:  # Defensive: normalization already enforces this.
            raise InvalidJobArgumentsError(f"Job {kind} yêu cầu experiment_id")
        experiment = self._registry.get(experiment_id)

        if kind == "train":
            self._require_keys(args, {"resume", "include_feedback"})
            resume = self._bool_arg(args, "resume", default=False)
            include_feedback = self._bool_arg(args, "include_feedback", default=False)
            if resume and include_feedback:
                raise InvalidJobArgumentsError("Phản hồi mới chỉ được thêm vào phiên train mới")
            command = [
                sys.executable,
                "scripts/train.py",
                "--config",
                self._command_path(experiment.config_path),
                "--progress-path",
                self._command_path(job_dir / "progress.json"),
            ]
            if resume:
                if not experiment.last_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Không thể resume {experiment_id}: thiếu {experiment.last_checkpoint}"
                    )
                split_metadata_path = self.project_root / "data/splits/split_metadata.json"
                if not split_metadata_path.is_file():
                    raise FileNotFoundError(
                        f"Không thể resume {experiment_id}: thiếu {split_metadata_path}"
                    )
                try:
                    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
                    split_fingerprint = str(split_metadata["fingerprint"])
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                    raise InvalidJobArgumentsError(
                        f"Split metadata không hợp lệ: {split_metadata_path}"
                    ) from error
                try:
                    validate_resume_checkpoint(
                        experiment.last_checkpoint,
                        expected_split_fingerprint=split_fingerprint,
                        expected_experiment_id=experiment.experiment_id,
                        expected_model_name=experiment.model_name,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    raise InvalidJobArgumentsError(
                        f"Checkpoint resume không hợp lệ: {error}"
                    ) from error
                command.extend(["--resume", self._command_path(experiment.last_checkpoint)])
            if include_feedback:
                from realwaste.inference.history import PredictionHistory

                manifest = PredictionHistory(self.project_root).create_training_snapshot()
                command.extend(["--feedback-manifest", self._command_path(manifest)])
            return tuple(command)

        if kind == "evaluate":
            self._require_keys(args, {"checkpoint", "split"})
            checkpoint_kind = self._choice_arg(args, "checkpoint", {"best", "last"}, default="best")
            split = self._choice_arg(args, "split", {"validation", "test"}, default="test")
            checkpoint = (
                experiment.best_checkpoint
                if checkpoint_kind == "best"
                else experiment.last_checkpoint
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")
            return (
                sys.executable,
                "scripts/evaluate.py",
                "--checkpoint",
                self._command_path(checkpoint),
                "--split",
                split,
                "--device",
                "auto",
                "--output-dir",
                self._command_path(experiment.evaluation_dir(split)),
            )

        if kind == "gradcam":
            self._require_keys(args, {"checkpoint", "samples"})
            checkpoint_kind = self._choice_arg(args, "checkpoint", {"best", "last"}, default="best")
            samples = self._int_arg(args, "samples", default=20)
            if not 2 <= samples <= 500:
                raise InvalidJobArgumentsError("samples phải nằm trong khoảng 2..500")
            checkpoint = (
                experiment.best_checkpoint
                if checkpoint_kind == "best"
                else experiment.last_checkpoint
            )
            predictions = experiment.predictions_path("test")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")
            if not predictions.is_file():
                raise FileNotFoundError(f"Grad-CAM cần kết quả dự đoán Test trước: {predictions}")
            return (
                sys.executable,
                "scripts/generate_gradcam.py",
                "--checkpoint",
                self._command_path(checkpoint),
                "--predictions",
                self._command_path(predictions),
                "--samples",
                str(samples),
                "--device",
                "auto",
            )

        raise UnknownJobKindError(f"Không có command builder cho {kind!r}")

    @staticmethod
    def _require_keys(args: Mapping[str, Any], allowed: set[str]) -> None:
        unknown = sorted(set(args) - allowed)
        if unknown:
            raise InvalidJobArgumentsError(f"Tham số không được phép: {', '.join(unknown)}")

    @staticmethod
    def _bool_arg(args: Mapping[str, Any], name: str, *, default: bool) -> bool:
        value = args.get(name, default)
        if not isinstance(value, bool):
            raise InvalidJobArgumentsError(f"{name} phải là boolean")
        return value

    @staticmethod
    def _int_arg(args: Mapping[str, Any], name: str, *, default: int) -> int:
        value = args.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidJobArgumentsError(f"{name} phải là số nguyên")
        return value

    @staticmethod
    def _ratio_arg(args: Mapping[str, Any], name: str, default: float) -> float:
        value = args.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvalidJobArgumentsError(f"{name} phải là số")
        result = float(value)
        if not 0 < result < 1:
            raise InvalidJobArgumentsError(f"{name} phải nằm trong khoảng (0, 1)")
        return result

    @staticmethod
    def _choice_arg(
        args: Mapping[str, Any],
        name: str,
        choices: set[str],
        *,
        default: str,
    ) -> str:
        value = args.get(name, default)
        if not isinstance(value, str) or value not in choices:
            expected = ", ".join(sorted(choices))
            raise InvalidJobArgumentsError(f"{name} phải là một trong: {expected}")
        return value

    def _command_path(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        try:
            return resolved.relative_to(self.project_root).as_posix()
        except ValueError:
            return str(resolved)

    def _new_job_id(self, kind: str, experiment_id: str | None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [timestamp]
        if experiment_id:
            parts.append(experiment_id)
        parts.append(kind)
        base = "_".join(parts)
        candidate = base
        suffix = 2
        while (self.jobs_dir / candidate).exists():
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _job_file(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not _is_safe_job_id(job_id):
            raise JobNotFoundError(f"Job id không hợp lệ: {job_id!r}")
        path = self.jobs_dir / job_id / "job.json"
        if not path.is_file():
            raise JobNotFoundError(f"Không tìm thấy job: {job_id}")
        return path

    def _read_record(self, path: Path) -> JobRecord:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise JobNotFoundError(f"Không tìm thấy metadata job: {path}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise JobMetadataError(f"Không thể đọc metadata job {path}: {error}") from error
        if not isinstance(value, dict):
            raise JobMetadataError(f"Metadata job phải là JSON object: {path}")
        record = JobRecord.from_dict(value)
        if path.parent.name != record.job_id:
            raise JobMetadataError(
                f"job_id {record.job_id!r} không khớp thư mục {path.parent.name!r}"
            )
        return record

    def _write_record(self, record: JobRecord) -> None:
        atomic_write_json(self.jobs_dir / record.job_id / "job.json", record.to_dict())

    def _list_records_unlocked(self) -> list[JobRecord]:
        records = []
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                records.append(self._read_record(path))
            except JobManagerError:
                continue
        return records

    def _get_active_job_unlocked(self) -> JobRecord | None:
        active = [
            record
            for record in self._list_records_unlocked()
            if record.status in ACTIVE_JOB_STATUSES
        ]
        active.sort(key=lambda record: (record.created_at, record.job_id), reverse=True)
        for record in active:
            refreshed = self._refresh_record(record)
            if refreshed.status in ACTIVE_JOB_STATUSES:
                return refreshed
        return None

    def _refresh_record(self, record: JobRecord) -> JobRecord:
        if record.status not in ACTIVE_JOB_STATUSES:
            self._local_processes.pop(record.job_id, None)
            return record
        if record.pid is None:
            return record

        local_process = self._local_processes.get(record.job_id)
        if local_process is not None:
            poll = getattr(local_process, "poll", None)
            if not callable(poll):
                return record
            return_code = poll()
            if return_code is None:
                return record
            self._local_processes.pop(record.job_id, None)
            latest = self.get_job(record.job_id)
            if latest.status not in ACTIVE_JOB_STATUSES:
                return latest
            status = (
                "stopped"
                if latest.status == "stopping"
                else ("succeeded" if return_code == 0 else "failed")
            )
            updated = replace(
                latest,
                status=status,
                finished_at=_utc_now(),
                exit_code=return_code,
                error=(
                    latest.error
                    if status != "failed"
                    else latest.error or f"Runner kết thúc với exit code {return_code}"
                ),
            )
            self._write_record(updated)
            return updated
        try:
            self._owned_process(record)
        except psutil.NoSuchProcess:
            final_status = "stopped" if record.status == "stopping" else "failed"
            error = record.error
            if final_status == "failed" and not error:
                error = "Tiến trình kết thúc mà không cập nhật trạng thái cuối"
            updated = replace(
                record,
                status=final_status,
                finished_at=record.finished_at or _utc_now(),
                error=error,
            )
            self._write_record(updated)
            return updated
        except UnsafeJobOperationError as error:
            updated = replace(
                record,
                status="failed",
                finished_at=_utc_now(),
                error=str(error),
            )
            self._write_record(updated)
            return updated
        return record

    def _owned_process(self, record: JobRecord) -> psutil.Process:
        if record.pid is None:
            raise psutil.NoSuchProcess(pid=-1)
        try:
            process = psutil.Process(record.pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                raise psutil.NoSuchProcess(record.pid)
            actual_cwd = Path(process.cwd()).resolve()
            actual_command = tuple(process.cmdline())
            actual_create_time = process.create_time()
        except psutil.NoSuchProcess:
            raise
        except (psutil.AccessDenied, OSError) as error:
            raise UnsafeJobOperationError(
                f"Không thể xác minh tiến trình PID {record.pid}: {error}"
            ) from error

        expected_cwd = Path(record.cwd).resolve()
        if actual_cwd != expected_cwd:
            raise UnsafeJobOperationError(
                f"Từ chối thao tác PID {record.pid}: cwd không khớp metadata"
            )
        if not record.runner_command or actual_command != record.runner_command:
            raise UnsafeJobOperationError(
                f"Từ chối thao tác PID {record.pid}: command line không khớp metadata"
            )
        if (
            record.process_create_time is not None
            and abs(actual_create_time - record.process_create_time) > 0.01
        ):
            raise UnsafeJobOperationError(
                f"Từ chối thao tác PID {record.pid}: thời điểm tạo process không khớp"
            )
        return process

    @staticmethod
    def _read_process_create_time(pid: int) -> float | None:
        try:
            return float(psutil.Process(pid).create_time())
        except (psutil.Error, OSError):
            return None

    @contextmanager
    def _manager_lock(self):
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(self._lock_path, threading.RLock())
        with thread_lock, self._lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:
                if handle.tell() == 0:
                    handle.write("0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _is_safe_job_id(value: str) -> bool:
    return bool(value) and Path(value).name == value and value not in {".", ".."}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ACTIVE_JOB_STATUSES",
    "JOB_KINDS",
    "JOB_STATUSES",
    "InvalidJobArgumentsError",
    "JobAlreadyRunningError",
    "JobManager",
    "JobManagerError",
    "JobMetadataError",
    "JobNotFoundError",
    "JobRecord",
    "JobStopTimeoutError",
    "UnknownJobKindError",
    "UnsafeJobOperationError",
]
