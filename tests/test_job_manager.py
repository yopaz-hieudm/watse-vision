from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import psutil
import pytest

import realwaste.ui.job_manager as job_manager_module
from realwaste.ui.job_manager import (
    InvalidJobArgumentsError,
    JobAlreadyRunningError,
    JobManager,
    UnknownJobKindError,
    UnsafeJobOperationError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakePopen:
    def __init__(self, command, **kwargs) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = os.getpid()
        self.terminated = False
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.terminated = True

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture
def copied_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "scripts").mkdir()
    for source in (PROJECT_ROOT / "configs").glob("*.yaml"):
        shutil.copy2(source, root / "configs" / source.name)
    return root


def _fake_launcher(monkeypatch: pytest.MonkeyPatch) -> list[_FakePopen]:
    calls: list[_FakePopen] = []

    def launch(command, **kwargs):
        process = _FakePopen(command, **kwargs)
        calls.append(process)
        return process

    monkeypatch.setattr(job_manager_module.subprocess, "Popen", launch)
    monkeypatch.setattr(JobManager, "_read_process_create_time", staticmethod(lambda _pid: None))
    return calls


def _touch_evaluation_inputs(root: Path, experiment: str = "mobilenet_v3_small_baseline") -> None:
    output = root / "outputs" / "experiments" / experiment
    (output / "evaluation" / "test").mkdir(parents=True)
    (output / "best.pt").touch()
    (output / "last.pt").touch()
    (output / "evaluation" / "test" / "predictions.csv").write_text(
        "path,correct,confidence,true_class,predicted_class\n", encoding="utf-8"
    )
    split_dir = root / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split_metadata.json").write_text(
        json.dumps({"fingerprint": "fixture-split"}), encoding="utf-8"
    )


def test_start_job_uses_allowlisted_command_and_shell_false(
    copied_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)

    record = manager.start_job("download", args={"verify_only": True})

    assert record.status == "running"
    assert record.command == (
        sys.executable,
        "scripts/download_data.py",
        "--verify-only",
    )
    assert calls[0].command[:2] == [sys.executable, "scripts/run_job.py"]
    assert calls[0].kwargs["shell"] is False
    assert calls[0].kwargs["cwd"] == copied_project
    persisted = json.loads(
        (manager.jobs_dir / record.job_id / "job.json").read_text(encoding="utf-8")
    )
    assert persisted["command"] == list(record.command)
    assert persisted["pid"] == os.getpid()


def test_alias_and_untrusted_arguments_are_rejected(
    copied_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)

    record = manager.start_job("inspect")
    assert record.kind == "eda"
    assert record.command[1] == "scripts/inspect_data.py"

    with pytest.raises(UnknownJobKindError):
        JobManager(copied_project, jobs_dir=copied_project / "other-jobs").start_job("shell")
    with pytest.raises(InvalidJobArgumentsError, match="Tham số không được phép"):
        JobManager(copied_project, jobs_dir=copied_project / "more-jobs").start_job(
            "download", args={"url": "https://example.invalid/payload"}
        )


def test_second_active_job_is_blocked(
    copied_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)
    manager.start_job("eda")
    monkeypatch.setattr(manager, "_owned_process", lambda _record: object())

    with pytest.raises(JobAlreadyRunningError):
        manager.start_job("split")
    assert len(calls) == 1


@pytest.mark.parametrize("epochs", [1, 12, 60])
def test_training_job_passes_selected_epochs(copied_project, monkeypatch, epochs):
    _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)
    record = manager.start_job("train", "E4", {"epochs": epochs})
    assert record.command[record.command.index("--epochs") + 1] == str(epochs)
    assert record.command[record.command.index("--config") + 1] == "configs/resnet18_aug.yaml"


@pytest.mark.parametrize("epochs", [0, -1, True, 2.5, "30"])
def test_training_job_rejects_invalid_epochs(copied_project, monkeypatch, epochs):
    calls = _fake_launcher(monkeypatch)
    with pytest.raises(InvalidJobArgumentsError):
        JobManager(copied_project).start_job("train", "E4", {"epochs": epochs})
    assert calls == []


def test_resume_keeps_saved_epoch_budget(copied_project, monkeypatch):
    _touch_evaluation_inputs(copied_project)
    _fake_launcher(monkeypatch)
    monkeypatch.setattr(
        job_manager_module,
        "validate_resume_checkpoint",
        lambda *_args, **_kwargs: {"config": {"training": {"epochs": 60}}},
    )
    record = JobManager(copied_project).start_job("train", "E1", {"resume": True})
    assert record.command[record.command.index("--epochs") + 1] == "60"


def test_train_evaluate_and_gradcam_paths_are_derived_from_registry(
    copied_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _touch_evaluation_inputs(copied_project)
    monkeypatch.setattr(
        job_manager_module, "validate_resume_checkpoint", lambda *_args, **_kwargs: {}
    )
    _fake_launcher(monkeypatch)

    train_manager = JobManager(copied_project, jobs_dir=copied_project / "jobs-train")
    train = train_manager.start_job("train", "E1", {"resume": True})
    assert train.command[:4] == (
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/mobilenet_v3_small_baseline.yaml",
    )
    assert train.command[-2:] == (
        "--resume",
        "outputs/experiments/mobilenet_v3_small_baseline/last.pt",
    )
    progress_index = train.command.index("--progress-path") + 1
    assert train.command[progress_index].endswith(f"{train.job_id}/progress.json")

    evaluate_manager = JobManager(copied_project, jobs_dir=copied_project / "jobs-evaluate")
    evaluate = evaluate_manager.start_job(
        "evaluate", "E1", {"checkpoint": "last", "split": "validation"}
    )
    assert evaluate.command[evaluate.command.index("--checkpoint") + 1].endswith("last.pt")
    assert evaluate.command[evaluate.command.index("--output-dir") + 1].endswith(
        "evaluation/validation"
    )

    gradcam_manager = JobManager(copied_project, jobs_dir=copied_project / "jobs-gradcam")
    gradcam = gradcam_manager.start_job("gradcam", "E1", {"checkpoint": "best", "samples": 12})
    predictions = gradcam.command[gradcam.command.index("--predictions") + 1]
    assert predictions.endswith("evaluation/test/predictions.csv")
    assert gradcam.command[gradcam.command.index("--samples") + 1] == "12"


def test_refresh_marks_disappeared_process_failed(
    copied_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)
    record = manager.start_job("split")
    manager._local_processes.clear()

    def missing(_record):
        raise psutil.NoSuchProcess(record.pid)

    monkeypatch.setattr(manager, "_owned_process", missing)
    refreshed = manager.refresh_job(record.job_id)

    assert refreshed.status == "failed"
    assert refreshed.finished_at is not None
    assert "không cập nhật" in (refreshed.error or "")
    assert manager.get_active_job() is None


def test_stop_validates_ownership_before_terminating(
    copied_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)
    record = manager.start_job("eda")

    def mismatch(_record):
        raise UnsafeJobOperationError("PID mismatch")

    monkeypatch.setattr(manager, "_owned_process", mismatch)
    with pytest.raises(UnsafeJobOperationError, match="PID mismatch"):
        manager.stop_job(record.job_id)
    assert not calls[0].terminated
    assert manager.get_job(record.job_id).status == "running"


def test_stop_terminates_process_tree_and_tail_log(
    copied_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)
    record = manager.start_job("eda")

    class Process:
        def __init__(self) -> None:
            self.terminated = False

        def children(self, *, recursive: bool):
            assert recursive is True
            return [child]

        def terminate(self) -> None:
            self.terminated = True

    process, child = Process(), Process()
    monkeypatch.setattr(manager, "_owned_process", lambda _record: process)
    monkeypatch.setattr(
        job_manager_module.psutil,
        "wait_procs",
        lambda processes, timeout: (processes, []),
    )
    stopped = manager.stop_job(record.job_id)

    assert stopped.status == "stopped"
    assert process.terminated and child.terminated
    log_path = manager.jobs_dir / record.job_id / "stdout.log"
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert manager.tail_log(record.job_id, lines=2) == "two\nthree\n"


def test_runner_persists_success_and_failure(tmp_path: Path) -> None:
    root = tmp_path / "integration-project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "run_job.py", scripts / "run_job.py")
    target = scripts / "download_data.py"
    target.write_text(
        "import sys\n"
        "print('runner-ok')\n"
        "raise SystemExit(0 if '--verify-only' in sys.argv else 3)\n",
        encoding="utf-8",
    )
    manager = JobManager(root, stop_timeout=1)

    succeeded = manager.start_job("download", args={"verify_only": True})
    succeeded = _wait_for_final(manager, succeeded.job_id)
    assert succeeded.status == "succeeded"
    assert succeeded.exit_code == 0
    assert "runner-ok" in manager.tail_log(succeeded.job_id)

    failed = manager.start_job("download")
    failed = _wait_for_final(manager, failed.job_id)
    assert failed.status == "failed"
    assert failed.exit_code == 3


def _wait_for_final(manager: JobManager, job_id: str, timeout: float = 5) -> object:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get_job(job_id)
        if record.status not in {"queued", "running", "stopping"}:
            return record
        time.sleep(0.02)
    stdout = manager.tail_log(job_id)
    stderr = manager.tail_log(job_id, "stderr")
    pytest.fail(f"Job did not finish in time. stdout={stdout!r}, stderr={stderr!r}")


def test_archive_preserves_broken_run_and_unblocks_fresh_training(copied_project):
    from realwaste.workflow import WorkflowAction, WorkflowCoordinator

    manager = JobManager(copied_project)
    experiment = manager._registry.get("E1")
    experiment.output_dir.mkdir(parents=True)
    experiment.best_checkpoint.write_bytes(b"broken-model")
    experiment.history_path.write_text("epoch\n1\n", encoding="utf-8")
    destination = manager.archive_experiment("E1")
    assert (destination / "best.pt").read_bytes() == b"broken-model"
    assert (destination / "history.csv").is_file()
    assert not experiment.output_dir.exists()
    state = WorkflowCoordinator(copied_project, job_manager=manager).get_state("E1")
    assert not state.selected_experiment.has_training_artifacts
    assert "artefact" not in ";".join(state.blockers_for(WorkflowAction.START_TRAINING))


def test_archive_is_blocked_while_any_job_is_active(copied_project, monkeypatch):
    _fake_launcher(monkeypatch)
    manager = JobManager(copied_project)
    manager.start_job("download")
    with pytest.raises(JobAlreadyRunningError):
        manager.archive_experiment("E1")
