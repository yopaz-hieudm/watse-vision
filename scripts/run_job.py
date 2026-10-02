from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one persisted RealWaste background job")
    parser.add_argument("--job-file", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_job(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("job metadata must be a JSON object")
    return value


def _atomic_write_job(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _wait_for_launch(job_file: Path, timeout: float = 10.0) -> dict[str, Any]:
    """Wait until JobManager persists our PID, preventing a final-state write race."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metadata = _read_job(job_file)
        if metadata.get("pid") == os.getpid() and metadata.get("status") in {
            "running",
            "stopping",
        }:
            return metadata
        time.sleep(0.01)
    raise TimeoutError("JobManager did not finish persisting runner metadata")


def run(job_file: Path, cwd: Path, command: list[str]) -> int:
    job_file = job_file.expanduser().resolve()
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise NotADirectoryError(f"Job cwd does not exist: {cwd}")

    metadata = _wait_for_launch(job_file)
    if metadata.get("command") != command:
        raise ValueError("Runner command does not match persisted job metadata")
    if Path(str(metadata.get("cwd", ""))).resolve() != cwd:
        raise ValueError("Runner cwd does not match persisted job metadata")

    child: subprocess.Popen[str] | None = None
    stop_requested = metadata.get("status") == "stopping"

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        if child is not None and child.poll() is None:
            child.terminate()

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    return_code = 1
    error: str | None = None
    try:
        if stop_requested:
            return_code = -signal.SIGTERM
        else:
            child = subprocess.Popen(command, cwd=cwd, shell=False, text=True)
            return_code = child.wait()
    except BaseException as caught:
        error = f"Runner failed: {caught}"
        return_code = 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    try:
        latest = _read_job(job_file)
        if latest.get("pid") != os.getpid():
            raise RuntimeError("Refusing to update metadata owned by another PID")
        stopped = stop_requested or latest.get("status") == "stopping"
        status = "stopped" if stopped else "succeeded" if return_code == 0 else "failed"
        latest.update(
            {
                "status": status,
                "finished_at": _utc_now(),
                "exit_code": return_code,
                "error": error,
            }
        )
        _atomic_write_job(job_file, latest)
    except BaseException as caught:
        print(f"Could not update final job metadata: {caught}", file=sys.stderr, flush=True)
        return 1 if return_code == 0 else return_code
    return return_code


def main() -> None:
    args = parse_args()
    return_code = run(args.job_file, args.cwd, list(args.command))
    if return_code < 0:
        return_code = 128 + abs(return_code)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
