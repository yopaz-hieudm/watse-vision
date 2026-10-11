from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import streamlit as st

from app.components.layout import (
    PROJECT_ROOT,
    format_timestamp,
    get_workflow_coordinator,
    show_error,
)

ACTIVE_STATUSES = {"queued", "running", "stopping"}


def _record_dict(record) -> dict:
    if record is None:
        return {}
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return dict(vars(record))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _tail(path: Path, lines: int = 120) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))
    except OSError:
        return ""


def _metric(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def _percentage(value) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def _elapsed_time(started_at: str | None, finished_at: str | None) -> str:
    if not started_at:
        return "—"
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = (
            datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            if finished_at
            else datetime.now(UTC)
        )
    except ValueError:
        return "—"
    total = max(0, int((finished - started).total_seconds()))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_job_details(record, *, compact: bool = False) -> None:
    data = _record_dict(record)
    if not data:
        st.caption("Không có job.")
        return
    job_id = str(data.get("job_id", ""))
    status = str(data.get("status", "unknown"))
    status_label = {
        "queued": "Đang chờ",
        "running": "Đang chạy",
        "succeeded": "Hoàn tất",
        "failed": "Có lỗi",
        "stopping": "Đang dừng",
        "stopped": "Đã dừng",
    }.get(status, status)
    kind_label = {
        "train": "Huấn luyện",
        "evaluate": "Đánh giá",
        "gradcam": "Giải thích dự đoán",
        "download": "Tải dữ liệu",
        "eda": "Kiểm tra dữ liệu",
        "split": "Chia tập dữ liệu",
        "compare": "So sánh mô hình",
    }.get(data.get("kind"), "Tác vụ")
    title = f"{kind_label} · {data['experiment_id']}" if data.get("experiment_id") else kind_label
    style = "is-error" if status == "failed" else "is-neutral" if status == "stopped" else ""
    st.markdown(
        f'<div class="rw-job-header"><span class="rw-job-title">{escape(title)}</span>'
        f'<span class="rw-badge {style}">{escape(status_label)}</span></div>',
        unsafe_allow_html=True,
    )
    if compact:
        return

    progress = _read_json(PROJECT_ROOT / "outputs/jobs" / job_id / "progress.json")
    if progress:
        epoch = int(progress.get("epoch", 0))
        total = max(int(progress.get("total_epochs", 1)), 1)
        st.progress(
            min(epoch / total, 1.0),
            text=f"Đã hoàn thành {epoch} epoch · Giới hạn {total} epoch",
        )
        if status == "succeeded":
            st.caption(f"Phiên này đã kết thúc sau {epoch} epoch; giới hạn cấu hình là {total}.")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Accuracy · Validation", _percentage(progress.get("val_accuracy")))
        metric_columns[1].metric("Macro-F1 · Validation", _percentage(progress.get("val_macro_f1")))
        metric_columns[2].metric("Macro-F1 tốt nhất", _percentage(progress.get("best_metric")))

    st.caption(
        f"Bắt đầu: {format_timestamp(data.get('started_at'))} · "
        f"Thời gian chạy: {_elapsed_time(data.get('started_at'), data.get('finished_at'))}"
    )

    job_dir = PROJECT_ROOT / "outputs/jobs" / job_id
    stdout, stderr = _tail(job_dir / "stdout.log"), _tail(job_dir / "stderr.log")
    if status == "failed" and data.get("error"):
        show_error(str(data["error"]))
    with st.expander("Chi tiết kỹ thuật và nhật ký"):
        st.caption(
            f"ID: {job_id} · PID: {data.get('pid') or '—'} · Exit code: {data.get('exit_code')}"
        )
        if progress:
            st.caption(
                f"Train loss: {_metric(progress.get('train_loss'))} · "
                f"Validation loss: {_metric(progress.get('val_loss'))} · "
                f"Learning rate: {_metric(progress.get('learning_rate'))}"
            )
        if stdout:
            st.code(stdout, language=None)
        if stderr:
            st.markdown("**stderr**")
            st.code(stderr, language=None)
        if not stdout and not stderr:
            st.caption("Job chưa ghi log.")

    if status == "stopping":
        confirm_force = st.checkbox(
            "Xác nhận buộc dừng tiến trình", key=f"force_stop_confirm_{job_id}"
        )
        if st.button(
            "Buộc dừng",
            disabled=not confirm_force,
            key=f"force_stop_{job_id}",
            type="secondary",
        ):
            try:
                get_workflow_coordinator().stop_job(job_id, force=True)
                st.rerun()
            except Exception as error:
                show_error(str(error))
    elif status in ACTIVE_STATUSES:
        confirm = st.checkbox("Xác nhận dừng job đang chạy", key=f"stop_confirm_{job_id}")
        if st.button("Dừng job", disabled=not confirm, key=f"stop_{job_id}"):
            try:
                get_workflow_coordinator().stop_job(job_id)
                st.rerun()
            except Exception as error:
                show_error(str(error))


@st.fragment(run_every=2)
def render_live_job(*, show_recent: bool = True, compact: bool = False) -> None:
    coordinator = get_workflow_coordinator()
    try:
        active = coordinator.get_state().active_job
        if active is not None:
            st.session_state["realwaste_observed_active_job"] = active.job_id
            render_job_details(active, compact=compact)
            return
        if st.session_state.pop("realwaste_observed_active_job", None) is not None:
            st.rerun(scope="app")
        if show_recent:
            recent = coordinator.get_recent_job()
            if recent is not None:
                render_job_details(recent, compact=compact)
                return
        st.caption(
            "Chưa có tác vụ đang chạy."
            if compact
            else "Chọn thí nghiệm và bắt đầu một phiên học để xem tiến độ tại đây."
        )
    except Exception as error:
        st.caption(f"Không đọc được trạng thái job: {error}")
