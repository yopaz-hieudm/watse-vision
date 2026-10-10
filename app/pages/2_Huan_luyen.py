from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import render_history_charts
from app.components.job_status import render_live_job
from app.components.layout import (
    device_details,
    get_registry,
    get_workflow_coordinator,
    get_workflow_state,
    page_header,
    path_label,
    render_workflow,
    show_error,
    show_success,
    show_warning,
)
from realwaste.inference.history import PredictionHistory
from realwaste.workflow import WorkflowAction

state = get_workflow_state()
coordinator = get_workflow_coordinator()
selected_id = state.selected_experiment_id
experiment_state = state.selected_experiment
experiment = get_registry().get(selected_id)
config = experiment.config

page_header(
    f"Huấn luyện {selected_id}",
    "Experiment được chọn một lần ở sidebar và được giữ xuyên suốt workflow.",
)
render_workflow("train", state)

device_name, device_description = device_details()
summary_columns = st.columns(4)
summary_columns[0].metric("Model", experiment.model_name)
summary_columns[1].metric("Augmentation", "Bật" if experiment.augmentation_enabled else "Tắt")
summary_columns[2].metric("Epoch", config["training"]["epochs"])
summary_columns[3].metric("Thiết bị", device_name)
if device_name == "CPU":
    show_warning("Máy không có GPU tăng tốc; 30 epoch có thể mất nhiều thời gian.")
else:
    st.caption(device_description)

with st.expander("Cấu hình chỉ đọc"):
    st.json(
        {
            "experiment_id": selected_id,
            "config": path_label(experiment.config_path),
            "model": config["model"],
            "augmentation": config["data"]["augmentation"],
            "batch_size": config["data"]["batch_size"],
            "optimizer": config["training"]["optimizer"],
            "loss": config["training"]["loss"],
            "scheduler": config["training"]["scheduler"],
            "checkpoint": config["checkpoint"],
            "output": path_label(experiment.output_dir),
        }
    )

start_action = WorkflowAction.START_TRAINING
resume_action = WorkflowAction.RESUME_TRAINING

feedback_count = PredictionHistory(coordinator.project_root).reviewed_count()
include_feedback = st.checkbox(
    f"Bổ sung ảnh đã xác nhận nhãn vào tập train ({feedback_count} ảnh)",
    disabled=feedback_count == 0 or start_action not in state.allowed_actions,
    help=(
        "Tạo bản chụp phản hồi cho phiên mới; giữ nguyên Validation/Test. "
        "Resume dùng lại bản chụp cũ."
    ),
)
action_columns = st.columns(2)

if action_columns[0].button(
    "Bắt đầu huấn luyện mới",
    type="primary",
    disabled=start_action not in state.allowed_actions,
    width="stretch",
):
    try:
        job = coordinator.execute(
            start_action, experiment_id=selected_id, options={"include_feedback": include_feedback}
        )
        show_success(f"Đã tạo job {job.job_id}.")
        st.rerun()
    except Exception as error:
        show_error(str(error))

if action_columns[1].button(
    "Tiếp tục từ last.pt",
    disabled=resume_action not in state.allowed_actions,
    width="stretch",
):
    try:
        job = coordinator.execute(resume_action, experiment_id=selected_id)
        show_success(f"Đã tạo job resume {job.job_id}.")
        st.rerun()
    except Exception as error:
        show_error(str(error))

start_blockers = state.blockers_for(start_action)
resume_blockers = state.blockers_for(resume_action)
if start_blockers and not experiment_state.has_training_artifacts:
    show_warning("Chưa thể train: " + "; ".join(start_blockers))
if experiment_state.last_checkpoint is not None and resume_blockers:
    show_warning("Không thể resume: " + "; ".join(resume_blockers))
if experiment_state.has_training_artifacts:
    st.caption(
        "Experiment đã có artefact; train mới được khóa để tránh ghi đè. "
        "Dùng resume khi last.pt còn tương thích."
    )

if experiment.output_dir.is_dir() and any(experiment.output_dir.iterdir()):
    with st.expander("Mở phiên huấn luyện mới"):
        st.write(
            "Lưu toàn bộ checkpoint, lịch sử và đánh giá của phiên hiện tại trước khi làm lại."
        )
        confirmed = st.checkbox(
            f"Lưu trữ kết quả {selected_id} và mở phiên mới", key=f"archive_confirm_{selected_id}"
        )
        if st.button(
            "Lưu trữ và mở phiên mới",
            disabled=not confirmed or state.active_job is not None,
            key=f"archive_run_{selected_id}",
        ):
            try:
                destination = coordinator.archive_experiment(selected_id)
                st.session_state["realwaste_archive_notice"] = path_label(destination)
                st.rerun()
            except Exception as error:
                show_error(str(error))
if notice := st.session_state.pop("realwaste_archive_notice", None):
    show_success(f"Đã lưu kết quả cũ tại {notice}. Có thể bắt đầu huấn luyện mới.")

archive_root = coordinator.project_root / "outputs/archive" / selected_id
if archive_root.is_dir():
    with st.expander("Các phiên đã lưu trữ"):
        for archived in sorted(archive_root.iterdir(), reverse=True):
            st.caption(path_label(archived))

st.subheader("Job nền")
render_live_job(show_recent=True)

st.subheader("Lịch sử huấn luyện")
history_path = experiment_state.history_path
if history_path is not None:
    try:
        render_history_charts(pd.read_csv(history_path))
    except (OSError, pd.errors.ParserError) as error:
        show_error(f"Không đọc được history.csv: {error}")
else:
    st.info(f"Chưa có history.csv cho {selected_id}.")
