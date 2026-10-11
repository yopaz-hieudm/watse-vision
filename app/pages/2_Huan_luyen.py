from __future__ import annotations

from contextlib import suppress

import pandas as pd
import streamlit as st
import yaml

from app.components.charts import render_history_charts
from app.components.job_status import render_live_job
from app.components.layout import (
    SELECTED_EXPERIMENT_KEY,
    device_details,
    empty_state,
    get_registry,
    get_selected_experiment_id,
    get_workflow_coordinator,
    get_workflow_state,
    page_header,
    path_label,
    render_workflow,
    show_error,
    show_success,
    show_warning,
)
from realwaste.config import load_config
from realwaste.inference.history import PredictionHistory
from realwaste.workflow import WorkflowAction


def select_training_experiment() -> None:
    st.session_state[SELECTED_EXPERIMENT_KEY] = st.session_state["training_experiment"]


page_header(
    "Huấn luyện",
    "Chọn thí nghiệm và số vòng học tối đa, sau đó bắt đầu huấn luyện.",
)
experiment_labels = {
    "E1": "E1 · MobileNetV3 Small · Dữ liệu gốc",
    "E2": "E2 · ResNet18 · Dữ liệu gốc",
    "E3": "E3 · MobileNetV3 Small · Tăng cường dữ liệu",
    "E4": "E4 · ResNet18 · Tăng cường dữ liệu",
}
state = get_workflow_state()
coordinator = get_workflow_coordinator()
selected_id = state.selected_experiment_id
experiment_state = state.selected_experiment
experiment = get_registry().get(selected_id)
config = experiment.config

render_workflow("train", state)

with st.container(border=True, key="training_setup"):
    st.markdown('<p class="rw-panel-heading">Thiết lập phiên học</p>', unsafe_allow_html=True)
    settings_columns = st.columns([2, 1])
    st.session_state["training_experiment"] = get_selected_experiment_id()
    settings_columns[0].selectbox(
        "Thí nghiệm cần huấn luyện",
        get_registry().experiment_ids,
        format_func=lambda identifier: experiment_labels[identifier],
        key="training_experiment",
        on_change=select_training_experiment,
        help="E1–E4 là bốn thí nghiệm độc lập. Lựa chọn này cũng được dùng ở Đánh giá và Dự đoán.",
    )

    device_name, device_description = device_details()
    maximum_epochs = int(config["training"]["epochs"])
    run_config_path = experiment.output_dir / "config.yaml"
    if run_config_path.is_file():
        with suppress(OSError, ValueError, yaml.YAMLError):
            maximum_epochs = int(load_config(run_config_path)["training"]["epochs"])
    epoch_key = f"training_epochs_{selected_id}"
    if experiment_state.has_training_artifacts:
        st.session_state[epoch_key] = maximum_epochs
    maximum_epochs = int(
        settings_columns[1].number_input(
            "Số epoch tối đa",
            min_value=1,
            value=maximum_epochs,
            step=1,
            key=epoch_key,
            disabled=experiment_state.has_training_artifacts or state.active_job is not None,
            help="Một epoch là một lượt học qua toàn bộ tập Train. Mặc định là 30 cho phiên mới.",
        )
    )
    st.caption(
        "MobileNetV3 Small nhẹ hơn; ResNet18 cần nhiều tài nguyên hơn. "
        "Tăng cường dữ liệu tạo biến thể ảnh như xoay, lật, đổi độ sáng khi học."
    )
    early_stopping = config["training"].get("early_stopping", {})
    early_stopping_patience = int(early_stopping.get("patience", 5))
    st.caption(
        f"Dừng sớm nếu Validation Macro-F1 không cải thiện đủ trong "
        f"{early_stopping_patience} epoch liên tiếp. Xem số vòng đã chạy ở Tiến độ huấn luyện."
    )
    if device_name != "CPU":
        st.caption(device_description)
    if experiment_state.has_training_artifacts:
        st.info(
            "Số epoch phía trên thuộc phiên đã lưu. Để đổi số epoch và huấn luyện lại, "
            "dùng mục ‘Mở phiên huấn luyện mới’ bên dưới. "
            "Tiếp tục phiên cũ sẽ giữ số epoch của phiên đó."
        )

    start_action = WorkflowAction.START_TRAINING
    resume_action = WorkflowAction.RESUME_TRAINING

    feedback_count = PredictionHistory(coordinator.project_root).reviewed_count()
    include_feedback = False
    if feedback_count > 0:
        include_feedback = st.checkbox(
            f"Bổ sung ảnh đã xác nhận nhãn vào tập train ({feedback_count} ảnh)",
            disabled=start_action not in state.allowed_actions,
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
                start_action,
                experiment_id=selected_id,
                options={"include_feedback": include_feedback, "epochs": maximum_epochs},
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

    if experiment.output_dir.is_dir() and any(experiment.output_dir.iterdir()):
        with st.expander("Mở phiên huấn luyện mới"):
            st.write(
                "Lưu toàn bộ checkpoint, lịch sử và đánh giá của phiên hiện tại trước khi làm lại."
            )
            confirmed = st.checkbox(
                f"Lưu trữ kết quả {selected_id} và mở phiên mới",
                key=f"archive_confirm_{selected_id}",
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

progress_tab, history_tab = st.tabs(["Tiến độ", f"Biểu đồ học · {selected_id}"])
with progress_tab, st.container(border=True, key="training_progress"):
    render_live_job(show_recent=True)
with history_tab:
    history_path = experiment_state.history_path
    if history_path is not None:
        try:
            render_history_charts(pd.read_csv(history_path))
        except (OSError, pd.errors.ParserError) as error:
            show_error(f"Không đọc được history.csv: {error}")
    else:
        empty_state(
            "Biểu đồ sẽ xuất hiện khi mô hình bắt đầu học",
            f"Chạy một phiên {selected_id} để theo dõi độ chính xác và sai số qua từng epoch.",
            icon="↗",
        )
