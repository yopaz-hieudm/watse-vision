from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.layout import (
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
from realwaste.workflow import WorkflowAction

state = get_workflow_state()
coordinator = get_workflow_coordinator()
selected_id = state.selected_experiment_id
experiment_state = state.selected_experiment
experiment = get_registry().get(selected_id)

page_header(
    f"Đánh giá {selected_id}",
    "Checkpoint, split và artefact đánh giá được điều phối theo experiment đang chọn.",
)
render_workflow("evaluate", state)

checkpoint_options = []
if experiment_state.best_checkpoint_compatible:
    checkpoint_options.append("best")
if experiment_state.last_checkpoint_compatible:
    checkpoint_options.append("last")

if not checkpoint_options:
    st.info(
        f"{selected_id} chưa có checkpoint tương thích. "
        "Hãy hoàn thành hoặc kiểm tra lại bước Huấn luyện."
    )
    for blocker in experiment_state.blockers:
        show_warning(blocker)
    st.stop()

checkpoint_kind = st.radio("Checkpoint", checkpoint_options, horizontal=True)
split = st.radio("Tập dữ liệu", ["test", "validation"], horizontal=True)
checkpoint_path = (
    experiment_state.best_checkpoint
    if checkpoint_kind == "best"
    else experiment_state.last_checkpoint
)
st.caption(f"Experiment: {selected_id} · Checkpoint: {path_label(checkpoint_path)}")

evaluate_action = (
    WorkflowAction.EVALUATE_TEST if split == "test" else WorkflowAction.EVALUATE_VALIDATION
)
action_col, gradcam_col = st.columns(2)
if action_col.button(
    "Chạy đánh giá",
    type="primary",
    width="stretch",
    disabled=evaluate_action not in state.allowed_actions,
):
    try:
        job = coordinator.execute(
            evaluate_action,
            experiment_id=selected_id,
            options={"checkpoint": checkpoint_kind},
        )
        show_success(f"Đã tạo job {job.job_id}.")
        st.rerun()
    except Exception as error:
        show_error(str(error))

gradcam_action = WorkflowAction.GENERATE_GRADCAM
if gradcam_col.button(
    "Tạo Grad-CAM từ Test",
    disabled=gradcam_action not in state.allowed_actions,
    width="stretch",
):
    try:
        job = coordinator.execute(
            gradcam_action,
            experiment_id=selected_id,
            options={"checkpoint": checkpoint_kind, "samples": 20},
        )
        show_success(f"Đã tạo job {job.job_id}.")
        st.rerun()
    except Exception as error:
        show_error(str(error))

metrics = experiment_state.test_metrics if split == "test" else experiment_state.validation_metrics
output_dir = experiment.evaluation_dir(split)
metrics_path = experiment.metrics_path(split)
if metrics is None:
    stale = experiment_state.test_stale if split == "test" else experiment_state.validation_stale
    message = f"Chưa có kết quả {split} hiện hành cho {selected_id}."
    if stale:
        message += " Artefact đang có đã stale và cần đánh giá lại."
    st.info(message)
    st.stop()

st.subheader("Chỉ số tổng hợp")
metric_columns = st.columns(5)
metric_columns[0].metric("Accuracy", f"{float(metrics.get('accuracy', 0)):.4f}")
metric_columns[1].metric("Macro Precision", f"{float(metrics.get('macro_precision', 0)):.4f}")
metric_columns[2].metric("Macro Recall", f"{float(metrics.get('macro_recall', 0)):.4f}")
metric_columns[3].metric("Macro-F1", f"{float(metrics.get('macro_f1', 0)):.4f}")
metric_columns[4].metric(
    "Loss",
    "—" if metrics.get("loss") is None else f"{float(metrics['loss']):.4f}",
)
st.caption(f"Split fingerprint: {metrics.get('split_fingerprint', '—')}")

figures = output_dir / "figures"
matrix_columns = st.columns(2)
matrix_paths = [
    ("Confusion matrix", figures / "confusion_matrix.png"),
    ("Confusion matrix chuẩn hóa", figures / "confusion_matrix_normalized.png"),
]
for column, (caption, image_path) in zip(matrix_columns, matrix_paths, strict=True):
    with column:
        if image_path.is_file():
            st.image(str(image_path), caption=caption, width="stretch")

training_curves = figures / "training_curves.png"
if training_curves.is_file():
    st.subheader("Đường cong huấn luyện")
    st.image(str(training_curves), caption="Loss, Accuracy và Macro-F1", width="stretch")

report_path = output_dir / "classification_report.csv"
if report_path.is_file():
    st.subheader("Classification report")
    st.dataframe(pd.read_csv(report_path), width="stretch", hide_index=True)

confusions_path = output_dir / "top_confusions.csv"
if confusions_path.is_file():
    st.subheader("Các cặp lớp dễ nhầm")
    confusions = pd.read_csv(confusions_path)
    if confusions.empty:
        show_warning("Không có cặp nhầm lẫn trong kết quả này.")
    else:
        st.dataframe(confusions, width="stretch", hide_index=True)

st.subheader("Tải kết quả")
download_columns = st.columns(3)
for column, path, mime in (
    (download_columns[0], metrics_path, "application/json"),
    (download_columns[1], report_path, "text/csv"),
    (download_columns[2], output_dir / "predictions.csv", "text/csv"),
):
    if path.is_file():
        column.download_button(
            f"Tải {path.name}",
            data=path.read_bytes(),
            file_name=f"{selected_id}_{split}_{path.name}",
            mime=mime,
            width="stretch",
        )
