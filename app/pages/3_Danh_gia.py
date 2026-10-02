from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.layout import (
    get_job_manager,
    get_registry,
    page_header,
    path_label,
    show_error,
    show_success,
    show_warning,
)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


page_header(
    "Đánh giá mô hình",
    "Chạy Validation/Test cho checkpoint và xem metric, confusion matrix cùng lỗi phổ biến.",
)

registry = get_registry()
experiments = registry.list()
available = [
    experiment
    for experiment in experiments
    if experiment.best_checkpoint.is_file() or experiment.last_checkpoint.is_file()
]
if not available:
    st.info("Chưa có checkpoint. Hãy huấn luyện ít nhất một experiment trước.")
    st.stop()

experiment_ids = [experiment.experiment_id for experiment in available]
selected_id = st.selectbox("Experiment", experiment_ids)
experiment = registry.get(selected_id)
checkpoint_options = []
if experiment.best_checkpoint.is_file():
    checkpoint_options.append("best")
if experiment.last_checkpoint.is_file():
    checkpoint_options.append("last")
checkpoint_kind = st.radio("Checkpoint", checkpoint_options, horizontal=True)
split = st.radio("Tập dữ liệu", ["test", "validation"], horizontal=True)

checkpoint_path = (
    experiment.best_checkpoint if checkpoint_kind == "best" else experiment.last_checkpoint
)
st.caption(f"Checkpoint: `{path_label(checkpoint_path)}`")

action_col, gradcam_col = st.columns(2)
if action_col.button("Chạy đánh giá", type="primary", width="stretch"):
    try:
        job = get_job_manager().start_job(
            "evaluate",
            selected_id,
            {"checkpoint": checkpoint_kind, "split": split},
        )
        show_success(f"Đã tạo job `{job.job_id}`.")
    except Exception as error:
        show_error(str(error))

test_predictions = experiment.evaluation_dir("test") / "predictions.csv"
if gradcam_col.button(
    "Tạo Grad-CAM từ Test",
    disabled=not test_predictions.is_file(),
    width="stretch",
):
    try:
        job = get_job_manager().start_job(
            "gradcam",
            selected_id,
            {"checkpoint": checkpoint_kind, "samples": 20},
        )
        show_success(f"Đã tạo job `{job.job_id}`.")
    except Exception as error:
        show_error(str(error))

output_dir = experiment.evaluation_dir(split)
metrics_path = experiment.metrics_path(split)
metrics = _read_json(metrics_path)
if not metrics:
    st.info(f"Chưa có kết quả `{split}` cho {selected_id}.")
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
st.caption(f"Split fingerprint: `{metrics.get('split_fingerprint', '—')}`")

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
