from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.components.charts import render_history_charts
from app.components.job_status import render_live_job
from app.components.layout import (
    PROJECT_ROOT,
    device_details,
    get_job_manager,
    get_registry,
    page_header,
    path_label,
    show_error,
    show_success,
    show_warning,
)
from realwaste.training.checkpoint import validate_resume_checkpoint

page_header(
    "Huấn luyện E1–E4",
    "Chọn cấu hình cố định, chạy train nền và theo dõi tiến độ mà không khóa giao diện.",
)

registry = get_registry()
experiment_ids = list(registry.experiment_ids)
selected_id = st.selectbox("Experiment", experiment_ids)
experiment = registry.get(selected_id)
config = experiment.config

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
    optimizer = config["training"]["optimizer"]
    loss = config["training"]["loss"]
    scheduler = config["training"]["scheduler"]
    st.json(
        {
            "experiment_id": selected_id,
            "config": path_label(experiment.config_path),
            "model": config["model"],
            "augmentation": config["data"]["augmentation"],
            "batch_size": config["data"]["batch_size"],
            "optimizer": optimizer,
            "loss": loss,
            "scheduler": scheduler,
            "checkpoint": config["checkpoint"],
            "output": path_label(experiment.output_dir),
        }
    )

required_paths = {
    "Dataset": PROJECT_ROOT / config["data"]["root"],
    "Train CSV": PROJECT_ROOT / config["data"]["train_csv"],
    "Validation CSV": PROJECT_ROOT / config["data"]["val_csv"],
    "Test CSV": PROJECT_ROOT / config["data"]["test_csv"],
    "Split metadata": PROJECT_ROOT / "data/splits/split_metadata.json",
}
missing = [label for label, path in required_paths.items() if not path.exists()]
if missing:
    show_error(f"Thiếu đầu vào: {', '.join(missing)}. Hãy hoàn tất trang Dữ liệu trước.")

manager = get_job_manager()
try:
    active_job = manager.get_active_job()
except Exception as error:
    active_job = None
    show_error(f"Không kiểm tra được job đang chạy: {error}")

has_existing_run = any(
    path.exists()
    for path in (experiment.best_checkpoint, experiment.last_checkpoint, experiment.history_path)
)
resume_error: str | None = None
if experiment.last_checkpoint.is_file():
    try:
        split_metadata = json.loads(
            (PROJECT_ROOT / "data/splits/split_metadata.json").read_text(encoding="utf-8")
        )
        validate_resume_checkpoint(
            experiment.last_checkpoint,
            expected_split_fingerprint=str(split_metadata["fingerprint"]),
            expected_experiment_id=experiment.experiment_id,
            expected_model_name=experiment.model_name,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        resume_error = str(error)
action_columns = st.columns(2)
start_disabled = bool(missing) or active_job is not None or has_existing_run
if action_columns[0].button(
    "Bắt đầu huấn luyện mới",
    type="primary",
    disabled=start_disabled,
    width="stretch",
):
    try:
        job = manager.start_job("train", selected_id)
        show_success(f"Đã tạo job `{job.job_id}`.")
        st.rerun()
    except Exception as error:
        show_error(str(error))

resume_disabled = (
    bool(missing)
    or active_job is not None
    or not experiment.last_checkpoint.is_file()
    or resume_error is not None
)
if action_columns[1].button(
    "Tiếp tục từ last.pt",
    disabled=resume_disabled,
    width="stretch",
):
    try:
        job = manager.start_job("train", selected_id, {"resume": True})
        show_success(f"Đã tạo job resume `{job.job_id}`.")
        st.rerun()
    except Exception as error:
        show_error(str(error))
if resume_error is not None:
    show_warning(f"Không thể resume checkpoint hiện tại: {resume_error}")


if has_existing_run:
    st.caption(
        "Experiment đã có artefact; nút train mới được khóa để tránh ghi đè. "
        "Dùng resume từ `last.pt` hoặc giữ kết quả hiện tại."
    )

st.subheader("Job nền")
render_live_job(show_recent=True)

st.subheader("Lịch sử huấn luyện")
history_path = experiment.history_path
if history_path.is_file():
    try:
        render_history_charts(pd.read_csv(history_path))
    except (OSError, pd.errors.ParserError) as error:
        show_error(f"Không đọc được history.csv: {error}")
else:
    st.info("Chưa có `history.csv` cho experiment này.")
