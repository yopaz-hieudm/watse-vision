from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.job_status import render_live_job
from app.components.layout import (
    device_details,
    get_reader,
    get_registry,
    page_header,
    show_warning,
)
from realwaste.ui.view_models import build_project_overview

page_header(
    "RealWaste Control Center",
    "Theo dõi dữ liệu, huấn luyện, đánh giá và trạng thái bốn thí nghiệm từ một nơi.",
)

registry = get_registry()
overview = build_project_overview(registry, get_reader())
dataset = overview.dataset_summary or {}
split = overview.split_metadata or {}
device_name, device_description = device_details()

metric_columns = st.columns(4)
metric_columns[0].metric("Dataset", dataset.get("valid_images", "—"), help="Số ảnh hợp lệ")
metric_columns[1].metric(
    "Split",
    sum(split.get("counts", {}).values()) if split.get("counts") else "—",
    help="Tổng số ảnh trong ba split",
)
metric_columns[2].metric("Đã train", f"{overview.trained_experiment_count}/4")
metric_columns[3].metric("Đã Test", f"{overview.evaluated_experiment_count}/4")

st.caption(f"Thiết bị: **{device_name}** · {device_description}")
for warning in overview.warnings:
    show_warning(warning)

st.subheader("Trạng thái E1–E4")
rows = []
for experiment in overview.experiments:
    test_metrics = experiment.test_metrics or {}
    rows.append(
        {
            "Experiment": experiment.experiment_id,
            "Model": experiment.model_name,
            "Augmentation": experiment.augmentation_enabled,
            "Checkpoint": "Có" if experiment.has_checkpoint else "Chưa có",
            "Test Accuracy": test_metrics.get("accuracy"),
            "Test Macro-F1": test_metrics.get("macro_f1"),
            "Bước tiếp theo": experiment.next_action,
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Test Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "Test Macro-F1": st.column_config.NumberColumn(format="%.4f"),
    },
)

st.subheader("Job gần nhất")
render_live_job(show_recent=True)

with st.expander("Quy trình đề xuất", expanded=not overview.trained_experiment_count):
    st.markdown(
        """
        1. Kiểm tra EDA và split ở trang **Dữ liệu**.
        2. Chạy E1–E4 lần lượt ở trang **Huấn luyện**.
        3. Đánh giá `best.pt` trên Test ở trang **Đánh giá**.
        4. Khi đủ bốn kết quả, tạo bảng ở trang **So sánh**.
        5. Dùng checkpoint tốt nhất tại trang **Dự đoán**.
        """
    )
