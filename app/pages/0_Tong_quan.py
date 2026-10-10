from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.job_status import render_live_job
from app.components.layout import (
    device_details,
    get_reader,
    get_workflow_state,
    page_header,
    render_workflow,
    show_warning,
)
from realwaste.ui.artifact_reader import ArtifactError

state = get_workflow_state()

page_header(
    "RealWaste Control Center",
    "Theo dõi một workflow thống nhất từ dữ liệu đến dự đoán cho bốn thí nghiệm.",
)
render_workflow(state=state)

try:
    verification = get_reader().read_optional_json_object("outputs/verification/latest.json")
except ArtifactError as error:
    verification = None
    show_warning(f"Không đọc được báo cáo kiểm chứng: {error}")
if verification is not None:
    with st.expander("Kiểm chứng vòng đời bằng ảnh thật"):
        st.write(f"Trạng thái: {verification.get('status', 'unknown')}")
        st.caption(
            "Chạy thử 1 epoch trên một phần dataset thật để kiểm tra luồng; "
            "chất lượng model cần được đánh giá bằng E1–E4 đầy đủ."
        )
        for step in verification.get("steps", []):
            st.write(f"✓ {step}")
        st.download_button(
            "Tải báo cáo kiểm chứng",
            data=get_reader().resolve("outputs/verification/latest.json").read_bytes(),
            file_name="lifecycle_verification.json",
            mime="application/json",
        )

dataset = state.dataset_summary or {}
split = state.split_metadata or {}
device_name, device_description = device_details()

metric_columns = st.columns(4)
metric_columns[0].metric("Dataset", dataset.get("valid_images", "—"), help="Số ảnh hợp lệ")
metric_columns[1].metric(
    "Split",
    sum(split.get("counts", {}).values()) if split.get("counts") else "—",
    help="Tổng số ảnh trong ba split",
)
metric_columns[2].metric("Đã train", f"{state.trained_experiment_count}/4")
metric_columns[3].metric("Đã Test", f"{state.evaluated_experiment_count}/4")

st.caption(
    f"Experiment đang chọn: **{state.selected_experiment_id}** · "
    f"Thiết bị: **{device_name}** · {device_description}"
)
for issue in state.issues:
    show_warning(issue)

st.subheader("Trạng thái E1–E4")
rows = []
for experiment in state.experiments.values():
    test_metrics = experiment.test_metrics or {}
    rows.append(
        {
            "Experiment": experiment.experiment_id,
            "Model": experiment.model_name,
            "Augmentation": experiment.augmentation_enabled,
            "Checkpoint": "Hợp lệ" if experiment.compatible_checkpoint else "Chưa có/Stale",
            "Test Accuracy": test_metrics.get("accuracy"),
            "Test Macro-F1": test_metrics.get("macro_f1"),
            "Bước tiếp theo": experiment.next_action_label,
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

with st.expander("Quy trình đề xuất", expanded=not state.trained_experiment_count):
    st.markdown(
        """
        1. Kiểm tra EDA và split ở trang **Dữ liệu**.
        2. Chọn E1–E4 một lần ở sidebar và huấn luyện tại trang **Huấn luyện**.
        3. Đánh giá checkpoint của cùng experiment tại trang **Đánh giá**.
        4. Khi đủ bốn kết quả Test, tạo bảng ở trang **So sánh**.
        5. Dùng checkpoint hợp lệ tại trang **Dự đoán**.
        """
    )
