from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.layout import (
    PROJECT_ROOT,
    get_workflow_coordinator,
    get_workflow_state,
    model_label,
    page_header,
    show_error,
    show_success,
    show_warning,
)
from realwaste.workflow import WorkflowAction

state = get_workflow_state()
coordinator = get_workflow_coordinator()

page_header(
    "So sánh E1–E4",
    "So sánh độ chính xác và ảnh hưởng của tăng cường dữ liệu trên cùng tập Test.",
    eyebrow="REALWASTE / KẾT QUẢ VÀ ỨNG DỤNG",
)

rows = []
missing = []
for experiment in state.experiments.values():
    metrics = experiment.test_metrics
    if metrics is None:
        missing.append(experiment.experiment_id)
    rows.append(
        {
            "Thí nghiệm": experiment.experiment_id,
            "Mô hình": model_label(experiment.model_name),
            "Tăng cường dữ liệu": "Có" if experiment.augmentation_enabled else "Không",
            "Trạng thái": "Đã đánh giá" if metrics else "Cần đánh giá",
            "Accuracy": float(metrics["accuracy"]) * 100
            if metrics and "accuracy" in metrics
            else None,
            "Macro-F1": float(metrics["macro_f1"]) * 100
            if metrics and "macro_f1" in metrics
            else None,
            "Ảnh phản hồi": metrics.get("feedback_count", 0) if metrics else None,
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    width="stretch",
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100),
        "Macro-F1": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100),
    },
)

if missing:
    show_warning(f"Cần đánh giá Test hiện hành cho: {', '.join(missing)}.")
    st.page_link(
        str(PROJECT_ROOT / "app/pages/3_Danh_gia.py"),
        label="Đến trang Đánh giá để bổ sung kết quả",
        icon=":material/arrow_forward:",
    )

compare_action = WorkflowAction.COMPARE_EXPERIMENTS
if not missing:
    for blocker in state.blockers_for(compare_action):
        show_warning(blocker)
if st.button(
    "Tạo lại bảng so sánh",
    disabled=compare_action not in state.allowed_actions,
    type="primary",
):
    try:
        job = coordinator.execute(compare_action)
        show_success(f"Đã tạo job {job.job_id}.")
        st.rerun()
    except Exception as error:
        show_error(str(error))

comparison_dir = PROJECT_ROOT / "outputs/comparison"
summary_path = comparison_dir / "experiment_summary.csv"
impact_path = comparison_dir / "augmentation_impact.csv"
chart_path = comparison_dir / "metrics_comparison.png"

if state.comparison_ready:
    st.subheader("Bảng xếp hạng")
    summary = pd.read_csv(summary_path).sort_values("macro_f1", ascending=False)
    st.dataframe(summary, width="stretch", hide_index=True)
    st.download_button(
        "Tải experiment_summary.csv",
        data=summary_path.read_bytes(),
        file_name=summary_path.name,
        mime="text/csv",
    )

    st.image(str(chart_path), caption="Accuracy và Macro-F1", width="stretch")

    st.subheader("Ảnh hưởng của augmentation")
    impact = pd.read_csv(impact_path)
    st.dataframe(impact, width="stretch", hide_index=True)
    st.download_button(
        "Tải augmentation_impact.csv",
        data=impact_path.read_bytes(),
        file_name=impact_path.name,
        mime="text/csv",
    )
else:
    st.info("Bảng xếp hạng sẽ sẵn sàng sau khi cả bốn thí nghiệm được đánh giá trên tập Test.")
