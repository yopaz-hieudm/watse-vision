from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.layout import (
    PROJECT_ROOT,
    get_workflow_coordinator,
    get_workflow_state,
    page_header,
    render_workflow,
    show_error,
    show_success,
    show_warning,
)
from realwaste.workflow import WorkflowAction

state = get_workflow_state()
coordinator = get_workflow_coordinator()

page_header(
    "So sánh E1–E4",
    "Xếp hạng các Test metrics hiện hành trên cùng split fingerprint.",
)
render_workflow("compare", state)

rows = []
missing = []
for experiment in state.experiments.values():
    metrics = experiment.test_metrics
    if metrics is None:
        missing.append(experiment.experiment_id)
    rows.append(
        {
            "Experiment": experiment.experiment_id,
            "Model": experiment.model_name,
            "Augmentation": experiment.augmentation_enabled,
            "Test metrics": "Hiện hành" if metrics else "Thiếu/Stale",
            "Accuracy": metrics.get("accuracy") if metrics else None,
            "Macro-F1": metrics.get("macro_f1") if metrics else None,
            "Ảnh phản hồi": metrics.get("feedback_count", 0) if metrics else None,
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    width="stretch",
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
    },
)

if missing:
    show_warning(f"Cần đánh giá Test hiện hành cho: {', '.join(missing)}.")

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
    st.info("Chưa có artefact so sánh hiện hành. Hãy đánh giá đủ E1–E4 trên Test và tạo lại bảng.")
