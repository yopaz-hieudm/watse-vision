from __future__ import annotations

from html import escape

import streamlit as st

from app.components.job_status import render_live_job
from app.components.layout import (
    get_workflow_state,
    model_label,
    page_header,
    render_workflow,
    section_header,
    select_experiment,
    show_warning,
)

state = get_workflow_state()
page_header(
    "Từ ảnh rác thải đến mô hình AI.",
    "Chuẩn bị dữ liệu, huấn luyện và khám phá kết quả của bốn thí nghiệm trong một không gian.",
    eyebrow="REALWASTE / TỔNG QUAN",
)
render_workflow(state=state)

dataset = state.dataset_summary or {}
metric_columns = st.columns(4)
image_count = dataset.get("valid_images")
metric_columns[0].metric("Ảnh trong bộ dữ liệu", f"{image_count:,}" if image_count else "—")
metric_columns[1].metric("Nhóm rác thải", dataset.get("number_of_classes", "—"))
metric_columns[2].metric("Đã huấn luyện", f"{state.trained_experiment_count} / 4")
metric_columns[3].metric("Đã đánh giá Test", f"{state.evaluated_experiment_count} / 4")

for issue in state.issues:
    show_warning(issue)

section_header("Các thí nghiệm", "Hai kiến trúc mô hình. Hai cách chuẩn bị dữ liệu.")
experiments = list(state.experiments.values())
for offset in range(0, len(experiments), 2):
    for column, experiment in zip(st.columns(2), experiments[offset : offset + 2], strict=False):
        identifier = experiment.experiment_id
        metrics = experiment.test_metrics
        ready = experiment.compatible_checkpoint is not None
        active = state.active_job
        running = active is not None and active.experiment_id == identifier
        label, style = "Chưa huấn luyện", "is-neutral"
        if running:
            label, style = "Đang xử lý", ""
        elif metrics:
            label, style = "Đã đánh giá", ""
        elif ready:
            label, style = "Sẵn sàng đánh giá", ""
        elif experiment.has_training_artifacts:
            label, style = "Cần kiểm tra phiên cũ", "is-warning"
        augmentation = "Có tăng cường dữ liệu" if experiment.augmentation_enabled else "Dữ liệu gốc"
        score = f"{float(metrics['accuracy']):.2%}" if metrics and "accuracy" in metrics else "—"
        with column.container(border=True, key=f"experiment_card_{identifier}"):
            st.markdown(
                f'<div class="rw-card-top"><span class="rw-experiment-id">{identifier}</span>'
                f'<span class="rw-badge {style}">{label}</span></div>'
                f'<p class="rw-model-name">{escape(model_label(experiment.model_name))}</p>'
                f'<p class="rw-card-detail">{augmentation}</p>'
                f'<div class="rw-card-score">{score}<small>Accuracy trên Test</small></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"Xem kết quả {identifier}"
                if metrics
                else (f"Đánh giá {identifier}" if ready else f"Huấn luyện {identifier}"),
                key=f"open_experiment_{identifier}",
                on_click=select_experiment,
                args=(identifier,),
                width="stretch",
                icon=":material/arrow_forward:",
            ):
                st.switch_page("pages/3_Danh_gia.py" if ready else "pages/2_Huan_luyen.py")

section_header("Hoạt động gần đây", "Tiến độ và kết quả của tác vụ mới nhất.")
with st.container(border=True, key="overview_activity"):
    render_live_job(show_recent=True)
