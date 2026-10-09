from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from app.components.charts import render_probability_chart
from app.components.layout import (
    get_workflow_state,
    page_header,
    path_label,
    render_workflow,
    show_error,
)
from realwaste.inference import Predictor
from realwaste.workflow.state import WorkflowState


@st.cache_resource(show_spinner=False)
def _load_predictor(checkpoint_path: str, modified_ns: int) -> Predictor:
    del modified_ns
    return Predictor(checkpoint_path)


def _available_checkpoints(state: WorkflowState) -> list[tuple[str, Path]]:
    choices: list[tuple[str, Path, float, bool, bool]] = []
    for experiment in state.experiments.values():
        score = float("-inf")
        if experiment.test_metrics is not None:
            try:
                score = float(experiment.test_metrics.get("macro_f1", float("-inf")))
            except (TypeError, ValueError):
                score = float("-inf")
        if experiment.best_checkpoint is not None and experiment.best_checkpoint_compatible:
            choices.append(
                (
                    f"{experiment.experiment_id} · best.pt",
                    experiment.best_checkpoint,
                    score,
                    experiment.experiment_id == state.selected_experiment_id,
                    True,
                )
            )
        if experiment.last_checkpoint is not None and experiment.last_checkpoint_compatible:
            choices.append(
                (
                    f"{experiment.experiment_id} · last.pt",
                    experiment.last_checkpoint,
                    score,
                    experiment.experiment_id == state.selected_experiment_id,
                    False,
                )
            )
    choices.sort(
        key=lambda item: (item[2], item[3], item[4], item[1].stat().st_mtime_ns),
        reverse=True,
    )
    return [(label, path) for label, path, _score, _selected, _best in choices]


state = get_workflow_state()
page_header(
    "Dự đoán ảnh rác",
    "Dùng checkpoint hiện hành trong cùng workflow để phân loại một ảnh mới.",
)
render_workflow("predict", state)

checkpoints = _available_checkpoints(state)
if not checkpoints:
    st.info("Chưa tìm thấy checkpoint tương thích. Hãy huấn luyện experiment đang chọn trước.")
    st.stop()

labels = [label for label, _path in checkpoints]
default_index = 0
if state.inference_checkpoint is not None:
    default_index = next(
        (
            index
            for index, (_label, path) in enumerate(checkpoints)
            if path == state.inference_checkpoint
        ),
        0,
    )
selected_label = st.selectbox("Checkpoint", labels, index=default_index)
checkpoint = dict(checkpoints)[selected_label]
top_k = st.slider("Số lớp hiển thị", min_value=1, max_value=5, value=3)
uploaded = st.file_uploader(
    "Tải ảnh cần phân loại",
    type=["jpg", "jpeg", "png", "webp"],
    help="Ứng dụng không lưu lâu dài ảnh bạn tải lên.",
)

st.caption(
    f"Ngữ cảnh: {state.selected_experiment_id} · Checkpoint suy luận: {path_label(checkpoint)}"
)
if uploaded is None:
    st.stop()

if uploaded.size > 20 * 1024 * 1024:
    show_error("Ảnh vượt quá giới hạn 20 MB.")
    st.stop()

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.image(uploaded, caption=uploaded.name, width="stretch")

try:
    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
    with tempfile.TemporaryDirectory(prefix="realwaste_ui_") as temporary_directory:
        image_path = Path(temporary_directory) / f"upload{suffix}"
        image_path.write_bytes(uploaded.getbuffer())
        predictor = _load_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns)
        started = time.perf_counter()
        predictions = predictor.predict(image_path, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
except Exception as error:
    show_error(f"Không thể dự đoán ảnh: {error}")
    st.stop()

winner = predictions[0]
with right:
    st.subheader("Kết quả")
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Lớp dự đoán", winner["class_name"])
    metric_right.metric("Độ tin cậy", f"{winner['probability']:.2%}")
    render_probability_chart(predictions)
    st.caption(
        f"Model: {predictor.config['model']['name']} · "
        f"Thời gian suy luận: {elapsed_ms:.1f} ms · "
        f"Checkpoint: {path_label(checkpoint)}"
    )
