from __future__ import annotations

import contextlib
import json
import tempfile
import time
from pathlib import Path

import streamlit as st

from app.components.charts import render_probability_chart
from app.components.layout import get_registry, page_header, path_label, show_error
from realwaste.inference import Predictor


@st.cache_resource(show_spinner=False)
def _load_predictor(checkpoint_path: str, modified_ns: int) -> Predictor:
    del modified_ns
    return Predictor(checkpoint_path)


def _available_checkpoints() -> list[tuple[str, Path]]:
    choices: list[tuple[str, Path, float, bool]] = []
    for experiment in get_registry().list():
        score = float("-inf")
        metrics_path = experiment.existing_metrics_path("test")
        if metrics_path is not None:
            with contextlib.suppress(OSError, ValueError, KeyError, json.JSONDecodeError):
                score = float(json.loads(metrics_path.read_text(encoding="utf-8"))["macro_f1"])
        if experiment.best_checkpoint.is_file():
            choices.append(
                (f"{experiment.experiment_id} · best.pt", experiment.best_checkpoint, score, True)
            )
        if experiment.last_checkpoint.is_file():
            choices.append(
                (f"{experiment.experiment_id} · last.pt", experiment.last_checkpoint, score, False)
            )
    choices.sort(key=lambda item: (item[2], item[3], item[1].stat().st_mtime_ns), reverse=True)
    return [(label, path) for label, path, _score, _is_best in choices]


page_header(
    "Dự đoán ảnh rác",
    "Chọn checkpoint đã huấn luyện, tải một ảnh lên và xem các lớp có xác suất cao nhất.",
)

checkpoints = _available_checkpoints()
if not checkpoints:
    st.info(
        "Chưa tìm thấy `best.pt` hoặc `last.pt`. Hãy huấn luyện ít nhất một experiment ở trang "
        "Huấn luyện trước."
    )
    st.stop()

labels = [label for label, _path in checkpoints]
selected_label = st.selectbox("Checkpoint", labels)
checkpoint = dict(checkpoints)[selected_label]
top_k = st.slider("Số lớp hiển thị", min_value=1, max_value=5, value=3)
uploaded = st.file_uploader(
    "Tải ảnh cần phân loại",
    type=["jpg", "jpeg", "png", "webp"],
    help="Ứng dụng không lưu lâu dài ảnh bạn tải lên.",
)

if uploaded is None:
    st.caption(f"Đang chọn: `{path_label(checkpoint)}`")
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
except Exception as error:  # Streamlit boundary: show a friendly message, log details in terminal.
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
        f"Model: `{predictor.config['model']['name']}` · "
        f"Thời gian suy luận: {elapsed_ms:.1f} ms · Checkpoint: `{path_label(checkpoint)}`"
    )
