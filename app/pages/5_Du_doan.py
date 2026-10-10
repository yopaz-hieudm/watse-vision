from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.charts import render_probability_chart
from app.components.layout import (
    PROJECT_ROOT,
    format_timestamp,
    get_workflow_state,
    page_header,
    path_label,
    render_workflow,
    show_error,
)
from realwaste.constants import CLASS_NAMES
from realwaste.inference import Predictor
from realwaste.inference.history import PredictionHistory, normalize_image
from realwaste.workflow.state import WorkflowState


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_predictor(checkpoint_path: str, modified_ns: int) -> Predictor:
    del modified_ns
    return Predictor(checkpoint_path)


def _available_checkpoints(state: WorkflowState) -> list[tuple[str, Path]]:
    experiment = state.selected_experiment
    choices = []
    if experiment.best_checkpoint is not None and experiment.best_checkpoint_compatible:
        choices.append((f"{experiment.experiment_id} · best.pt", experiment.best_checkpoint))
    if experiment.last_checkpoint is not None and experiment.last_checkpoint_compatible:
        choices.append((f"{experiment.experiment_id} · last.pt", experiment.last_checkpoint))
    return choices


def render_prediction(state: WorkflowState, history: PredictionHistory) -> None:
    if (
        state.active_job is not None
        and state.active_job.kind == "train"
        and state.active_job.experiment_id == state.selected_experiment_id
    ):
        st.info(
            "Experiment đang huấn luyện. Hãy đợi hoặc dừng job trước khi dự đoán bằng model này."
        )
        return
    checkpoints = _available_checkpoints(state)
    if not checkpoints:
        st.info(
            "Experiment đang chọn chưa có checkpoint tương thích. "
            "Chuyển tới Huấn luyện để tạo model."
        )
        return
    selected_id = state.selected_experiment_id
    labels = [label for label, _path in checkpoints]
    selected_label = st.selectbox("Checkpoint", labels, key=f"prediction_checkpoint_{selected_id}")
    checkpoint = dict(checkpoints)[selected_label]
    top_k = st.slider("Số lớp hiển thị", min_value=1, max_value=5, value=3)
    st.caption(f"Experiment: {selected_id} · Model: {path_label(checkpoint)}")
    with st.form(f"prediction_form_{selected_id}"):
        uploaded = st.file_uploader("Tải ảnh cần phân loại", type=["jpg", "jpeg", "png", "webp"])
        retain_image = st.checkbox(
            "Cho phép lưu ảnh để xác nhận nhãn và cải thiện lần huấn luyện sau",
            help="Nếu không chọn, lịch sử chỉ lưu thông tin kết quả, không lưu ảnh.",
        )
        submitted = st.form_submit_button("Dự đoán và lưu kết quả", type="primary")
    result_key = f"prediction_result_{selected_id}"
    signature = (str(checkpoint), checkpoint.stat().st_mtime_ns)
    if submitted:
        if uploaded is None:
            show_error("Hãy chọn một ảnh trước khi dự đoán.")
        else:
            try:
                png, _digest = normalize_image(uploaded.getvalue())
                with tempfile.TemporaryDirectory(prefix="realwaste_ui_") as temporary_directory:
                    image_path = Path(temporary_directory) / "upload.png"
                    image_path.write_bytes(png)
                    predictor = _load_predictor(*signature)
                    started = time.perf_counter()
                    predictions = predictor.predict(image_path, top_k=top_k)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                identifier = history.save(
                    experiment_id=selected_id,
                    checkpoint=checkpoint,
                    filename=uploaded.name,
                    image=png,
                    predictions=predictions,
                    elapsed_ms=elapsed_ms,
                    retain_image=retain_image,
                )
                st.session_state[result_key] = {
                    "signature": signature,
                    "image": png,
                    "predictions": predictions,
                    "elapsed_ms": elapsed_ms,
                    "id": identifier,
                }
            except Exception as error:
                show_error(f"Không thể dự đoán hoặc lưu kết quả: {error}")
    result = st.session_state.get(result_key)
    if result is None or result["signature"] != signature:
        return
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(result["image"], width="stretch")
    with right:
        winner = result["predictions"][0]
        st.subheader("Kết quả")
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Lớp dự đoán", winner["class_name"])
        metric_right.metric("Độ tin cậy", f"{winner['probability']:.2%}")
        render_probability_chart(result["predictions"])
        st.caption(
            f"Thời gian suy luận: {result['elapsed_ms']:.1f} ms. Đã lưu trong lịch sử bên dưới."
        )
        st.download_button(
            "Tải kết quả JSON",
            json.dumps(result["predictions"], ensure_ascii=False, indent=2),
            file_name="prediction.json",
            mime="application/json",
        )


def render_history(history: PredictionHistory, selected_id: str) -> None:
    st.subheader(f"Lịch sử và phản hồi · {selected_id}")
    records = history.list(selected_id)
    if not records:
        st.info("Chưa có dự đoán cho experiment này.")
        return
    frame = pd.DataFrame(
        [
            {
                "ID": record["id"],
                "Thời gian": format_timestamp(record["created_at"]),
                "Ảnh": record["filename"],
                "Checkpoint": record["checkpoint"],
                "Phiên model": record["checkpoint_sha256"][:12],
                "Dự đoán": record["predictions"][0]["class_name"],
                "Độ tin cậy": record["predictions"][0]["probability"],
                "Nhãn xác nhận": record["reviewed_label"],
                "Đã lưu ảnh": bool(record["image_retained"]),
            }
            for record in records
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("Hiển thị tối đa 100 kết quả gần nhất; kết quả cùng ảnh và cùng model được gộp.")
    st.download_button(
        "Tải lịch sử CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"predictions_{selected_id}.csv",
        mime="text/csv",
    )
    with st.expander("Xem ảnh và xác nhận/sửa nhãn"):
        lookup = {record["id"]: record for record in records}
        identifier = st.selectbox(
            "Kết quả cần xem",
            list(lookup),
            key=f"review_record_{selected_id}",
            format_func=lambda value: (
                f"{lookup[value]['filename']} · {format_timestamp(lookup[value]['created_at'])}"
                f" · {value[:8]}"
            ),
        )
        record = lookup[identifier]
        st.write(f"Model đã dự đoán: {record['predictions'][0]['class_name']}")
        image = history.get_image(identifier)
        if image is None:
            st.info(
                "Ảnh này chưa được lưu. Dự đoán lại cùng ảnh và chọn cho phép lưu ảnh "
                "để gửi phản hồi."
            )
            return
        st.image(image, width=280)
        label = record["reviewed_label"] or record["predictions"][0]["class_name"]
        with st.form(f"review_form_{identifier}"):
            corrected = st.selectbox("Nhãn đúng", CLASS_NAMES, index=CLASS_NAMES.index(label))
            confirmed = st.checkbox("Tôi đã kiểm tra ảnh và xác nhận nhãn này")
            save = st.form_submit_button("Lưu nhãn xác nhận")
            clear = st.form_submit_button("Bỏ xác nhận")
        if save or clear:
            try:
                if save and not confirmed:
                    show_error("Hãy kiểm tra và xác nhận nhãn trước khi lưu.")
                else:
                    history.review(identifier, corrected if save else None)
                    st.rerun()
            except Exception as error:
                show_error(str(error))
        st.caption(
            "Sang Huấn luyện, mở phiên mới và chọn bổ sung ảnh đã xác nhận để dùng phản hồi."
        )


state = get_workflow_state()
page_header("Dự đoán ảnh rác", "Phân loại bằng experiment đang chọn, lưu kết quả và xác nhận nhãn.")
render_workflow("predict", state)
history = PredictionHistory(PROJECT_ROOT)
render_prediction(state, history)
render_history(history, state.selected_experiment_id)
