from __future__ import annotations

import pandas as pd
import streamlit as st


def render_history_charts(history: pd.DataFrame) -> None:
    if history.empty:
        st.info("Chưa có lịch sử huấn luyện.")
        return
    frame = history.copy()
    epoch_column = "epoch" if "epoch" in frame.columns else None
    if epoch_column:
        frame = frame.set_index(epoch_column)

    loss_columns = [column for column in ("train_loss", "val_loss") if column in frame]
    score_columns = [
        column
        for column in (
            "train_accuracy",
            "val_accuracy",
            "train_macro_f1",
            "val_macro_f1",
        )
        if column in frame
    ]
    left, right = st.columns(2)
    with left:
        st.subheader("Loss")
        if loss_columns:
            st.line_chart(frame[loss_columns], height=300)
        else:
            st.caption("Chưa có cột loss.")
    with right:
        st.subheader("Accuracy và Macro-F1")
        if score_columns:
            st.line_chart(frame[score_columns], height=300)
        else:
            st.caption("Chưa có cột score.")


def render_probability_chart(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if {"class_name", "probability"}.issubset(frame.columns):
        frame = frame.set_index("class_name")[["probability"]]
        st.bar_chart(frame, horizontal=True, height=max(220, len(frame) * 48))
