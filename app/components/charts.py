from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def _history_chart(frame: pd.DataFrame, metric: str, *, percentage: bool = False) -> None:
    columns = [name for name in (f"train_{metric}", f"val_{metric}") if name in frame]
    if not columns:
        st.caption("Chưa có dữ liệu cho chỉ số này.")
        return
    data = frame[["epoch", *columns]].melt("epoch", var_name="Tập", value_name="Giá trị")
    data["Tập"] = data["Tập"].map({f"train_{metric}": "Train", f"val_{metric}": "Validation"})
    axis = alt.Axis(format=".0%") if percentage else alt.Axis()
    scale = alt.Scale(domain=[0, 1]) if percentage else alt.Scale(zero=True)
    chart = (
        alt.Chart(data)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=28))
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Giá trị:Q", title=None, axis=axis, scale=scale),
            color=alt.Color(
                "Tập:N",
                scale=alt.Scale(domain=["Train", "Validation"], range=["#7e99b3", "#187b66"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("epoch:Q", title="Epoch", format="d"),
                alt.Tooltip("Tập:N"),
                alt.Tooltip("Giá trị:Q", format=".2%" if percentage else ".4f"),
            ],
        )
        .properties(height=280)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def render_history_charts(history: pd.DataFrame) -> None:
    if history.empty:
        st.info("Chưa có lịch sử huấn luyện.")
        return
    frame = history.copy()
    if "epoch" not in frame:
        frame["epoch"] = range(1, len(frame) + 1)
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Sai số qua từng vòng học**")
        st.caption("Loss càng thấp, sai số trên tập dữ liệu càng nhỏ.")
        _history_chart(frame, "loss")
    with right.container(border=True):
        st.markdown("**Chất lượng phân loại**")
        accuracy_tab, f1_tab = st.tabs(["Accuracy", "Macro-F1"])
        with accuracy_tab:
            _history_chart(frame, "accuracy", percentage=True)
        with f1_tab:
            _history_chart(frame, "macro_f1", percentage=True)


def render_probability_chart(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if {"class_name", "probability"}.issubset(frame.columns):
        chart = (
            alt.Chart(frame)
            .mark_bar(color="#187b66", cornerRadiusEnd=5, size=24)
            .encode(
                x=alt.X(
                    "probability:Q",
                    title="Độ tin cậy",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%"),
                ),
                y=alt.Y("class_name:N", title=None, sort="-x"),
                tooltip=[
                    alt.Tooltip("class_name:N", title="Lớp"),
                    alt.Tooltip("probability:Q", title="Độ tin cậy", format=".2%"),
                ],
            )
            .properties(height=max(200, len(frame) * 48))
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, width="stretch")
