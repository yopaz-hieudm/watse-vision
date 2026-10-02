from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.layout import (
    PROJECT_ROOT,
    get_job_manager,
    get_registry,
    page_header,
    show_error,
    show_success,
    show_warning,
)


def _read_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


page_header(
    "So sánh E1–E4",
    "Xếp hạng theo Test Macro-F1 và đo ảnh hưởng của augmentation trên cùng một split.",
)

experiments = get_registry().list()
rows = []
missing = []
fingerprints: set[str] = set()
for experiment in experiments:
    metrics_path = experiment.metrics_path("test")
    metrics = _read_metrics(metrics_path)
    if (
        metrics is None
        or str(metrics.get("split", "test")).lower() != "test"
        or not metrics.get("split_fingerprint")
    ):
        missing.append(experiment.experiment_id)
        metrics = None
    else:
        fingerprints.add(str(metrics["split_fingerprint"]))
    rows.append(
        {
            "Experiment": experiment.experiment_id,
            "Model": experiment.config["model"]["name"],
            "Augmentation": bool(experiment.config["data"]["augmentation"]["enabled"]),
            "Test metrics": "Đã có" if metrics else "Còn thiếu",
            "Accuracy": metrics.get("accuracy") if metrics else None,
            "Macro-F1": metrics.get("macro_f1") if metrics else None,
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
    show_warning(f"Cần đánh giá Test cho: {', '.join(missing)}.")

fingerprint_mismatch = not missing and len(fingerprints) != 1
if fingerprint_mismatch:
    show_warning("Kết quả E1-E4 không cùng split fingerprint; cần đánh giá lại trước khi so sánh.")

if st.button(
    "Tạo lại bảng so sánh", disabled=bool(missing) or fingerprint_mismatch, type="primary"
):
    try:
        job = get_job_manager().start_job("compare")
        show_success(f"Đã tạo job `{job.job_id}`.")
    except Exception as error:
        show_error(str(error))

comparison_dir = PROJECT_ROOT / "outputs/comparison"
summary_path = comparison_dir / "experiment_summary.csv"
impact_path = comparison_dir / "augmentation_impact.csv"
chart_path = comparison_dir / "metrics_comparison.png"

if summary_path.is_file():
    st.subheader("Bảng xếp hạng")
    summary = pd.read_csv(summary_path).sort_values("macro_f1", ascending=False)
    st.dataframe(summary, width="stretch", hide_index=True)
    st.download_button(
        "Tải experiment_summary.csv",
        data=summary_path.read_bytes(),
        file_name=summary_path.name,
        mime="text/csv",
    )

if chart_path.is_file():
    st.image(str(chart_path), caption="Accuracy và Macro-F1", width="stretch")

if impact_path.is_file():
    st.subheader("Ảnh hưởng của augmentation")
    impact = pd.read_csv(impact_path)
    st.dataframe(impact, width="stretch", hide_index=True)
    st.download_button(
        "Tải augmentation_impact.csv",
        data=impact_path.read_bytes(),
        file_name=impact_path.name,
        mime="text/csv",
    )

if not any(path.is_file() for path in (summary_path, impact_path, chart_path)):
    st.info("Chưa có artefact so sánh. Hãy đánh giá đủ E1–E4 trên Test trước.")
