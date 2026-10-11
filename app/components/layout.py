from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
import torch

from realwaste.workflow.state import WorkflowState

WORKFLOW_STEPS = (
    ("data", "Dữ liệu", "Chuẩn bị và kiểm tra"),
    ("train", "Huấn luyện", "Chạy E1–E4"),
    ("evaluate", "Đánh giá", "Đo lường và giải thích"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTED_EXPERIMENT_KEY = "realwaste_selected_experiment_id"


def inject_styles() -> None:
    css = (PROJECT_ROOT / "app/assets/styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(
    title: str,
    description: str,
    *,
    eyebrow: str = "REALWASTE / KHÔNG GIAN LÀM VIỆC",
) -> None:
    st.markdown(
        f"""
        <header class="rw-page-header">
            <div class="rw-page-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
                stroke-linecap="round" stroke-linejoin="round">
                <path d="M5 15C5 7 12 3 21 3c0 9-4 16-12 16M3 21 16 8"/>
                </svg></div>
            <div>
                <p class="rw-eyebrow">{escape(eyebrow)}</p>
                <h1 class="rw-page-title">{escape(title)}</h1>
                <p class="rw-subtitle">{escape(description)}</p>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_workflow(current: str | None = None, state: WorkflowState | None = None) -> None:
    workflow_state = state or get_workflow_state()
    statuses = workflow_state.workflow_statuses()
    destinations = {
        "data": "app/pages/1_Du_lieu.py",
        "train": "app/pages/2_Huan_luyen.py",
        "evaluate": "app/pages/3_Danh_gia.py",
        "compare": "app/pages/4_So_sanh.py",
        "predict": "app/pages/5_Du_doan.py",
    }
    destinations = {key: str(PROJECT_ROOT / path) for key, path in destinations.items()}
    status_labels = {
        "complete": "Hoàn tất",
        "partial": "Đang thực hiện",
        "pending": "Chưa thực hiện",
    }
    with st.container(key="workflow"):
        columns = st.columns(len(WORKFLOW_STEPS))
        for column, (step_id, title, description) in zip(columns, WORKFLOW_STEPS, strict=True):
            status = statuses[step_id]
            style = "current" if current == step_id else status
            with column.container(key=f"step_{style}_{step_id}"):
                st.page_link(destinations[step_id], label=title, help=description, width="stretch")
                st.caption(
                    "Đang xem" if current == step_id else status_labels[status],
                    text_alignment="center",
                )

    from realwaste.workflow import WorkflowAction

    action = workflow_state.next_action
    action_routes = {
        WorkflowAction.DOWNLOAD_DATA: ("data", "Tải dữ liệu"),
        WorkflowAction.RUN_EDA: ("data", "Kiểm tra dữ liệu và tạo EDA"),
        WorkflowAction.CREATE_SPLIT: ("data", "Tạo tập Train/Validation/Test"),
        WorkflowAction.START_TRAINING: ("train", "Huấn luyện experiment đang chọn"),
        WorkflowAction.RESUME_TRAINING: ("train", "Khôi phục hoặc mở phiên huấn luyện mới"),
        WorkflowAction.EVALUATE_TEST: ("evaluate", "Đánh giá Test của experiment đang chọn"),
        WorkflowAction.COMPARE_EXPERIMENTS: ("compare", "Tạo bảng so sánh"),
    }
    if workflow_state.active_job is not None:
        st.caption("Tiến trình đang chạy · Kết quả được cập nhật tự động.")
    elif (
        workflow_state.selected_experiment.has_training_artifacts
        and workflow_state.selected_experiment.compatible_checkpoint is None
    ):
        st.page_link(
            destinations["train"],
            label="Bước tiếp theo: Lưu kết quả cũ và mở phiên huấn luyện mới",
            icon="➡️",
        )
    elif action in action_routes:
        step_id, label = action_routes[action]
        if step_id != current:
            st.page_link(destinations[step_id], label=f"Bước tiếp theo: {label}", icon="➡️")
    else:
        unfinished = [
            item.experiment_id
            for item in workflow_state.experiments.values()
            if item.test_metrics is None
        ]
        if unfinished:
            st.caption(
                f"Để hoàn tất so sánh, chọn {', '.join(unfinished)} ở sidebar rồi train/đánh giá."
            )
        if workflow_state.inference_checkpoint is not None:
            st.page_link(
                destinations["predict"], label="Dự đoán ảnh mới và gửi phản hồi", icon="🔎"
            )


def section_header(title: str, description: str | None = None) -> None:
    detail = f'<p class="rw-section-description">{escape(description)}</p>' if description else ""
    st.markdown(
        f"""
        <div class="rw-section-header">
            <h2 class="rw-section-title">{escape(title)}</h2>
            {detail}
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def content_panel(
    title: str | None = None,
    description: str | None = None,
) -> Iterator[None]:
    with st.container(border=True):
        if title:
            st.markdown(f'<p class="rw-panel-heading">{escape(title)}</p>', unsafe_allow_html=True)
        if description:
            st.markdown(
                f'<p class="rw-panel-description">{escape(description)}</p>',
                unsafe_allow_html=True,
            )
        yield


def empty_state(title: str, description: str, *, icon: str = "🧭") -> None:
    st.markdown(
        f"""
        <div class="rw-empty-state">
            <div class="rw-empty-icon" aria-hidden="true">{escape(icon)}</div>
            <div class="rw-empty-title">{escape(title)}</div>
            <div class="rw-empty-description">{escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_registry():
    from realwaste.ui.experiment_registry import ExperimentRegistry

    return ExperimentRegistry(PROJECT_ROOT)


@st.cache_resource
def get_reader():
    from realwaste.ui.artifact_reader import ArtifactReader

    return ArtifactReader(PROJECT_ROOT)


def get_job_manager():
    # Job state is persisted on disk. Do not retain old command validators after a reload.
    from realwaste.ui.job_manager import JobManager

    return JobManager(PROJECT_ROOT)


def get_workflow_coordinator():
    # A cached coordinator would also retain its previous JobManager implementation.
    from realwaste.workflow import WorkflowCoordinator

    return WorkflowCoordinator(
        PROJECT_ROOT,
        registry=get_registry(),
        reader=get_reader(),
        job_manager=get_job_manager(),
    )


def get_selected_experiment_id() -> str:
    experiment_ids = get_registry().experiment_ids
    selected = st.session_state.get(SELECTED_EXPERIMENT_KEY)
    if selected not in experiment_ids:
        selected = experiment_ids[0]
        st.session_state[SELECTED_EXPERIMENT_KEY] = selected
    return str(selected)


def render_experiment_selector() -> str:
    experiment_ids = get_registry().experiment_ids
    get_selected_experiment_id()
    selected = st.selectbox(
        "Thí nghiệm đang chọn",
        experiment_ids,
        key=SELECTED_EXPERIMENT_KEY,
        help="Ngữ cảnh này được dùng chung ở Huấn luyện, Đánh giá và Dự đoán.",
    )
    experiment = get_registry().get(selected)
    augmentation = "Tăng cường dữ liệu" if experiment.augmentation_enabled else "Dữ liệu gốc"
    st.caption(f"{model_label(experiment.model_name)} · {augmentation}")
    return str(selected)


def model_label(name: str) -> str:
    return {"mobilenet_v3_small": "MobileNetV3 Small", "resnet18": "ResNet18"}.get(name, name)


def select_experiment(experiment_id: str) -> None:
    st.session_state[SELECTED_EXPERIMENT_KEY] = experiment_id


def get_workflow_state() -> WorkflowState:
    return get_workflow_coordinator().get_state(get_selected_experiment_id())


def device_details() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "CUDA", torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "MPS", "Apple Metal Performance Shaders"
    return "CPU", "Không phát hiện GPU tăng tốc"


def path_label(path: str | Path | None) -> str:
    if path is None:
        return "—"
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_timestamp(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone().strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return value


def show_error(message: str) -> None:
    st.error(message, icon="🚫")


def show_warning(message: str) -> None:
    st.warning(message, icon="⚠️")


def show_success(message: str) -> None:
    st.success(message, icon="✅")
