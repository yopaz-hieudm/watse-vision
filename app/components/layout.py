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
    ("compare", "So sánh", "Chọn mô hình"),
    ("predict", "Dự đoán", "Thử ảnh thực tế"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTED_EXPERIMENT_KEY = "realwaste_selected_experiment_id"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rw-green-700: #15803d;
            --rw-green-600: #16a34a;
            --rw-green-50: #f0fdf4;
            --rw-slate-900: #17211b;
            --rw-slate-600: #64748b;
            --rw-slate-300: #cbd5e1;
            --rw-slate-200: #e2e8f0;
            --rw-slate-50: #f8fafc;
            --rw-radius: 0.9rem;
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 4rem;
            max-width: 1240px;
        }
        .rw-page-header {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        .rw-page-icon {
            display: grid;
            place-items: center;
            flex: 0 0 3rem;
            height: 3rem;
            border-radius: 0.85rem;
            background: var(--rw-green-50);
            border: 1px solid #bbf7d0;
            font-size: 1.45rem;
        }
        .rw-eyebrow {
            color: var(--rw-green-700);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            line-height: 1.2;
            margin: 0 0 0.28rem;
            text-transform: uppercase;
        }
        .rw-page-title {
            color: var(--rw-slate-900);
            font-size: clamp(1.75rem, 3vw, 2.35rem);
            font-weight: 760;
            letter-spacing: -0.035em;
            line-height: 1.12;
            margin: 0;
        }
        .rw-subtitle {
            color: var(--rw-slate-600);
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.42rem 0 0;
            max-width: 780px;
        }
        .rw-workflow {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.5rem;
            margin: 0 0 1.6rem;
        }
        .rw-workflow-step {
            min-height: 4.2rem;
            padding: 0.72rem 0.8rem;
            border: 1px solid var(--rw-slate-200);
            border-radius: 0.75rem;
            background: #fff;
        }
        .rw-workflow-step.is-current {
            border-color: var(--rw-green-600);
            background: var(--rw-green-50);
            box-shadow: inset 0 0 0 1px var(--rw-green-600);
        }
        .rw-workflow-step.is-complete {border-color: #bbf7d0;}
        .rw-workflow-step.is-partial {border-color: #fde68a;}
        .rw-step-index {
            color: var(--rw-slate-600);
            font-size: 0.7rem;
            font-weight: 750;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .rw-workflow-step.is-current .rw-step-index,
        .rw-workflow-step.is-complete .rw-step-index {color: var(--rw-green-700);}
        .rw-step-title {
            color: var(--rw-slate-900);
            font-size: 0.92rem;
            font-weight: 720;
            line-height: 1.2;
            margin-top: 0.16rem;
        }
        .rw-step-description {
            color: var(--rw-slate-600);
            font-size: 0.74rem;
            line-height: 1.3;
            margin-top: 0.16rem;
        }
        .rw-section-header {margin: 2rem 0 0.85rem;}
        .rw-section-title {
            color: var(--rw-slate-900);
            font-size: 1.2rem;
            font-weight: 740;
            letter-spacing: -0.015em;
            line-height: 1.3;
            margin: 0;
        }
        .rw-section-description {
            color: var(--rw-slate-600);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0.22rem 0 0;
        }
        .rw-panel-heading {
            color: var(--rw-slate-900);
            font-size: 1rem;
            font-weight: 720;
            margin: 0 0 0.15rem;
        }
        .rw-panel-description {
            color: var(--rw-slate-600);
            font-size: 0.86rem;
            line-height: 1.45;
            margin: 0 0 0.85rem;
        }
        .rw-empty-state {
            border: 1px dashed var(--rw-slate-300);
            border-radius: var(--rw-radius);
            background: var(--rw-slate-50);
            padding: 2rem 1.25rem;
            text-align: center;
        }
        .rw-empty-icon {font-size: 1.7rem; margin-bottom: 0.45rem;}
        .rw-empty-title {font-weight: 720; margin-bottom: 0.25rem;}
        .rw-empty-description {color: var(--rw-slate-600); font-size: 0.9rem;}
        [data-testid="stMetric"] {
            background: color-mix(in srgb, var(--background-color) 92%, #16a34a 8%);
            border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            border-radius: var(--rw-radius);
            padding: 0.9rem 1rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rw-slate-200);
            border-radius: var(--rw-radius);
        }
        .stButton > button, .stDownloadButton > button {
            min-height: 2.65rem;
            border-radius: 0.7rem;
            font-weight: 650;
        }
        [data-testid="stDataFrame"] {border-radius: 0.75rem; overflow: hidden;}
        .rw-ok {color: #15803d; font-weight: 650;}
        .rw-warn {color: #b45309; font-weight: 650;}
        .rw-error {color: #b91c1c; font-weight: 650;}
        .rw-muted {color: #64748b; font-size: 0.92rem;}
        @media (max-width: 900px) {
            .rw-workflow {grid-template-columns: 1fr;}
            .rw-workflow-step {min-height: auto;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    description: str,
    *,
    icon: str = "♻️",
    eyebrow: str = "RealWaste Control Center",
) -> None:
    inject_styles()
    st.markdown(
        f"""
        <header class="rw-page-header">
            <div class="rw-page-icon" aria-hidden="true">{escape(icon)}</div>
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
    columns = st.columns(len(WORKFLOW_STEPS))
    status_labels = {
        "complete": "Hoàn tất",
        "partial": "Đang thực hiện",
        "pending": "Chưa sẵn sàng",
    }
    for index, (column, (step_id, title, description)) in enumerate(
        zip(columns, WORKFLOW_STEPS, strict=True), start=1
    ):
        status = statuses[step_id]
        marker = "✓" if status == "complete" else f"{index:02d}"
        with column.container(border=True):
            st.page_link(destinations[step_id], label=f"{marker} · {title}")
            st.caption(description)
            detail = status_labels[status]
            st.caption(f"{detail} · Đang xem" if current == step_id else detail)

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
        st.info("Job đang chạy. Trạng thái và thao tác sẽ cập nhật khi job kết thúc.")
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


@st.cache_resource
def get_job_manager():
    from realwaste.ui.job_manager import JobManager

    return JobManager(PROJECT_ROOT)


@st.cache_resource
def get_workflow_coordinator():
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
        "Experiment đang chọn",
        experiment_ids,
        key=SELECTED_EXPERIMENT_KEY,
        help="Ngữ cảnh này được dùng chung ở Huấn luyện, Đánh giá và Dự đoán.",
    )
    experiment = get_registry().get(selected)
    augmentation = "Có augmentation" if experiment.augmentation_enabled else "Baseline"
    st.caption(f"{experiment.model_name} · {augmentation}")
    return str(selected)


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
