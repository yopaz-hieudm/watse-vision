from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.gallery import (
    GalleryImage,
    create_thumbnail,
    dataset_signature,
    discover_dataset_images,
    filter_gallery_images,
    paginate_gallery,
)
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
from realwaste.constants import CLASS_NAMES
from realwaste.workflow import WorkflowAction


def _start(action: WorkflowAction, *, options: dict | None = None) -> None:
    try:
        job = get_workflow_coordinator().execute(action, options=options)
        show_success(f"Đã tạo job {job.job_id}. Theo dõi ở thanh bên hoặc trang Tổng quan.")
        st.rerun()
    except Exception as error:
        show_error(str(error))


@st.cache_data(show_spinner=False)
def _load_gallery(
    dataset_root: str,
    signature: tuple[tuple[str, int], ...],
) -> tuple[GalleryImage, ...]:
    del signature
    return discover_dataset_images(dataset_root)


@st.cache_data(show_spinner=False, max_entries=512)
def _load_thumbnail(path: str, modified_ns: int) -> bytes:
    del modified_ns
    return create_thumbnail(path)


state = get_workflow_state()
page_header(
    "Dữ liệu RealWaste",
    "Khám phá các nhóm ảnh và chuẩn bị tập Train, Validation, Test dùng chung cho E1–E4.",
)
render_workflow("data", state)

checksum_path = PROJECT_ROOT / "data/checksums/realwaste.sha256"
eda_dir = PROJECT_ROOT / "outputs/eda"
dataset_root = PROJECT_ROOT / "data/raw/RealWaste"
summary = dict(state.dataset_summary or {})
split = dict(state.split_metadata or {})

status_columns = st.columns(4)
status_columns[0].metric("Ảnh hợp lệ", summary.get("valid_images", "—"))
status_columns[1].metric("Ảnh lỗi", summary.get("invalid_images", "—"))
status_columns[2].metric("Số lớp", summary.get("number_of_classes", "—"))
status_columns[3].metric(
    "Ảnh đã chia tập",
    sum(split.get("counts", {}).values()) if split.get("counts") else "—",
)

with st.expander("Checksum và thao tác dữ liệu", expanded=not bool(summary)):
    if checksum_path.is_file():
        st.code(checksum_path.read_text(encoding="utf-8").strip(), language=None)
    else:
        show_warning("Chưa có checksum. Hãy tải dữ liệu bằng pipeline tin cậy trước.")

    data_action = WorkflowAction.VERIFY_DATA if state.data_ready else WorkflowAction.DOWNLOAD_DATA
    verify_col, eda_col, split_col = st.columns(3)
    if verify_col.button(
        "Kiểm tra dữ liệu" if state.data_ready else "Tải dữ liệu",
        disabled=data_action not in state.allowed_actions,
        width="stretch",
    ):
        _start(data_action)
    if eda_col.button(
        "Tạo lại EDA",
        disabled=WorkflowAction.RUN_EDA not in state.allowed_actions,
        width="stretch",
    ):
        _start(WorkflowAction.RUN_EDA)

    confirm_split = st.checkbox(
        "Tôi hiểu thao tác tạo lại split sẽ ghi đè CSV hiện tại.",
        key="confirm_recreate_split",
    )
    if split_col.button(
        "Tạo lại split",
        disabled=(not confirm_split or WorkflowAction.CREATE_SPLIT not in state.allowed_actions),
        width="stretch",
        type="secondary",
    ):
        _start(WorkflowAction.CREATE_SPLIT)

if not summary:
    st.info("Chưa có EDA. Bấm **Tạo lại EDA** sau khi dataset đã sẵn sàng.")
else:
    st.subheader("Phân bố lớp")
    distribution_image = eda_dir / "class_distribution.png"
    if distribution_image.is_file():
        st.image(str(distribution_image), width="stretch")
    distribution_csv = eda_dir / "class_distribution.csv"
    if distribution_csv.is_file():
        st.dataframe(pd.read_csv(distribution_csv), width="stretch", hide_index=True)

    st.subheader("Ảnh mẫu")
    sample_grid = eda_dir / "sample_grid.png"
    if sample_grid.is_file():
        with st.expander("Lưới EDA — 36 ảnh đại diện (4 ảnh mỗi lớp)"):
            st.image(
                str(sample_grid),
                caption="Lưới cố định được tạo với seed của pipeline EDA.",
                width="stretch",
            )

    st.markdown("#### Duyệt toàn bộ dataset")
    st.caption(
        "Có thể xem mọi ảnh theo từng trang; ứng dụng chỉ tải các ảnh của trang hiện tại "
        "để không làm trình duyệt bị chậm."
    )
    gallery = _load_gallery(str(dataset_root), dataset_signature(dataset_root))
    if not gallery:
        show_warning("Chưa tìm thấy ảnh trong data/raw/RealWaste.")
    else:
        filter_column, size_column, page_column = st.columns([2, 1, 1])
        selected_class = filter_column.selectbox(
            "Lớp",
            ("Tất cả lớp", *CLASS_NAMES),
            key="dataset_gallery_class",
        )
        page_size = size_column.selectbox(
            "Ảnh mỗi trang",
            (12, 24, 48),
            key="dataset_gallery_page_size",
        )
        class_filter = None if selected_class == "Tất cả lớp" else selected_class
        filtered_gallery = filter_gallery_images(gallery, class_filter)
        first_page = paginate_gallery(filtered_gallery, page=1, page_size=page_size)
        page_number = page_column.number_input(
            "Trang",
            min_value=1,
            max_value=first_page.total_pages,
            value=1,
            step=1,
            key=f"dataset_gallery_page_{selected_class}_{page_size}",
        )
        current_page = paginate_gallery(
            filtered_gallery,
            page=page_number,
            page_size=page_size,
        )
        first_item = (current_page.page - 1) * current_page.page_size + 1
        last_item = min(
            current_page.page * current_page.page_size,
            current_page.total_items,
        )
        st.caption(
            f"Đang hiển thị {first_item:,}–{last_item:,} / "
            f"{current_page.total_items:,} ảnh · Trang "
            f"{current_page.page}/{current_page.total_pages}"
        )
        image_columns = st.columns(4)
        for index, gallery_image in enumerate(current_page.items):
            with image_columns[index % len(image_columns)]:
                try:
                    modified_ns = gallery_image.path.stat().st_mtime_ns
                    thumbnail = _load_thumbnail(str(gallery_image.path), modified_ns)
                    st.image(
                        thumbnail,
                        caption=f"{gallery_image.class_name} · {gallery_image.path.name}",
                        width="stretch",
                    )
                except (OSError, ValueError) as error:
                    show_warning(f"Không thể mở {gallery_image.path.name}: {error}")

    st.subheader("Ảnh lỗi")
    invalid_path = eda_dir / "invalid_images.csv"
    if invalid_path.is_file():
        invalid = pd.read_csv(invalid_path)
        if invalid.empty:
            show_success("Không phát hiện ảnh lỗi trong dataset.")
        else:
            st.dataframe(invalid, width="stretch", hide_index=True)
    else:
        show_warning("Chưa có invalid_images.csv.")

st.subheader("Train / Validation / Test")
if not split:
    st.info("Chưa có split metadata.")
else:
    counts = split.get("counts", {})
    split_columns = st.columns(3)
    split_columns[0].metric("Train", counts.get("train", "—"))
    split_columns[1].metric("Validation", counts.get("validation", "—"))
    split_columns[2].metric("Test", counts.get("test", "—"))
    st.caption(f"Seed: {split.get('seed', '—')} · Fingerprint: {split.get('fingerprint', '—')}")
