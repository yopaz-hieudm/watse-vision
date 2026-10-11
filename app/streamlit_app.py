from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.job_status import render_live_job
from app.components.layout import inject_styles, render_experiment_selector

APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="RealWaste · Phân loại rác thải",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()
st.logo(str(APP_ROOT / "assets/logo.svg"), size="large")

pages = {
    "Dự án": [
        st.Page(
            APP_ROOT / "pages/0_Tong_quan.py",
            title="Tổng quan",
            icon=":material/grid_view:",
            default=True,
        ),
    ],
    "Xây dựng mô hình": [
        st.Page(APP_ROOT / "pages/1_Du_lieu.py", title="Dữ liệu", icon=":material/dataset:"),
        st.Page(
            APP_ROOT / "pages/2_Huan_luyen.py", title="Huấn luyện", icon=":material/model_training:"
        ),
        st.Page(APP_ROOT / "pages/3_Danh_gia.py", title="Đánh giá", icon=":material/analytics:"),
    ],
    "Kết quả và ứng dụng": [
        st.Page(APP_ROOT / "pages/4_So_sanh.py", title="So sánh", icon=":material/compare_arrows:"),
        st.Page(APP_ROOT / "pages/5_Du_doan.py", title="Dự đoán", icon=":material/image_search:"),
    ],
}

with st.sidebar:
    st.markdown("### Thí nghiệm")
    render_experiment_selector()
    st.divider()
    st.markdown("### Hoạt động")
    render_live_job(show_recent=False, compact=True)
    st.divider()
    st.caption("Từ dữ liệu đến mô hình phân loại rác thải.")

navigation = st.navigation(pages, position="sidebar")
navigation.run()
