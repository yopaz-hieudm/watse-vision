from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.job_status import render_live_job
from app.components.layout import inject_styles

APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="RealWaste Control Center",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

pages = {
    "Pipeline": [
        st.Page(APP_ROOT / "pages/0_Tong_quan.py", title="Tổng quan", icon="🏠", default=True),
        st.Page(APP_ROOT / "pages/1_Du_lieu.py", title="Dữ liệu", icon="🗂️"),
        st.Page(APP_ROOT / "pages/2_Huan_luyen.py", title="Huấn luyện", icon="🧠"),
        st.Page(APP_ROOT / "pages/3_Danh_gia.py", title="Đánh giá", icon="📊"),
        st.Page(APP_ROOT / "pages/4_So_sanh.py", title="So sánh", icon="⚖️"),
    ],
    "Sử dụng": [
        st.Page(APP_ROOT / "pages/5_Du_doan.py", title="Dự đoán", icon="🔎"),
    ],
}

with st.sidebar:
    st.markdown("### Job đang chạy")
    render_live_job(show_recent=False, compact=True)
    st.divider()
    st.caption("RealWaste · MobileNetV3 Small · ResNet18")

navigation = st.navigation(pages, position="sidebar")
navigation.run()
