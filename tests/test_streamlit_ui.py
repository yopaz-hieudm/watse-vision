from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    (
        "app/streamlit_app.py",
        "app/pages/0_Tong_quan.py",
        "app/pages/1_Du_lieu.py",
        "app/pages/2_Huan_luyen.py",
        "app/pages/3_Danh_gia.py",
        "app/pages/4_So_sanh.py",
        "app/pages/5_Du_doan.py",
    ),
)
def test_streamlit_page_renders_without_exception(relative_path: str) -> None:
    app = AppTest.from_file(PROJECT_ROOT / relative_path, default_timeout=20)
    app.run()
    assert not app.exception
