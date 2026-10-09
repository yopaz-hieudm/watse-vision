from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"


def test_pages_do_not_start_jobs_without_coordinator() -> None:
    offenders = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ".start_job(" in source:
            offenders.append(path.name)

    assert offenders == []


def test_experiment_selection_persists_in_shared_session_state() -> None:
    source = """
import streamlit as st
from app.components.layout import get_selected_experiment_id, render_experiment_selector

render_experiment_selector()
st.write(get_selected_experiment_id())
"""
    app = AppTest.from_string(source, default_timeout=20)
    app.run()
    app.selectbox[0].select("E3").run()

    assert app.session_state["realwaste_selected_experiment_id"] == "E3"
    assert any("E3" in item.value for item in app.markdown)
