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


def test_reloaded_job_validator_accepts_new_epoch_option(tmp_path, monkeypatch):
    import shutil

    import app.components.layout as layout
    import realwaste.ui.job_manager as job_module
    from realwaste.ui.artifact_reader import ArtifactReader
    from realwaste.ui.experiment_registry import ExperimentRegistry

    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    monkeypatch.setattr(layout, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(layout, "get_registry", lambda: ExperimentRegistry(tmp_path))
    monkeypatch.setattr(layout, "get_reader", lambda: ArtifactReader(tmp_path))
    current_manager = job_module.JobManager

    class PreviousJobManager(current_manager):
        def _build_command(self, kind, experiment_id, args, job_dir):
            self._require_keys(args, {"resume", "include_feedback"})
            return super()._build_command(kind, experiment_id, args, job_dir)

    monkeypatch.setattr(job_module, "JobManager", PreviousJobManager)
    previous = layout.get_workflow_coordinator()
    with pytest.raises(job_module.InvalidJobArgumentsError, match="epochs"):
        previous.job_manager._build_command("train", "E4", {"epochs": 7}, tmp_path / "job")

    # Streamlit reloads the module; a subsequent page run must use the new validator.
    monkeypatch.setattr(job_module, "JobManager", current_manager)
    refreshed = layout.get_workflow_coordinator()
    command = refreshed.job_manager._build_command("train", "E4", {"epochs": 7}, tmp_path / "job")
    assert command[command.index("--epochs") + 1] == "7"
