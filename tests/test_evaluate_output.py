from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pytest

EVALUATE_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate.py"
EVALUATE_GLOBALS = run_path(str(EVALUATE_SCRIPT))
resolve_history_path = EVALUATE_GLOBALS["resolve_history_path"]
resolve_output_dir = EVALUATE_GLOBALS["resolve_output_dir"]


def test_evaluation_output_defaults_to_split_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "experiment" / "best.pt"
    assert resolve_output_dir(checkpoint, "validation") == (
        checkpoint.parent / "evaluation" / "validation"
    )
    assert resolve_output_dir(checkpoint, "test") == checkpoint.parent / "evaluation" / "test"


def test_evaluation_output_accepts_explicit_directory(tmp_path: Path) -> None:
    output = tmp_path / "custom"
    assert resolve_output_dir(tmp_path / "best.pt", "test", output) == output


def test_evaluation_output_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split must be validation or test"):
        resolve_output_dir(tmp_path / "best.pt", "train")


def test_training_curves_use_history_next_to_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "experiment" / "best.pt"
    custom_output = tmp_path / "somewhere-else"

    assert resolve_output_dir(checkpoint, "test", custom_output) == custom_output
    assert resolve_history_path(checkpoint) == checkpoint.parent / "history.csv"
    assert resolve_history_path(checkpoint).parent != custom_output
