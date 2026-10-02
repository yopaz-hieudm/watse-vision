from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realwaste.evaluation.ensemble import load_and_combine_predictions


def _predictions(path, probabilities):
    frame = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg"],
            "true_label": [0, 1],
            "true_class": ["A", "B"],
            "probability_A": [row[0] for row in probabilities],
            "probability_B": [row[1] for row in probabilities],
        }
    )
    frame.to_csv(path, index=False)


def test_combines_aligned_probabilities(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _predictions(first, [[0.8, 0.2], [0.4, 0.6]])
    _predictions(second, [[0.6, 0.4], [0.2, 0.8]])

    _frame, probabilities, columns = load_and_combine_predictions([first, second], [0.25, 0.75])

    assert columns == ["probability_A", "probability_B"]
    np.testing.assert_allclose(probabilities, [[0.65, 0.35], [0.25, 0.75]])


def test_rejects_weights_that_do_not_sum_to_one(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _predictions(first, [[0.8, 0.2], [0.4, 0.6]])
    _predictions(second, [[0.6, 0.4], [0.2, 0.8]])

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_and_combine_predictions([first, second], [0.7, 0.4])
