from __future__ import annotations

import pytest

from realwaste.evaluation.metrics import compute_classification_metrics


def test_metrics_use_macro_average() -> None:
    metrics = compute_classification_metrics([0, 0, 0, 1], [0, 0, 1, 1])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx((1.0 + 0.5) / 2)
    assert metrics["macro_recall"] == pytest.approx(((2 / 3) + 1.0) / 2)
    assert metrics["sample_count"] == 4
