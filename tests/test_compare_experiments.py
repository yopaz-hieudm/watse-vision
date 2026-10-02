from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from realwaste.evaluation.comparison import collect_experiment_metrics, compare_experiments


def _metrics(experiment_id: str, fingerprint: str = "same") -> dict:
    model = "mobilenet_v3_small" if experiment_id in {"E1", "E3"} else "resnet18"
    augmentation = experiment_id in {"E3", "E4"}
    return {
        "experiment_id": experiment_id,
        "model": model,
        "augmentation": augmentation,
        "best_epoch": 3,
        "accuracy": 0.75 + int(experiment_id[-1]) * 0.01,
        "macro_precision": 0.7,
        "macro_recall": 0.71,
        "macro_f1": 0.72 + int(experiment_id[-1]) * 0.01,
        "checkpoint_path": f"{experiment_id}/best.pt",
        "split_fingerprint": fingerprint,
    }


def _write_metrics(root: Path, experiment_id: str, fingerprint: str = "same") -> None:
    directory = root / experiment_id
    directory.mkdir(parents=True)
    metrics = _metrics(experiment_id, fingerprint)
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    config = {
        "experiment_id": experiment_id,
        "seed": 42,
        "model": {"name": metrics["model"]},
        "data": {
            "test_csv": "data/splits/test.csv",
            "augmentation": {"enabled": metrics["augmentation"]},
        },
    }
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (directory / "history.csv").write_text("epoch,val_macro_f1\n3,0.75\n", encoding="utf-8")


def test_comparison_creates_all_outputs(tmp_path: Path) -> None:
    inputs = tmp_path / "experiments"
    for experiment_id in ("E1", "E2", "E3", "E4"):
        _write_metrics(inputs, experiment_id)
    output = tmp_path / "comparison"
    ranked = compare_experiments(inputs, output)
    assert ranked[0]["experiment_id"] == "E4"
    for name in (
        "experiment_summary.csv",
        "experiment_summary.md",
        "metrics_comparison.png",
        "augmentation_impact.csv",
    ):
        assert (output / name).is_file()


def test_comparison_rejects_mismatched_splits(tmp_path: Path) -> None:
    inputs = tmp_path / "experiments"
    for experiment_id in ("E1", "E2", "E3"):
        _write_metrics(inputs, experiment_id)
    _write_metrics(inputs, "E4", fingerprint="different")
    with pytest.raises(ValueError, match="split fingerprint"):
        compare_experiments(inputs, tmp_path / "comparison")


def test_comparison_rejects_mismatched_seeds(tmp_path: Path) -> None:
    inputs = tmp_path / "experiments"
    for experiment_id in ("E1", "E2", "E3", "E4"):
        _write_metrics(inputs, experiment_id)
    config_path = inputs / "E4" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seed"] = 7
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="same seed"):
        compare_experiments(inputs, tmp_path / "comparison")


def test_comparison_prefers_nested_test_metrics_over_legacy_metrics(tmp_path: Path) -> None:
    inputs = tmp_path / "experiments"
    for experiment_id in ("E1", "E2", "E3", "E4"):
        _write_metrics(inputs, experiment_id)

    legacy_path = inputs / "E1" / "metrics.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy["macro_f1"] = 0.01
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    nested_path = inputs / "E1" / "evaluation" / "test" / "metrics.json"
    nested_path.parent.mkdir(parents=True)
    nested = _metrics("E1")
    nested.update({"split": "test", "macro_f1": 0.99})
    nested_path.write_text(json.dumps(nested), encoding="utf-8")

    rows = collect_experiment_metrics(inputs)

    e1 = next(row for row in rows if row["experiment_id"] == "E1")
    assert e1["macro_f1"] == 0.99


def test_comparison_never_uses_validation_metrics(tmp_path: Path) -> None:
    inputs = tmp_path / "experiments"
    for experiment_id in ("E1", "E2", "E3", "E4"):
        _write_metrics(inputs, experiment_id)

    legacy_path = inputs / "E1" / "metrics.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy["split"] = "validation"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    validation_path = inputs / "E1" / "evaluation" / "validation" / "metrics.json"
    validation_path.parent.mkdir(parents=True)
    validation = _metrics("E1")
    validation["split"] = "validation"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    with pytest.raises(ValueError, match="is not Test metrics"):
        collect_experiment_metrics(inputs)


def test_comparison_does_not_scan_validation_directory(tmp_path: Path) -> None:
    inputs = tmp_path / "experiments"
    for experiment_id in ("E1", "E2", "E3", "E4"):
        _write_metrics(inputs, experiment_id)

    (inputs / "E1" / "metrics.json").unlink()
    validation_path = inputs / "E1" / "evaluation" / "validation" / "metrics.json"
    validation_path.parent.mkdir(parents=True)
    validation = _metrics("E1")
    validation["split"] = "validation"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Missing experiments: \['E1'\]"):
        collect_experiment_metrics(inputs)
