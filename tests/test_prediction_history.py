from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from realwaste.data.dataloader import build_dataloaders
from realwaste.inference.history import PredictionHistory, validate_feedback_snapshot


def _image(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(buffer, "PNG")
    return buffer.getvalue()


def _save(root: Path, *, retain: bool = True, experiment: str = "E1", color: str = "red"):
    checkpoint = root / f"{experiment}.pt"
    checkpoint.write_bytes(experiment.encode())
    return PredictionHistory(root).save(
        experiment_id=experiment,
        checkpoint=checkpoint,
        filename="upload.png",
        image=_image(color),
        predictions=[{"class_name": "Glass", "probability": 0.6}],
        elapsed_ms=10,
        retain_image=retain,
    )


def _splits(root: Path) -> dict:
    dataset = root / "data/raw/RealWaste"
    dataset.mkdir(parents=True)
    (dataset / "sample.png").write_bytes(_image("blue"))
    splits = root / "data/splits"
    splits.mkdir(parents=True)
    for name in ("train", "val", "test"):
        (splits / f"{name}.csv").write_text(
            "path,label,class_name\nsample.png,0,Cardboard\n", encoding="utf-8"
        )
    return {
        "seed": 42,
        "data": {
            "root": str(dataset),
            "image_size": 12,
            "resize_size": 12,
            "augmentation": {"enabled": False},
            "batch_size": 1,
            "num_workers": 0,
            "train_csv": str(splits / "train.csv"),
            "val_csv": str(splits / "val.csv"),
            "test_csv": str(splits / "test.csv"),
        },
    }


def test_history_survives_reload_and_upgrades_image_consent_without_duplicate(tmp_path):
    identifier = _save(tmp_path, retain=False)
    history = PredictionHistory(tmp_path)
    assert history.get_image(identifier) is None
    with pytest.raises(ValueError, match="Cần cho phép lưu ảnh"):
        history.review(identifier, "Glass")
    assert _save(tmp_path, retain=True) == identifier
    history.review(identifier, "Paper")
    record = PredictionHistory(tmp_path).list("E1")[0]
    assert record["reviewed_label"] == "Paper"
    assert len(history.list()) == 1
    assert history.get_image(identifier) is not None
    history.review(identifier, None)
    assert history.reviewed_count() == 0


def test_reviewed_snapshot_adds_only_training_images_and_is_immutable(tmp_path):
    config = _splits(tmp_path)
    history = PredictionHistory(tmp_path)
    identifier = _save(tmp_path)
    history.review(identifier, "Paper")
    manifest = history.create_training_snapshot()
    config["feedback"] = validate_feedback_snapshot(manifest, tmp_path)
    loaders = build_dataloaders(config)
    assert len(loaders["train"].dataset) == 2
    assert len(loaders["validation"].dataset) == len(loaders["test"].dataset) == 1
    assert loaders["train"].dataset[1][1].item() == 5
    history.review(identifier, "Plastic")
    assert pd.read_csv(manifest).iloc[0]["class_name"] == "Paper"
    image_path = manifest.parent / pd.read_csv(manifest).iloc[0]["path"]
    image_path.write_bytes(_image("green"))
    with pytest.raises(ValueError, match="Ảnh phản hồi đã thay đổi"):
        validate_feedback_snapshot(manifest, tmp_path)


def test_feedback_rejects_held_out_image_and_conflicting_labels(tmp_path):
    _splits(tmp_path)
    history = PredictionHistory(tmp_path)
    identifier = _save(tmp_path, color="blue")
    history.review(identifier, "Glass")
    with pytest.raises(ValueError, match="Validation/Test"):
        history.create_training_snapshot()
    history.review(identifier, None)
    first = _save(tmp_path, color="red")
    second = _save(tmp_path, color="red", experiment="E2")
    history.review(first, "Glass")
    history.review(second, "Plastic")
    with pytest.raises(ValueError, match="nhiều nhãn"):
        history.create_training_snapshot()


def test_changed_checkpoint_produces_distinct_history_version(tmp_path):
    first = _save(tmp_path)
    checkpoint = tmp_path / "E1.pt"
    checkpoint.write_bytes(b"new-model")
    second = PredictionHistory(tmp_path).save(
        experiment_id="E1",
        checkpoint=checkpoint,
        filename="upload.png",
        image=_image(),
        predictions=[{"class_name": "Paper", "probability": 0.9}],
        elapsed_ms=9,
        retain_image=False,
    )
    assert first != second
    assert len(PredictionHistory(tmp_path).list()) == 2
