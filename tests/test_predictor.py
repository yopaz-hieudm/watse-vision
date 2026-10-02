from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from realwaste.constants import CLASS_NAMES
from realwaste.inference import Predictor
from realwaste.models import create_model


def test_predictor_returns_top_k(tmp_path: Path) -> None:
    model = create_model("mobilenet_v3_small", 9, pretrained=False)
    config = {
        "data": {"image_size": 64, "resize_size": 72},
        "model": {"name": "mobilenet_v3_small"},
    }
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": config,
            "class_names": list(CLASS_NAMES),
            "model_name": "mobilenet_v3_small",
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 80), color=(120, 80, 40)).save(image_path)
    predictions = Predictor(checkpoint, "cpu").predict(image_path, top_k=3)
    assert len(predictions) == 3
    assert all(0 <= row["probability"] <= 1 for row in predictions)
