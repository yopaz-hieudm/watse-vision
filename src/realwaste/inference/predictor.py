from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from realwaste.data.transforms import build_transforms
from realwaste.models import create_model
from realwaste.seed import resolve_device


class Predictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.class_names = checkpoint["class_names"]
        self.model = create_model(
            checkpoint["model_name"],
            len(self.class_names),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        data = self.config["data"]
        self.transform = build_transforms(
            int(data.get("image_size", 224)),
            False,
            resize_size=int(data.get("resize_size", 256)),
        )

    def prepare_image(self, image_path: str | Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_path: str | Path, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with torch.no_grad():
            probabilities = torch.softmax(self.model(self.prepare_image(image_path)), dim=1)[0]
        values, indices = probabilities.topk(min(top_k, len(self.class_names)))
        return [
            {
                "label": int(index),
                "class_name": self.class_names[int(index)],
                "probability": float(value),
            }
            for value, index in zip(values.cpu(), indices.cpu(), strict=True)
        ]
