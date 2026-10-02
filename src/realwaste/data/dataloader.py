from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from realwaste.data.dataset import CsvImageDataset
from realwaste.data.transforms import build_transforms
from realwaste.seed import seed_worker


def build_dataloaders(config: dict[str, Any]):
    data = config["data"]
    aug = data["augmentation"]
    train_transform = build_transforms(
        int(data["image_size"]),
        bool(aug["enabled"]),
        resize_size=int(data.get("resize_size", 256)),
        random_resized_crop_scale=tuple(aug.get("random_resized_crop_scale", [0.8, 1.0])),
        horizontal_flip_probability=float(aug.get("horizontal_flip_probability", 0.5)),
        rotation_degrees=float(aug.get("rotation_degrees", 15)),
        color_jitter=aug.get("color_jitter"),
    )
    eval_transform = build_transforms(
        int(data["image_size"]),
        False,
        resize_size=int(data.get("resize_size", 256)),
    )
    datasets = {
        "train": CsvImageDataset(data["train_csv"], data["root"], train_transform),
        "validation": CsvImageDataset(data["val_csv"], data["root"], eval_transform),
        "test": CsvImageDataset(data["test_csv"], data["root"], eval_transform),
    }
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(data["batch_size"]),
        "num_workers": int(data["num_workers"]),
        "pin_memory": bool(data.get("pin_memory", False)) and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": int(data["num_workers"]) > 0,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
