from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from realwaste.data.transforms import build_transforms


def test_validation_transform_is_deterministic() -> None:
    image = Image.fromarray(np.full((300, 300, 3), 127, dtype=np.uint8))
    transform = build_transforms(224, False)
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)


def test_augmentation_returns_valid_tensor() -> None:
    image = Image.fromarray(np.full((300, 300, 3), 127, dtype=np.uint8))
    output = build_transforms(224, True)(image)
    assert output.shape == (3, 224, 224)
    assert torch.isfinite(output).all()
