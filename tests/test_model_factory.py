from __future__ import annotations

import pytest
import torch

from realwaste.models import create_model, set_backbone_trainable


@pytest.mark.parametrize("name", ["mobilenet_v3_small", "resnet18"])
def test_model_output_shape(name: str) -> None:
    model = create_model(name, 9, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 9)


@pytest.mark.parametrize("name", ["mobilenet_v3_small", "resnet18"])
def test_head_mode_freezes_backbone(name: str) -> None:
    model = create_model(name, 9, pretrained=False)
    set_backbone_trainable(model, "head")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    assert trainable
    assert frozen
