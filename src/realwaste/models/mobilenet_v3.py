from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet_v3_small(
    num_classes: int = 9,
    *,
    pretrained: bool = True,
) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Unexpected MobileNetV3 classifier layout")
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
    model.realwaste_model_name = "mobilenet_v3_small"
    model.realwaste_weights_id = "IMAGENET1K_V1" if pretrained else "none"
    return model
