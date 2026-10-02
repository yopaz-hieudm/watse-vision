from __future__ import annotations

from torch import nn

from realwaste.models.mobilenet_v3 import build_mobilenet_v3_small
from realwaste.models.resnet18 import build_resnet18


def create_model(
    name: str,
    num_classes: int = 9,
    *,
    pretrained: bool = True,
) -> nn.Module:
    if name == "mobilenet_v3_small":
        return build_mobilenet_v3_small(num_classes, pretrained=pretrained)
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model: {name}")


def _freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _unfreeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def set_backbone_trainable(model: nn.Module, mode: str) -> nn.Module:
    """Configure trainable layers for head-only, last-block or full fine-tuning."""
    model_name = getattr(model, "realwaste_model_name", "")
    mode = mode.lower()
    if mode not in {"head", "last_block", "all"}:
        raise ValueError("mode must be one of: head, last_block, all")
    if mode == "all":
        _unfreeze(model)
        return model

    _freeze_all(model)
    if model_name == "mobilenet_v3_small":
        _unfreeze(model.classifier)
        if mode == "last_block":
            _unfreeze(model.features[-1])
    elif model_name == "resnet18":
        _unfreeze(model.fc)
        if mode == "last_block":
            _unfreeze(model.layer4)
    else:
        raise ValueError("model was not created by realwaste.models.create_model")
    return model
