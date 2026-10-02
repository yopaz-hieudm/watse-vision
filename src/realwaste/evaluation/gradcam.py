from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


def default_target_layer(model: nn.Module) -> nn.Module:
    model_name = getattr(model, "realwaste_model_name", "")
    if model_name == "mobilenet_v3_small":
        return model.features[-1]
    if model_name == "resnet18":
        return model.layer4[-1]
    raise ValueError("Unsupported model for Grad-CAM")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or default_target_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError("Grad-CAM expects a tensor with shape [1, C, H, W]")
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        predicted = int(logits.argmax(dim=1).item())
        target = predicted if target_class is None else int(target_class)
        logits[0, target].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        return cam.cpu().numpy(), predicted, torch.softmax(logits.detach(), dim=1)[0]


def save_gradcam_figure(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    heatmap_resized = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(image)
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
