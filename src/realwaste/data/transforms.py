from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from realwaste.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(
    image_size: int = 224,
    augmentation: bool = False,
    *,
    resize_size: int = 256,
    random_resized_crop_scale: tuple[float, float] = (0.8, 1.0),
    horizontal_flip_probability: float = 0.5,
    rotation_degrees: float = 15.0,
    color_jitter: dict[str, float] | None = None,
):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if not augmentation:
        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    jitter = color_jitter or {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.05,
    }
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=random_resized_crop_scale,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
            transforms.RandomRotation(rotation_degrees),
            transforms.ColorJitter(**jitter),
            transforms.ToTensor(),
            normalize,
        ]
    )
