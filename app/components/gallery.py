from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import ceil
from pathlib import Path

from PIL import Image, ImageOps

from realwaste.constants import CLASS_NAMES, IMAGE_EXTENSIONS


@dataclass(frozen=True, slots=True)
class GalleryImage:
    path: Path
    class_name: str


@dataclass(frozen=True, slots=True)
class GalleryPage:
    items: tuple[GalleryImage, ...]
    page: int
    page_size: int
    total_items: int
    total_pages: int


def dataset_signature(dataset_root: str | Path) -> tuple[tuple[str, int], ...]:
    """Return a cheap cache key that changes when a class directory changes."""

    root = Path(dataset_root)
    signature: list[tuple[str, int]] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        modified_ns = class_dir.stat().st_mtime_ns if class_dir.is_dir() else 0
        signature.append((class_name, modified_ns))
    return tuple(signature)


def discover_dataset_images(dataset_root: str | Path) -> tuple[GalleryImage, ...]:
    """Discover every supported image in the nine canonical RealWaste classes."""

    root = Path(dataset_root)
    images: list[GalleryImage] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        paths = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        images.extend(GalleryImage(path=path, class_name=class_name) for path in paths)
    return tuple(images)


def filter_gallery_images(
    images: tuple[GalleryImage, ...], class_name: str | None
) -> tuple[GalleryImage, ...]:
    if class_name is None:
        return images
    if class_name not in CLASS_NAMES:
        raise ValueError(f"Unknown RealWaste class: {class_name}")
    return tuple(image for image in images if image.class_name == class_name)


def paginate_gallery(images: tuple[GalleryImage, ...], *, page: int, page_size: int) -> GalleryPage:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    total_pages = max(1, ceil(len(images) / page_size))
    current_page = min(max(page, 1), total_pages)
    start = (current_page - 1) * page_size
    return GalleryPage(
        items=images[start : start + page_size],
        page=current_page,
        page_size=page_size,
        total_items=len(images),
        total_pages=total_pages,
    )


def create_thumbnail(
    image_path: str | Path,
    *,
    size: tuple[int, int] = (360, 270),
) -> bytes:
    """Create a small, consistently sized JPEG thumbnail for browser rendering."""

    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("thumbnail dimensions must be positive")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, color="#F2F6F3")
        offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        canvas.paste(image, offset)

    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()
