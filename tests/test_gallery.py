from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.components.gallery import (
    create_thumbnail,
    discover_dataset_images,
    filter_gallery_images,
    paginate_gallery,
)
from realwaste.constants import CLASS_NAMES


def test_gallery_discovers_filters_and_paginates_images(tmp_path: Path) -> None:
    for class_index, class_name in enumerate(CLASS_NAMES[:2]):
        class_dir = tmp_path / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(3):
            Image.new("RGB", (20, 10), color=(class_index * 30, image_index * 20, 90)).save(
                class_dir / f"image_{image_index}.jpg"
            )
        (class_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

    images = discover_dataset_images(tmp_path)
    assert len(images) == 6
    assert [image.class_name for image in images[:3]] == [CLASS_NAMES[0]] * 3

    filtered = filter_gallery_images(images, CLASS_NAMES[1])
    assert len(filtered) == 3
    assert all(image.class_name == CLASS_NAMES[1] for image in filtered)

    page = paginate_gallery(images, page=2, page_size=4)
    assert page.page == 2
    assert page.total_pages == 2
    assert page.total_items == 6
    assert len(page.items) == 2


def test_gallery_rejects_unknown_class_and_invalid_page_size(tmp_path: Path) -> None:
    images = discover_dataset_images(tmp_path)
    with pytest.raises(ValueError, match="Unknown RealWaste class"):
        filter_gallery_images(images, "Unknown")
    with pytest.raises(ValueError, match="page_size must be positive"):
        paginate_gallery(images, page=1, page_size=0)


def test_thumbnail_has_bounded_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (800, 200), color=(12, 120, 40)).save(image_path)

    thumbnail = create_thumbnail(image_path, size=(160, 120))

    with Image.open(BytesIO(thumbnail)) as image:
        assert image.format == "JPEG"
        assert image.size == (160, 120)
