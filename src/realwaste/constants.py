from __future__ import annotations

from pathlib import Path

CLASS_NAMES: tuple[str, ...] = (
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
)

EXPECTED_CLASS_COUNTS: dict[str, int] = {
    "Cardboard": 461,
    "Food Organics": 411,
    "Glass": 420,
    "Metal": 790,
    "Miscellaneous Trash": 495,
    "Paper": 500,
    "Plastic": 921,
    "Textile Trash": 318,
    "Vegetation": 436,
}

EXPECTED_IMAGE_COUNT = sum(EXPECTED_CLASS_COUNTS.values())
REALWASTE_DATASET_ID = 908
REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
DEFAULT_ARCHIVE = Path("data/downloads/realwaste.zip")
DEFAULT_CHECKSUM = Path("data/checksums/realwaste.sha256")
DEFAULT_DATASET_ROOT = Path("data/raw/RealWaste")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
