from __future__ import annotations

from pathlib import Path

from PIL import Image

from realwaste.constants import CLASS_NAMES
from realwaste.data.inspection import write_eda_outputs


def test_eda_creates_required_outputs(tmp_path: Path) -> None:
    root = tmp_path / "RealWaste"
    for index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        Image.new("RGB", (16, 16), color=(index * 20, 40, 100)).save(class_dir / f"{index}.jpg")
    output = tmp_path / "eda"
    summary = write_eda_outputs(root, output, strict=False)
    assert summary["valid_images"] == 9
    for name in (
        "class_distribution.csv",
        "class_distribution.png",
        "sample_grid.png",
        "invalid_images.csv",
        "dataset_summary.json",
    ):
        assert (output / name).is_file()
    assert (output / "invalid_images.csv").read_text().startswith("path,class_name,error")
