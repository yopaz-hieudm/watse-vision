from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from realwaste.ui.artifact_reader import (
    ArtifactFormatError,
    ArtifactNotFoundError,
    ArtifactReader,
    UnsafeArtifactPathError,
)


def test_reads_json_object_and_optional_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "summary.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"valid_images": 12}), encoding="utf-8")
    reader = ArtifactReader(tmp_path)

    assert reader.read_json_object("outputs/summary.json") == {"valid_images": 12}
    assert reader.read_optional_json("outputs/missing.json") is None
    with pytest.raises(ArtifactNotFoundError, match="Chưa có tệp kết quả"):
        reader.read_json("outputs/missing.json")


@pytest.mark.parametrize(
    ("contents", "message"),
    [("{broken", "JSON không hợp lệ"), ("[]", "phải chứa một object")],
)
def test_reports_invalid_json_with_friendly_error(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ArtifactFormatError, match=message):
        ArtifactReader(tmp_path).read_json_object(path)


def test_reads_csv_and_accepts_header_only_csv(tmp_path: Path) -> None:
    table = tmp_path / "table.csv"
    table.write_text("name,count\nPlastic,10\nPaper,5\n", encoding="utf-8")
    empty_result = tmp_path / "invalid_images.csv"
    empty_result.write_text("path,error\n", encoding="utf-8")
    reader = ArtifactReader(tmp_path)

    assert reader.read_csv(table) == [
        {"name": "Plastic", "count": "10"},
        {"name": "Paper", "count": "5"},
    ]
    assert reader.read_csv(empty_result) == []


def test_rejects_completely_empty_csv(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ArtifactFormatError, match="trống hoặc thiếu header"):
        ArtifactReader(tmp_path).read_csv(path)


def test_png_is_verified_before_returning_path(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (8, 6), "green").save(image_path)
    reader = ArtifactReader(tmp_path)

    assert reader.read_png("chart.png") == image_path.resolve()
    image_path.write_bytes(b"not an image")
    with pytest.raises(ArtifactFormatError, match="Không thể mở ảnh PNG"):
        reader.read_png(image_path)


def test_reader_rejects_path_traversal_and_outside_symlink(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")
    (project / "link.json").symlink_to(outside)
    reader = ArtifactReader(project)

    with pytest.raises(UnsafeArtifactPathError, match="ngoài thư mục dự án"):
        reader.read_json("../secret.json")
    with pytest.raises(UnsafeArtifactPathError, match="ngoài thư mục dự án"):
        reader.read_json("link.json")


def test_tail_text_is_bounded_by_line_count(tmp_path: Path) -> None:
    log = tmp_path / "stdout.log"
    log.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    assert ArtifactReader(tmp_path).tail_text(log, max_lines=2) == "three\nfour"
