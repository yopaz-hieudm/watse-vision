from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from realwaste.data.download import (
    DatasetIntegrityError,
    download_and_extract,
    safe_extract_zip,
    sha256_file,
    verify_checksum,
)


def test_checksum_verification(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    archive.write_bytes(b"trusted bytes")
    checksum = tmp_path / "sample.sha256"
    checksum.write_text(f"{sha256_file(archive)}  sample.zip\n")
    assert verify_checksum(archive, checksum) == sha256_file(archive)


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    archive.write_bytes(b"wrong bytes")
    checksum = tmp_path / "sample.sha256"
    checksum.write_text(f"{'0' * 64}  sample.zip\n")
    with pytest.raises(DatasetIntegrityError, match="mismatch"):
        verify_checksum(archive, checksum)


def test_zip_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "unsafe")
    with pytest.raises(DatasetIntegrityError, match="Unsafe ZIP path"):
        safe_extract_zip(archive, tmp_path / "extract")


def test_verify_only_does_not_extract(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("file.txt", "content")
    checksum = tmp_path / "sample.sha256"
    checksum.write_text(f"{sha256_file(archive)}  sample.zip\n")
    dataset_root = tmp_path / "missing_dataset"

    result = download_and_extract(archive, checksum, dataset_root, verify_only=True)

    assert result == archive
    assert not dataset_root.exists()
