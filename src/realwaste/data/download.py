from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from realwaste.constants import (
    CLASS_NAMES,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_IMAGE_COUNT,
    IMAGE_EXTENSIONS,
    REALWASTE_URL,
)


class DatasetIntegrityError(RuntimeError):
    """Raised when the archive or extracted dataset does not match expectations."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_expected_checksum(path: str | Path) -> str:
    checksum_path = Path(path)
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Checksum file not found: {checksum_path}. "
            "Create it from a trusted UCI download before normal use."
        )
    checksum = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise DatasetIntegrityError(f"Invalid SHA-256 value in {checksum_path}")
    return checksum


def verify_checksum(archive_path: str | Path, checksum_path: str | Path) -> str:
    expected = read_expected_checksum(checksum_path)
    actual = sha256_file(archive_path)
    if actual != expected:
        raise DatasetIntegrityError(
            f"SHA-256 mismatch for {archive_path}: expected {expected}, got {actual}"
        )
    return actual


def download_archive(
    url: str,
    archive_path: str | Path,
    *,
    timeout_seconds: int = 60,
) -> Path:
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    existing_size = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(10, timeout_seconds),
    ) as response:
        response.raise_for_status()
        resumed = existing_size > 0 and response.status_code == 206
        if existing_size and not resumed:
            partial_path.unlink(missing_ok=True)
            existing_size = 0
        mode = "ab" if resumed else "wb"
        content_length = int(response.headers.get("content-length", 0))
        total = existing_size + content_length if content_length else None
        with (
            partial_path.open(mode) as handle,
            tqdm(
                total=total,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                desc="Downloading RealWaste",
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))

    os.replace(partial_path, archive_path)
    return archive_path


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = info.external_attr >> 16
    return stat.S_ISLNK(unix_mode)


def safe_extract_zip(archive_path: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if _is_zip_symlink(member):
                raise DatasetIntegrityError(f"Refusing symbolic link in ZIP: {member.filename}")
            target = (destination / member.filename).resolve()
            if target != resolved_destination and resolved_destination not in target.parents:
                raise DatasetIntegrityError(f"Unsafe ZIP path: {member.filename}")
        archive.extractall(destination)
    return destination


def _class_image_count(class_dir: Path) -> int:
    return sum(
        1
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset_structure(
    dataset_root: str | Path,
    *,
    strict_counts: bool = True,
) -> dict[str, int]:
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise DatasetIntegrityError(f"Dataset directory not found: {dataset_root}")
    missing = [name for name in CLASS_NAMES if not (dataset_root / name).is_dir()]
    if missing:
        raise DatasetIntegrityError(f"Missing class directories: {', '.join(missing)}")

    counts = {name: _class_image_count(dataset_root / name) for name in CLASS_NAMES}
    if strict_counts and counts != EXPECTED_CLASS_COUNTS:
        raise DatasetIntegrityError(
            f"Unexpected class counts. Expected {EXPECTED_CLASS_COUNTS}, got {counts}"
        )
    if strict_counts and sum(counts.values()) != EXPECTED_IMAGE_COUNT:
        raise DatasetIntegrityError(
            f"Expected {EXPECTED_IMAGE_COUNT} images, got {sum(counts.values())}"
        )
    return counts


def find_extracted_dataset_root(search_root: str | Path) -> Path:
    search_root = Path(search_root)
    candidates = [search_root] + [path for path in search_root.rglob("RealWaste") if path.is_dir()]
    for candidate in candidates:
        if all((candidate / class_name).is_dir() for class_name in CLASS_NAMES):
            return candidate
    raise DatasetIntegrityError("Could not locate a RealWaste directory with all 9 classes")


def extract_dataset(
    archive_path: str | Path,
    dataset_root: str | Path,
    *,
    force: bool = False,
) -> Path:
    dataset_root = Path(dataset_root)
    if dataset_root.exists():
        if not force:
            validate_dataset_structure(dataset_root)
            return dataset_root
        shutil.rmtree(dataset_root)

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="realwaste_extract_", dir=dataset_root.parent) as tmp:
        temp_root = Path(tmp)
        safe_extract_zip(archive_path, temp_root)
        source_root = find_extracted_dataset_root(temp_root)
        validate_dataset_structure(source_root)
        shutil.move(str(source_root), str(dataset_root))
    validate_dataset_structure(dataset_root)
    return dataset_root


def download_and_extract(
    archive_path: str | Path,
    checksum_path: str | Path,
    dataset_root: str | Path,
    *,
    url: str = REALWASTE_URL,
    verify_only: bool = False,
    force_extract: bool = False,
) -> Path:
    archive_path = Path(archive_path)
    dataset_root = Path(dataset_root)
    dataset_is_valid = False
    if dataset_root.exists():
        validate_dataset_structure(dataset_root)
        dataset_is_valid = True

    # A strictly validated extracted dataset is sufficient when the local archive
    # was intentionally removed to save disk space.
    if dataset_is_valid and not archive_path.exists() and not force_extract:
        return dataset_root
    if not archive_path.exists():
        if verify_only:
            raise FileNotFoundError(f"Archive not found for verification: {archive_path}")
        download_archive(url, archive_path)
    verify_checksum(archive_path, checksum_path)
    if verify_only:
        if dataset_is_valid:
            return dataset_root
        return archive_path
    if dataset_is_valid and not force_extract:
        return dataset_root
    return extract_dataset(archive_path, dataset_root, force=force_extract)
