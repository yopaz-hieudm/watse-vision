from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


class ArtifactError(RuntimeError):
    """Base exception for an artifact that cannot be presented in the UI."""


class UnsafeArtifactPathError(ArtifactError, ValueError):
    """Raised when an artifact path escapes the project directory."""


class ArtifactNotFoundError(ArtifactError, FileNotFoundError):
    """Raised when an expected artifact does not exist."""


class ArtifactFormatError(ArtifactError, ValueError):
    """Raised when an artifact exists but has invalid contents."""


def resolve_within_root(project_root: str | Path, path: str | Path) -> Path:
    """Resolve ``path`` and reject traversal or symlinks outside ``project_root``."""

    root = Path(project_root).expanduser().resolve()
    supplied = Path(path).expanduser()
    candidate = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not candidate.is_relative_to(root):
        raise UnsafeArtifactPathError(
            f"Đường dẫn nằm ngoài thư mục dự án nên không thể đọc: {path}"
        )
    return candidate


class ArtifactReader:
    """Read UI artifacts with consistent path validation and friendly errors."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"Thư mục dự án không tồn tại: {self.project_root}")

    def resolve(self, path: str | Path) -> Path:
        return resolve_within_root(self.project_root, path)

    def exists(self, path: str | Path) -> bool:
        return self.resolve(path).is_file()

    def require_file(self, path: str | Path) -> Path:
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise ArtifactNotFoundError(f"Chưa có tệp kết quả: {self._display_path(resolved)}")
        return resolved

    def read_json(self, path: str | Path) -> Any:
        resolved = self.require_file(path)
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ArtifactFormatError(
                f"Tệp JSON không hợp lệ: {self._display_path(resolved)}"
            ) from error
        except OSError as error:
            raise ArtifactError(
                f"Không thể đọc tệp JSON {self._display_path(resolved)}: {error}"
            ) from error

    def read_json_object(self, path: str | Path) -> dict[str, Any]:
        value = self.read_json(path)
        if not isinstance(value, Mapping):
            resolved = self.resolve(path)
            raise ArtifactFormatError(
                f"Tệp JSON phải chứa một object: {self._display_path(resolved)}"
            )
        return dict(value)

    def read_optional_json(self, path: str | Path) -> Any | None:
        try:
            return self.read_json(path)
        except ArtifactNotFoundError:
            return None

    def read_optional_json_object(self, path: str | Path) -> dict[str, Any] | None:
        try:
            return self.read_json_object(path)
        except ArtifactNotFoundError:
            return None

    def read_csv(self, path: str | Path) -> list[dict[str, str]]:
        resolved = self.require_file(path)
        try:
            with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ArtifactFormatError(
                        f"Tệp CSV trống hoặc thiếu header: {self._display_path(resolved)}"
                    )
                return [dict(row) for row in reader]
        except ArtifactFormatError:
            raise
        except (csv.Error, UnicodeDecodeError) as error:
            raise ArtifactFormatError(
                f"Tệp CSV không hợp lệ: {self._display_path(resolved)}"
            ) from error
        except OSError as error:
            raise ArtifactError(
                f"Không thể đọc tệp CSV {self._display_path(resolved)}: {error}"
            ) from error

    def read_optional_csv(self, path: str | Path) -> list[dict[str, str]] | None:
        try:
            return self.read_csv(path)
        except ArtifactNotFoundError:
            return None

    def read_png(self, path: str | Path) -> Path:
        resolved = self.require_file(path)
        if resolved.suffix.lower() != ".png":
            raise ArtifactFormatError(
                f"Tệp ảnh phải có định dạng PNG: {self._display_path(resolved)}"
            )
        try:
            with Image.open(resolved) as image:
                image.verify()
                if image.format != "PNG":
                    raise ArtifactFormatError(
                        f"Nội dung tệp không phải ảnh PNG: {self._display_path(resolved)}"
                    )
        except ArtifactFormatError:
            raise
        except (UnidentifiedImageError, OSError) as error:
            raise ArtifactFormatError(
                f"Không thể mở ảnh PNG: {self._display_path(resolved)}"
            ) from error
        return resolved

    def find_png(self, path: str | Path) -> Path | None:
        try:
            return self.read_png(path)
        except ArtifactNotFoundError:
            return None

    def tail_text(
        self,
        path: str | Path,
        *,
        max_lines: int = 200,
        max_bytes: int = 256_000,
    ) -> str:
        if max_lines <= 0 or max_bytes <= 0:
            raise ValueError("max_lines and max_bytes must be positive")
        resolved = self.require_file(path)
        try:
            size = resolved.stat().st_size
            with resolved.open("rb") as handle:
                handle.seek(max(0, size - max_bytes))
                content = handle.read()
        except OSError as error:
            raise ArtifactError(
                f"Không thể đọc tệp log {self._display_path(resolved)}: {error}"
            ) from error
        lines = content.decode("utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)
