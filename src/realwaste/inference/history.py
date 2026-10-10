from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from realwaste.constants import CLASS_NAMES
from realwaste.training.progress import atomic_write_json
from realwaste.ui.artifact_reader import resolve_within_root


def _pixel_digest(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def image_content_digest(content: bytes) -> str:
    """Compare decoded pixels without re-encoding every held-out image."""
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Ảnh vượt quá giới hạn 20 MB")
    with Image.open(io.BytesIO(content)) as source:
        return _pixel_digest(ImageOps.exif_transpose(source).convert("RGB"))


def normalize_image(content: bytes) -> tuple[bytes, str]:
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Ảnh vượt quá giới hạn 20 MB")
    with Image.open(io.BytesIO(content)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = _pixel_digest(image)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return buffer.getvalue(), digest


class PredictionHistory:
    """Persist predictions and explicitly reviewed labels in a local SQLite database."""

    def __init__(self, project_root: str | Path, *, storage_path: str | Path | None = None):
        self.root = Path(project_root).resolve()
        self.path = resolve_within_root(
            self.root, storage_path or "outputs/predictions/history.sqlite3"
        )

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    experiment_id TEXT NOT NULL, checkpoint TEXT NOT NULL,
                    checkpoint_sha256 TEXT NOT NULL, filename TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL, predictions_json TEXT NOT NULL,
                    elapsed_ms REAL NOT NULL, image_png BLOB,
                    reviewed_label TEXT, reviewed_at TEXT,
                    UNIQUE(experiment_id, checkpoint_sha256, image_sha256)
                )
            """)
            yield connection

    def save(
        self,
        *,
        experiment_id: str,
        checkpoint: Path,
        filename: str,
        image: bytes,
        predictions: list[dict],
        elapsed_ms: float,
        retain_image: bool,
    ) -> str:
        checkpoint = resolve_within_root(self.root, checkpoint)
        png, image_digest = normalize_image(image)
        with checkpoint.open("rb") as handle:
            checkpoint_digest = hashlib.file_digest(handle, "sha256").hexdigest()
        identifier = uuid4().hex
        timestamp = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO predictions
                   (id, created_at, experiment_id, checkpoint, checkpoint_sha256, filename,
                    image_sha256, predictions_json, elapsed_ms, image_png)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(experiment_id, checkpoint_sha256, image_sha256)
                   DO UPDATE SET image_png = COALESCE(predictions.image_png, excluded.image_png),
                                 predictions_json = excluded.predictions_json,
                                 elapsed_ms = excluded.elapsed_ms
                """,
                (
                    identifier,
                    timestamp,
                    experiment_id,
                    str(checkpoint.relative_to(self.root)),
                    checkpoint_digest,
                    Path(filename).name,
                    image_digest,
                    json.dumps(predictions, ensure_ascii=False, allow_nan=False),
                    elapsed_ms,
                    png if retain_image else None,
                ),
            )
            row = connection.execute(
                "SELECT id FROM predictions WHERE experiment_id=? AND checkpoint_sha256=? "
                "AND image_sha256=?",
                (experiment_id, checkpoint_digest, image_digest),
            ).fetchone()
            return str(row["id"])

    def list(self, experiment_id: str | None = None, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        query = (
            "SELECT id, created_at, experiment_id, checkpoint, checkpoint_sha256, filename, "
            "image_sha256, predictions_json, elapsed_ms, reviewed_label, reviewed_at, "
            "image_png IS NOT NULL AS image_retained FROM predictions"
        )
        arguments: list = []
        if experiment_id is not None:
            query += " WHERE experiment_id=?"
            arguments.append(experiment_id)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        arguments.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        result = dict(row)
        result.pop("image_png", None)
        result["predictions"] = json.loads(result.pop("predictions_json"))
        return result

    def get_image(self, identifier: str) -> bytes | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT image_png FROM predictions WHERE id=?", (identifier,)
            ).fetchone()
        return row["image_png"] if row else None

    def review(self, identifier: str, label: str | None) -> None:
        if label is not None and label not in CLASS_NAMES:
            raise ValueError("Nhãn phản hồi không thuộc 9 lớp RealWaste")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT image_png FROM predictions WHERE id=?", (identifier,)
            ).fetchone()
            if row is None:
                raise ValueError("Không tìm thấy dự đoán")
            if label is not None and row["image_png"] is None:
                raise ValueError("Cần cho phép lưu ảnh trước khi dùng phản hồi để huấn luyện")
            connection.execute(
                "UPDATE predictions SET reviewed_label=?, reviewed_at=? WHERE id=?",
                (label, datetime.now(UTC).isoformat() if label else None, identifier),
            )

    def reviewed_count(self) -> int:
        if not self.path.exists():
            return 0
        with self._connection() as connection:
            return connection.execute(
                "SELECT COUNT(DISTINCT image_sha256) FROM predictions "
                "WHERE reviewed_label IS NOT NULL"
            ).fetchone()[0]

    def create_training_snapshot(self, *, split_directory: str | Path | None = None) -> Path:
        """Freeze reviewed images; reject contradictory labels and held-out duplicates."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions WHERE reviewed_label IS NOT NULL ORDER BY id"
            ).fetchall()
        unique: dict[str, sqlite3.Row] = {}
        for row in rows:
            digest = row["image_sha256"]
            if digest in unique and unique[digest]["reviewed_label"] != row["reviewed_label"]:
                raise ValueError("Một ảnh có nhiều nhãn phản hồi khác nhau; hãy sửa lịch sử trước")
            unique[digest] = row
        if not unique:
            raise ValueError("Chưa có ảnh được xác nhận nhãn")
        held_out: set[str] = set()
        split_root = resolve_within_root(self.root, split_directory or "data/splits")
        for name in ("val", "test"):
            split = split_root / f"{name}.csv"
            with split.open(encoding="utf-8", newline="") as handle:
                for item in csv.DictReader(handle):
                    path = resolve_within_root(self.root / "data/raw/RealWaste", item["path"])
                    held_out.add(image_content_digest(path.read_bytes()))
        if held_out.intersection(unique):
            raise ValueError("Phản hồi chứa ảnh của Validation/Test; hãy bỏ xác nhận các ảnh đó")
        snapshot = resolve_within_root(self.root, f"outputs/feedback/snapshots/{uuid4().hex}")
        snapshot.mkdir(parents=True)
        manifest_rows = []
        for image_index, (digest, row) in enumerate(sorted(unique.items()), start=1):
            png = row["image_png"]
            if png is None or image_content_digest(png) != digest:
                raise ValueError("Ảnh phản hồi bị thiếu hoặc không khớp nội dung đã lưu")
            filename = f"{image_index:06d}.png"
            (snapshot / filename).write_bytes(png)
            label = row["reviewed_label"]
            manifest_rows.append(
                {
                    "path": filename,
                    "label": CLASS_NAMES.index(label),
                    "class_name": label,
                    "sha256": hashlib.sha256(png).hexdigest(),
                }
            )
        manifest = snapshot / "train.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "label", "class_name", "sha256"))
            writer.writeheader()
            writer.writerows(manifest_rows)
        atomic_write_json(
            snapshot / "metadata.json",
            {
                "fingerprint": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "count": len(manifest_rows),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return manifest


def validate_feedback_snapshot(manifest: str | Path, project_root: str | Path) -> dict:
    path = resolve_within_root(Path(project_root), manifest)
    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["fingerprint"]:
        raise ValueError("Manifest phản hồi đã thay đổi")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != metadata["count"] or not rows:
        raise ValueError("Số ảnh phản hồi không khớp manifest")
    for row in rows:
        image_path = resolve_within_root(path.parent, row["path"])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Ảnh phản hồi đã thay đổi")
        if row["class_name"] not in CLASS_NAMES or CLASS_NAMES.index(row["class_name"]) != int(
            row["label"]
        ):
            raise ValueError("Nhãn phản hồi không hợp lệ")
    return {**metadata, "manifest": str(path)}
