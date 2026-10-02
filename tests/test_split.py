from __future__ import annotations

from pathlib import Path

import pandas as pd

import realwaste.data.split as split_module
from realwaste.constants import CLASS_NAMES


def test_stratified_split_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    records = [
        {
            "path": f"{class_name}/{index}.jpg",
            "label": label,
            "class_name": class_name,
            "width": 16,
            "height": 16,
        }
        for label, class_name in enumerate(CLASS_NAMES)
        for index in range(20)
    ]
    monkeypatch.setattr(split_module, "inspect_dataset", lambda _root: (records, []))
    first = split_module.create_stratified_splits(tmp_path, tmp_path / "first")
    split_module.create_stratified_splits(tmp_path, tmp_path / "second")
    assert first["counts"] == {"train": 126, "validation": 27, "test": 27}
    for filename in ("train.csv", "val.csv", "test.csv"):
        assert pd.read_csv(tmp_path / "first" / filename).equals(
            pd.read_csv(tmp_path / "second" / filename)
        )
    train = set(pd.read_csv(tmp_path / "first/train.csv")["path"])
    validation = set(pd.read_csv(tmp_path / "first/val.csv")["path"])
    test = set(pd.read_csv(tmp_path / "first/test.csv")["path"])
    assert not train & validation
    assert not train & test
    assert not validation & test
