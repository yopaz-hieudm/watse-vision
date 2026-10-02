from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CsvImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.frame = pd.read_csv(self.csv_path)
        required = {"path", "label", "class_name"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.root / str(row["path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label, str(row["path"])
