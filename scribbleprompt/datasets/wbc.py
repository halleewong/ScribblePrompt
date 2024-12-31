"""
Example data loader adapted from https://github.com/JJGO/UniverSeg/blob/main/example_data/wbc.py
"""
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = np.stack([seg == 0, seg == 128, seg == 255])
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("*.bmp")):
        img = process_img(file, size=size)
        seg_file = file.with_suffix(".png")
        seg = process_seg(seg_file, size=size)
        data.append((img / 255.0, seg))
    return data


def require_download_wbc():
    dest_folder = pathlib.Path("/tmp/universeg_wbc/")

    if not dest_folder.exists():
        repo_url = "https://github.com/zxaoyou/segmentation_WBC.git"
        subprocess.run(
            ["git", "clone", repo_url, str(dest_folder),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class WBC(Dataset):
    subdataset: Literal["JTSC", "CV"]
    split: Literal["train", "val", "test"]
    label: Optional[Literal["nucleus", "cytoplasm", "background"]] = None
    splits_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)

    def __post_init__(self):
        root = require_download_wbc()
        path = root / {"JTSC": "Dataset 1", "CV": "Dataset 2"}[self.subdataset]
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(path)]
        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "nucleus": 2, "background": 0}[self.label]
        self._idxs = self._split_indexes()

    def _split_indexes(self, seed: int = 42):
        N = len(self._data)

        train_size, val_size, test_size = self.splits_ratio
        trainval, test = train_test_split(
            range(N), test_size=test_size, random_state=42
        )
        val_ratio = val_size / (train_size + val_size)
        train, val = train_test_split(
            trainval, test_size=val_ratio, random_state=42
        )
        return {"train": train, "val": val, "test": test}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = seg[self._ilabel][None]
        return img, seg
    
    @property
    def attr(self):
        return {
            "dataset": "WBC",
            "subdataset": self.subdataset,
            "modality": "Microscopy",
            "axis": 0,
            "label": self.label,
            "split": self.split,
        }