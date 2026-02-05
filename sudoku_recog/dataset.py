from pathlib import Path

import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset

from sudoku_recog.constants import DATA_DIR


class PairedDatImageDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, transform=None, dat_dtype=np.float32):
        """
        Args:
            data_dir (str or Path): directory containing .dat and .jpg files
            transform (callable, optional): image transform
            dat_dtype: numpy dtype for .dat files
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.dat_dtype = dat_dtype

        self.samples = self._index_files()

    def _index_files(self):
        grouped = defaultdict(dict)

        for file in self.data_dir.iterdir():
            if not file.is_file():
                continue

            if file.suffix not in {".dat", ".jpg"}:
                continue

            grouped[file.stem][file.suffix] = file

        samples = []
        for key, files in grouped.items():
            if ".dat" in files and ".jpg" in files:
                samples.append({
                    "key": key,
                    "dat": files[".dat"],
                    "img": files[".jpg"],
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # load .dat
        dat = np.fromfile(sample["dat"], dtype=self.dat_dtype)
        dat = torch.from_numpy(dat)

        # load image
        img = Image.open(sample["img"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return dat, img
    

    def read_sudoku_dat(path):
        """
        Parse a .dat file with:
        line 1: phone model
        line 2: resolution / image info
        lines 3-11: 9x9 sudoku grid
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 11:
            raise ValueError(f"Invalid .dat file (too few lines): {path}")

        phone_model = lines[0]
        image_info = lines[1]

        grid = []
        for i, line in enumerate(lines[2:11], start=3):
            row = list(map(int, line.split()))
            if len(row) != 9:
                raise ValueError(
                    f"Invalid sudoku row length at line {i} in {path}"
                )
            grid.append(row)

        return phone_model, image_info, grid
