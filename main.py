import kagglehub
import os
from pathlib import Path

import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset

from sudoku_recog.utils import move_folder, check_data_exists
from sudoku_recog.constants import DATA_DIR


def main():
    # check if files have already been downloaded
    if check_data_exists(Path(DATA_DIR)):
        print("Data has already been downloaded")
    else:
        # Download latest version
        path = kagglehub.dataset_download("mexwell/sudoku-image-dataset")

        print("Path to cached dataset files:", path)

        move_folder(
            Path(path),
            Path(DATA_DIR)
        )

        print("Hello from sudoku-recog!")

    # Load Data
    test_dataset = PairedDatImageDataset(data_dir="data")
    print(f"{len(test_dataset)}")
    

if __name__ == "__main__":
    main()
