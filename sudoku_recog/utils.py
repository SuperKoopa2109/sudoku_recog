from pathlib import Path
import shutil

def copy_folder(src: Path, dst: Path):
    shutil.copytree(src, dst, dirs_exist_ok=True)

def move_folder(src: Path, dst: Path):
    shutil.move(src, dst)

def check_data_exists(folder: Path) -> bool:
    has_files = False
    if folder.exists() and folder.is_dir():
        has_files = any(p.is_file() for p in folder.iterdir())

    return has_files