"""Скачивание датасета BBBC041Seg и краткая сводка по его содержимому.

Источник: https://github.com/Deponker/Blood-cell-segmentation-dataset
Сам архив лежит на Google Drive, поэтому качаем через gdown.

Запуск из корня репозитория:
    python scripts/download_bbbc041seg.py
"""
import zipfile
from collections import Counter
from pathlib import Path

import gdown
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "bbbc041seg.zip"
EXTRACT_DIR = RAW_DIR / "bbbc041seg"

GDRIVE_FILE_ID = "1nG-ra6BPAZSTsdYCvedzCo-JLD7jdH71"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def download_archive():
    """Скачивает архив, если его ещё нет на диске."""
    if ARCHIVE_PATH.exists():
        print("Архив уже скачан:", ARCHIVE_PATH)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Скачиваю архив с Google Drive (около 2 ГБ)...")
    gdown.download(id=GDRIVE_FILE_ID, output=str(ARCHIVE_PATH), quiet=False)


def extract_archive():
    """Распаковывает архив, если он ещё не распакован."""
    if EXTRACT_DIR.exists():
        print("Архив уже распакован:", EXTRACT_DIR)
        return

    if not zipfile.is_zipfile(ARCHIVE_PATH):
        # На случай если внутри не zip (например rar) — распаковать вручную
        raise RuntimeError(
            "Файл не является zip-архивом: " + str(ARCHIVE_PATH)
            + ". Распакуй его вручную в " + str(EXTRACT_DIR)
        )

    print("Распаковываю...")
    with zipfile.ZipFile(ARCHIVE_PATH) as zf:
        zf.extractall(EXTRACT_DIR)


def print_summary():
    """Печатает, какие папки и файлы внутри, и формат одной маски.

    Нужно, чтобы понять структуру датасета перед написанием загрузчика.
    """
    files = []
    for path in EXTRACT_DIR.rglob("*"):
        if path.is_file():
            files.append(path)

    print()
    print("Всего файлов:", len(files))

    # Сколько файлов каждого расширения лежит в каждой папке
    counter = Counter()
    for path in files:
        folder = path.parent.relative_to(EXTRACT_DIR)
        counter[(str(folder), path.suffix.lower())] += 1

    print("Папка | расширение | количество:")
    for (folder, ext), count in sorted(counter.items()):
        print("  ", folder, "|", ext, "|", count)

    # Примеры имён файлов из каждой папки
    print()
    print("Примеры имён:")
    shown = Counter()
    for path in sorted(files):
        folder = str(path.parent.relative_to(EXTRACT_DIR))
        if shown[folder] < 3:
            print("  ", path.relative_to(EXTRACT_DIR))
            shown[folder] += 1

    # Смотрим формат первой картинки в каждой папке
    print()
    print("Формат первой картинки в каждой папке:")
    checked = set()
    for path in sorted(files):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        folder = str(path.parent.relative_to(EXTRACT_DIR))
        if folder in checked:
            continue
        checked.add(folder)

        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            unique_values = np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)
        else:
            unique_values = np.unique(arr)
        print(
            "  ", folder, "| форма:", arr.shape, "| тип:", arr.dtype,
            "| уникальных значений:", len(unique_values),
        )


def main():
    download_archive()
    extract_archive()
    print_summary()


if __name__ == "__main__":
    main()
