"""Сегментация снимков готовой моделью Cellpose (без дообучения на наших данных).

Cellpose ставится в ОТДЕЛЬНОЕ окружение (у него свои требования к версиям Python и PyTorch),
поэтому этот скрипт только сохраняет предсказания в файлы, а метрики считает наш evaluate.py:

    # в окружении Cellpose:
    python scripts/run_cellpose.py --split test
    # в основном окружении:
    python scripts/evaluate.py --method saved --pred-dir results/predictions/cellpose --split test

Предсказания: results/predictions/cellpose/<stem>.png — 16-битные PNG,
значение пикселя = номер клетки (0 — фон).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import list_pairs, load_image

DATA_ROOT = ROOT / "data" / "raw" / "bbbc041seg"
SPLIT_PATH = ROOT / "configs" / "split_bbbc041seg.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--diameter", type=float, default=0.0,
                        help="диаметр клетки в пикселях; 0 — пусть Cellpose определит сам")
    args = parser.parse_args()

    # Импорт здесь, а не наверху: в основном окружении проекта cellpose не установлен
    from cellpose import models

    with open(SPLIT_PATH) as f:
        stems = set(json.load(f)[args.split])
    pairs = []
    for stem, image_path, mask_path in list_pairs(DATA_ROOT):
        if stem in stems:
            pairs.append((stem, image_path, mask_path))
    if args.limit > 0:
        pairs = pairs[:args.limit]

    out_dir = ROOT / "results" / "predictions" / "cellpose"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Модель по умолчанию — Cellpose-SAM, обученная на очень разных клетках и снимках
    model = models.CellposeModel(gpu=True)

    diameter = None
    if args.diameter > 0:
        diameter = args.diameter

    start = time.time()
    for stem, image_path, _ in tqdm(pairs, desc="cellpose"):
        image = load_image(image_path)
        masks, _, _ = model.eval(image, diameter=diameter)
        Image.fromarray(masks.astype(np.uint16)).save(out_dir / (stem + ".png"))

    seconds = round((time.time() - start) / max(1, len(pairs)), 3)
    with open(out_dir / "_info.json", "w") as f:
        json.dump({"split": args.split, "images": len(pairs), "seconds_per_image": seconds}, f, indent=2)
    print("Готово:", len(pairs), "снимков,", seconds, "с на снимок. Предсказания в", out_dir)


if __name__ == "__main__":
    main()
