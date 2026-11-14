"""Фиксированное разбиение датасета на train / val / test.

Разбиение сохраняется в configs/split_bbbc041seg.json и коммитится в git,
чтобы на Маке и на ПК использовались одни и те же снимки.

Запуск из корня репозитория:
    python scripts/make_split.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import list_pairs

DATA_ROOT = ROOT / "data" / "raw" / "bbbc041seg"
SPLIT_PATH = ROOT / "configs" / "split_bbbc041seg.json"

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def main():
    if SPLIT_PATH.exists():
        print("Разбиение уже существует:", SPLIT_PATH)
        print("Чтобы пересоздать, удали файл вручную.")
        return

    stems = []
    for stem, _, _ in list_pairs(DATA_ROOT):
        stems.append(stem)

    # Перемешиваем со своим генератором и фиксированным seed:
    # при любом запуске получится одинаковый порядок
    rng = random.Random(SEED)
    rng.shuffle(stems)

    n_total = len(stems)
    n_test = round(n_total * TEST_FRACTION)
    n_val = round(n_total * VAL_FRACTION)

    split = {
        "seed": SEED,
        "test": sorted(stems[:n_test]),
        "val": sorted(stems[n_test:n_test + n_val]),
        "train": sorted(stems[n_test + n_val:]),
    }

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    print("Всего снимков:", n_total)
    print("train:", len(split["train"]), "| val:", len(split["val"]), "| test:", len(split["test"]))
    print("Сохранено:", SPLIT_PATH)


if __name__ == "__main__":
    main()
