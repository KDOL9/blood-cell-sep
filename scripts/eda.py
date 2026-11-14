"""Разведочный анализ (EDA) датасета BBBC041Seg.

Считает по каждому снимку:
  - сколько клеток (связных областей маски);
  - площади клеток;
  - какие клетки касаются соседей и образуют комки;
  - подозрительно большие области — вероятно, слитые клетки (разделительная линия порвалась).

Результаты: results/eda/ — summary.json, per_image.csv и картинки для отчёта.

Запуск из корня репозитория:
    python scripts/eda.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, без окон
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import (
    MAX_SOLIDITY,
    MERGE_FACTOR,
    border_labels,
    find_suspected_merges,
    list_pairs,
    load_image,
    load_mask,
    mask_to_instances,
)

DATA_ROOT = ROOT / "data" / "raw" / "bbbc041seg"
OUT_DIR = ROOT / "results" / "eda"

MIN_AREA = 200          # области меньше — шум, а не клетки
TOUCH_DISTANCE = 2      # клетки ближе этого числа пикселей считаем касающимися
N_EXAMPLES = 6          # сколько примеров сохранить картинками


def find_touching_pairs(labels):
    """Находит пары номеров клеток, которые касаются друг друга.

    Идея: в окне размером (2 * TOUCH_DISTANCE + 1) вокруг каждого пикселя
    берём максимальный и минимальный ненулевой номер клетки.
    Если они разные — в этом месте рядом две разные клетки.
    """
    size = 2 * TOUCH_DISTANCE + 1
    max_label = ndi.maximum_filter(labels, size=size)

    # Для минимума фон (0) заменяем очень большим числом, чтобы он не мешал
    big = np.iinfo(np.int32).max
    no_background = np.where(labels > 0, labels, big)
    min_label = ndi.minimum_filter(no_background, size=size)

    contact = (max_label > 0) & (min_label != big) & (max_label != min_label)

    pairs = set()
    for a, b in zip(min_label[contact], max_label[contact]):
        pairs.add((int(a), int(b)))
    return pairs


def group_into_clumps(count, pairs):
    """Объединяет касающиеся клетки в комки (система непересекающихся множеств).

    Возвращает список комков; комок — список номеров клеток (только размера 2+).
    """
    parent = list(range(count + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    groups = {}
    for cell in range(1, count + 1):
        root = find(cell)
        if root not in groups:
            groups[root] = []
        groups[root].append(cell)

    clumps = []
    for members in groups.values():
        if len(members) >= 2:
            clumps.append(members)
    return clumps


def save_example(stem, image, labels, highlight, title, path):
    """Сохраняет снимок рядом с маской, где выделенные клетки покрашены красным."""
    overlay = np.zeros(labels.shape + (3,), dtype=np.uint8)
    overlay[labels > 0] = (200, 200, 200)
    for cell in highlight:
        overlay[labels == cell] = (230, 60, 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].imshow(image)
    axes[0].set_title(stem[:8] + "…  снимок")
    axes[1].imshow(overlay)
    axes[1].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = list_pairs(DATA_ROOT)
    print("Пар снимок–маска:", len(pairs))

    rows = []
    all_areas = []
    per_image = {}

    # Проход 1: клетки, площади, комки
    for stem, image_path, mask_path in tqdm(pairs, desc="Анализ"):
        mask = load_mask(mask_path)
        labels, count = mask_to_instances(mask, min_area=MIN_AREA)

        areas = np.bincount(labels.ravel())[1:]
        all_areas.extend(areas.tolist())

        touching = find_touching_pairs(labels)
        clumps = group_into_clumps(count, touching)
        cells_in_clumps = 0
        for clump in clumps:
            cells_in_clumps += len(clump)

        border_cells = border_labels(labels)

        per_image[stem] = {
            "areas": areas,
            "clumps": clumps,
            "border": border_cells,
            "image_path": image_path,
            "mask_path": mask_path,
        }
        rows.append({
            "stem": stem,
            "height": mask.shape[0],
            "width": mask.shape[1],
            "cells": count,
            "clumps": len(clumps),
            "cells_in_clumps": cells_in_clumps,
            "largest_clump": max([len(c) for c in clumps], default=0),
            "border_cells": len(border_cells),
        })

    all_areas = np.array(all_areas)
    median_area = float(np.median(all_areas))

    # Проход 2: подозрительные области — вероятно, слитые клетки (см. src/data_io.py)
    suspected_total = 0
    for row in rows:
        info = per_image[row["stem"]]
        labels, _ = mask_to_instances(load_mask(info["mask_path"]), min_area=MIN_AREA)
        suspected = find_suspected_merges(labels)
        info["suspected"] = suspected
        row["suspected_merges"] = len(suspected)
        suspected_total += len(suspected)

    # --- Таблица по снимкам ---
    with open(OUT_DIR / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- Итоговая сводка ---
    cells = np.array([r["cells"] for r in rows])
    clumps = np.array([r["clumps"] for r in rows])
    in_clumps = np.array([r["cells_in_clumps"] for r in rows])
    sizes = set()
    for r in rows:
        sizes.add((r["height"], r["width"]))

    summary = {
        "images": len(rows),
        "image_sizes": sorted([list(s) for s in sizes]),
        "cells_total": int(cells.sum()),
        "cells_per_image_mean": round(float(cells.mean()), 1),
        "cells_per_image_min": int(cells.min()),
        "cells_per_image_max": int(cells.max()),
        "cell_area_median_px": round(median_area, 1),
        "clumps_total": int(clumps.sum()),
        "images_with_clumps": int((clumps > 0).sum()),
        "cells_in_clumps_share": round(float(in_clumps.sum() / cells.sum()), 3),
        "suspected_merged_regions": suspected_total,
        "images_with_suspected_merges": int(sum(1 for r in rows if r["suspected_merges"] > 0)),
        "settings": {
            "min_area": MIN_AREA,
            "touch_distance": TOUCH_DISTANCE,
            "merge_factor": MERGE_FACTOR,
            "max_solidity": MAX_SOLIDITY,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Графики ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(cells, bins=40, color="#4C72B0")
    axes[0].set_xlabel("Клеток на снимке")
    axes[0].set_ylabel("Снимков")
    axes[1].hist(all_areas, bins=80, range=(0, 4 * median_area), color="#4C72B0")
    axes[1].set_xlabel("Площадь области, пикселей")
    axes[1].set_ylabel("Областей")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distributions.png", dpi=110)
    plt.close(fig)

    # --- Примеры: самые большие комки и подозрительные слияния ---
    by_clump = sorted(rows, key=lambda r: r["largest_clump"], reverse=True)
    for i, row in enumerate(by_clump[:N_EXAMPLES]):
        info = per_image[row["stem"]]
        biggest = max(info["clumps"], key=len)
        image = load_image(info["image_path"])
        labels, _ = mask_to_instances(load_mask(info["mask_path"]), min_area=MIN_AREA)
        save_example(row["stem"], image, labels, biggest,
                     "крупнейший комок, клеток: " + str(len(biggest)),
                     OUT_DIR / ("clump_" + str(i) + ".png"))

    by_merge = sorted(rows, key=lambda r: r["suspected_merges"], reverse=True)
    for i, row in enumerate(by_merge[:N_EXAMPLES]):
        info = per_image[row["stem"]]
        if len(info["suspected"]) == 0:
            break
        image = load_image(info["image_path"])
        labels, _ = mask_to_instances(load_mask(info["mask_path"]), min_area=MIN_AREA)
        save_example(row["stem"], image, labels, info["suspected"],
                     "подозрений на слияние: " + str(len(info["suspected"])),
                     OUT_DIR / ("merge_" + str(i) + ".png"))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Результаты в", OUT_DIR)


if __name__ == "__main__":
    main()
