"""Итоговые таблицы и рисунки для отчёта и презентации.

Запуск из корня репозитория (на ПК — там есть данные, чекпоинты и предсказания Cellpose):
    python scripts/make_report_figures.py

Если данных или чекпоинтов нет (например, на Маке), визуальное сравнение пропускается,
а таблицы и графики строятся из results/ — они лежат в git:
    python scripts/make_report_figures.py --no-qualitative

Результаты в results/figures/:
    tables.md          — итоговые таблицы (все клетки и без клеток у края)
    f1_vs_iou.png      — F1 при разных порогах IoU для всех методов
    seed_tradeoff.png  — склейки и разрезы в зависимости от порога маркеров
    comparison_*.png   — один и тот же плотный участок, сегментированный разными методами
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS_DIR = ROOT / "results" / "metrics"
TUNING_DIR = ROOT / "results" / "tuning"
OUT_DIR = ROOT / "results" / "figures"

# Порядок строк в таблицах и подписи методов
METHODS = [
    ("otsu_components", "Otsu + связные области"),
    ("otsu_watershed", "Otsu + watershed"),
    ("unet_mask_only", "U-Net (только маска)"),
    ("cellpose", "Cellpose-SAM (без дообучения)"),
    ("unet_dist_synth_tuned", "Наш метод + синтетика"),
    ("unet_dist_tuned", "Наш метод (U-Net + карта расстояний)"),
]

COLUMNS = [
    ("precision@0.5", "Precision"),
    ("recall@0.5", "Recall"),
    ("f1@0.5", "F1@0.5"),
    ("f1@0.75", "F1@0.75"),
    ("ap@[0.5:0.95]", "AP[.5:.95]"),
    ("aji", "AJI"),
    ("merge_rate", "Merge rate"),
    ("split_rate", "Split rate"),
    ("count_error", "Ошибка подсчёта"),
    ("dice", "Dice"),
    ("seconds_per_image", "с/снимок"),
]

# Для этих метрик меньше — лучше
LOWER_IS_BETTER = {"merge_rate", "split_rate", "count_error", "seconds_per_image"}


def load_metrics(suffix):
    """Словарь method -> метрики из results/metrics/<method><suffix>_test.json."""
    result = {}
    for key, _ in METHODS:
        path = METRICS_DIR / (key + suffix + "_test.json")
        if path.exists():
            with open(path) as f:
                result[key] = json.load(f)
    return result


def make_table(metrics, title):
    """Markdown-таблица; лучшее значение в каждом столбце выделено жирным."""
    best = {}
    for col, _ in COLUMNS:
        values = []
        for key in metrics:
            values.append(metrics[key][col])
        if len(values) == 0:
            continue
        if col in LOWER_IS_BETTER:
            best[col] = min(values)
        else:
            best[col] = max(values)

    lines = ["### " + title, ""]
    header = "| Метод | " + " | ".join(name for _, name in COLUMNS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(COLUMNS) + 1))
    for key, name in METHODS:
        if key not in metrics:
            continue
        cells = []
        for col, _ in COLUMNS:
            value = metrics[key][col]
            text = str(round(value, 3))
            if value == best[col]:
                text = "**" + text + "**"
            cells.append(text)
        lines.append("| " + name + " | " + " | ".join(cells) + " |")

    any_method = next(iter(metrics.values()))
    lines.append("")
    lines.append("Снимков: " + str(any_method["images"]) + ", клеток в разметке: " + str(any_method["cells_gt"]) + ".")
    lines.append("")
    return lines


def plot_f1_vs_iou(metrics, path):
    thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, name in METHODS:
        if key not in metrics:
            continue
        style = "-"
        width = 1.5
        if key == "unet_dist_tuned":
            width = 3.0
        if key == "unet_dist_synth_tuned":
            style = "--"
        ax.plot(thresholds, metrics[key]["f1_per_threshold"], style, marker="o",
                markersize=3, linewidth=width, label=name)
    ax.set_xlabel("Порог IoU для совпадения клеток")
    ax.set_ylabel("F1")
    ax.set_title("F1 при разных порогах IoU (тест, без клеток у края)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_seed_tradeoff(tuning_path, path):
    """Склейки и разрезы в зависимости от seed_threshold при лучшем fg_threshold (val)."""
    with open(tuning_path) as f:
        tuning = json.load(f)
    best_fg = tuning["best"]["fg_threshold"]

    rows = []
    for row in tuning["table"]:
        if row["fg_threshold"] == best_fg:
            rows.append(row)
    rows.sort(key=lambda r: r["seed_threshold"])

    seeds = [r["seed_threshold"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(seeds, [100 * r["merge_rate"] for r in rows], "o-", color="#C44E52", label="склеено, %")
    ax1.plot(seeds, [100 * r["split_rate"] for r in rows], "s-", color="#4C72B0", label="разрезано, %")
    ax1.set_xlabel("Порог маркеров seed_threshold")
    ax1.set_ylabel("Доля клеток, %")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(seeds, [r["f1@0.5"] for r in rows], "^--", color="#55A868", label="F1@0.5")
    ax2.set_ylabel("F1@0.5")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title("Компромисс склеек и разрезов (val, fg_threshold = " + str(best_fg) + ")")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def colorize(labels, seed=0):
    """Каждой клетке — свой случайный цвет, фон — белый."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 230, size=(int(labels.max()) + 1, 3)).astype(np.uint8)
    palette[0] = (255, 255, 255)
    return palette[labels]


def densest_crop(labels, size):
    """Левый верхний угол окна size×size с наибольшим числом касающихся клеток.

    Координаты всегда зажимаются в границы снимка, чтобы окно не вышло за край.
    """
    from scipy import ndimage as ndi

    height, width = labels.shape
    size = min(size, height, width)
    half = size // 2

    # Касание: рядом есть клетка с другим номером (в том числе через тонкую линию фона)
    neighbor_max = ndi.maximum_filter(labels, size=5)
    big = np.iinfo(np.int32).max
    neighbor_min = ndi.minimum_filter(np.where(labels > 0, labels, big), size=5)
    touching = (labels > 0) & ((neighbor_max != labels) | (neighbor_min != labels)) & (neighbor_min != big)
    density = ndi.uniform_filter(touching.astype(np.float32), size=size)

    if density.max() > 0:
        cy, cx = np.unravel_index(int(density.argmax()), density.shape)
    else:
        cy = height // 2
        cx = width // 2

    y0 = int(np.clip(cy - half, 0, height - size))
    x0 = int(np.clip(cx - half, 0, width - size))
    return y0, x0, size


def make_comparisons(n_images, crop_size):
    """Визуальное сравнение методов на самых плотных тестовых снимках."""
    from PIL import Image

    from baselines.classical import otsu_watershed
    from src.data_io import list_pairs, load_image, load_mask, mask_to_instances
    from src.model import load_checkpoint
    from src.predict import segment
    from src.utils import get_device

    data_root = ROOT / "data" / "raw" / "bbbc041seg"
    with open(ROOT / "configs" / "split_bbbc041seg.json") as f:
        test_stems = set(json.load(f)["test"])

    # Самые «комковатые» тестовые снимки по результатам EDA
    with open(ROOT / "results" / "eda" / "per_image.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["stem"] in test_stems]
    rows.sort(key=lambda r: int(r["cells_in_clumps"]), reverse=True)
    chosen = [r["stem"] for r in rows[:n_images]]

    paths = {}
    for stem, image_path, mask_path in list_pairs(data_root):
        paths[stem] = (image_path, mask_path)

    device = get_device()
    model_dist, config_dist = load_checkpoint(ROOT / "checkpoints" / "unet_dist" / "best.pth", device)
    model_mask, _ = load_checkpoint(ROOT / "checkpoints" / "unet_mask_only" / "best.pth", device)
    with open(TUNING_DIR / "unet_dist_val.json") as f:
        best = json.load(f)["best"]
    cellpose_dir = ROOT / "results" / "predictions" / "cellpose"

    for index, stem in enumerate(chosen):
        image_path, mask_path = paths[stem]
        image = load_image(image_path)
        gt, _ = mask_to_instances(load_mask(mask_path), min_area=200)

        panels = [("Снимок", image, None), ("Разметка", None, gt)]
        panels.append(("Otsu + watershed", None, otsu_watershed(image)))
        panels.append(("U-Net (только маска)", None,
                       segment(model_mask, image, device, postprocess="components")))
        cellpose_path = cellpose_dir / (stem + ".png")
        if cellpose_path.exists():
            panels.append(("Cellpose-SAM", None, np.array(Image.open(cellpose_path)).astype(np.int32)))
        panels.append(("Наш метод", None,
                       segment(model_dist, image, device, postprocess=config_dist["postprocess"],
                               fg_threshold=best["fg_threshold"], seed_threshold=best["seed_threshold"])))

        y0, x0, size = densest_crop(gt, crop_size)
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.0), layout="constrained")
        for ax, (title, picture, labels) in zip(axes, panels):
            if picture is not None:
                shown = picture[y0:y0 + size, x0:x0 + size]
            else:
                crop = labels[y0:y0 + size, x0:x0 + size]
                count = len(np.unique(crop)) - 1
                shown = colorize(crop, seed=index)
                title = title + "\n" + str(count) + " объектов"
            ax.imshow(shown)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.savefig(OUT_DIR / ("comparison_" + str(index) + ".png"), dpi=130)
        plt.close(fig)
        print("Сохранено сравнение:", stem[:8])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qualitative", action="store_true",
                        help="не строить визуальные сравнения (нужны данные и чекпоинты)")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=500)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cells = load_metrics("")
    no_border = load_metrics("_noborder")
    lines = ["# Итоговые результаты на тестовой выборке", ""]
    lines += make_table(no_border, "Без клеток у края снимка (основная)")
    lines += make_table(all_cells, "Все клетки")
    with open(OUT_DIR / "tables.md", "w") as f:
        f.write("\n".join(lines))
    print("Сохранено:", OUT_DIR / "tables.md")

    plot_f1_vs_iou(no_border, OUT_DIR / "f1_vs_iou.png")
    print("Сохранено:", OUT_DIR / "f1_vs_iou.png")

    plot_seed_tradeoff(TUNING_DIR / "unet_dist_val.json", OUT_DIR / "seed_tradeoff.png")
    print("Сохранено:", OUT_DIR / "seed_tradeoff.png")

    if not args.no_qualitative:
        make_comparisons(args.n_images, args.crop_size)


if __name__ == "__main__":
    main()
