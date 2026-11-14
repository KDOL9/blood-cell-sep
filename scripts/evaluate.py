"""Оценка метода сегментации на выбранной части датасета.

Запуск из корня репозитория:
    python scripts/evaluate.py --method otsu_watershed --split test
    python scripts/evaluate.py --method otsu_components --split test --limit 20
    python scripts/evaluate.py --method model --checkpoint checkpoints/unet_dist/best.pth --split test

Результаты: results/metrics/<method>_<split>.json и .csv (по снимкам).
Пара примеров с наложением предсказания: results/metrics/examples/.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.classical import METHODS
from src.data_io import find_suspected_merges, list_pairs, load_image, load_mask, mask_to_instances
from src.metrics import aggregate_stats, image_stats
from src.model import load_checkpoint
from src.predict import segment
from src.utils import get_device

DATA_ROOT = ROOT / "data" / "raw" / "bbbc041seg"
SPLIT_PATH = ROOT / "configs" / "split_bbbc041seg.json"
OUT_DIR = ROOT / "results" / "metrics"
GT_MIN_AREA = 200
N_EXAMPLES = 3


def save_example(stem, image, gt, pred, path):
    """Снимок | разметка | предсказание; границы клеток — цветом."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].imshow(image)
    axes[0].set_title("снимок")

    for ax, labels, title in [(axes[1], gt, "разметка"), (axes[2], pred, "предсказание")]:
        shown = image.copy()
        # Границы утолщаем, иначе на уменьшенной картинке их не видно
        edges = binary_dilation(find_boundaries(labels, mode="inner"), iterations=2)
        shown[edges] = (255, 40, 40)
        ax.imshow(shown)
        ax.set_title(title + ": " + str(len(np.unique(labels)) - 1) + " клеток")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(stem[:8])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS) + ["model"])
    parser.add_argument("--checkpoint", default="", help="путь к чекпоинту (для --method model)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N снимков (для быстрой проверки)")
    args = parser.parse_args()

    with open(SPLIT_PATH) as f:
        split_stems = set(json.load(f)[args.split])

    pairs = []
    for stem, image_path, mask_path in list_pairs(DATA_ROOT):
        if stem in split_stems:
            pairs.append((stem, image_path, mask_path))
    if args.limit > 0:
        pairs = pairs[:args.limit]

    method_name = args.method
    if args.method == "model":
        if args.checkpoint == "":
            parser.error("для --method model нужен --checkpoint")
        device = get_device()
        model, config = load_checkpoint(args.checkpoint, device)
        method_name = config["run_name"]

        def method(image):
            return segment(model, image, device, postprocess=config["postprocess"])
    else:
        method = METHODS[args.method]
    out_dir = OUT_DIR
    examples_dir = OUT_DIR / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    stats_list = []
    rows = []
    start = time.time()

    for index, (stem, image_path, mask_path) in enumerate(tqdm(pairs, desc=method_name)):
        image = load_image(image_path)
        gt, _ = mask_to_instances(load_mask(mask_path), min_area=GT_MIN_AREA)
        ignore_ids = find_suspected_merges(gt)

        pred = method(image)

        stats = image_stats(gt, pred, ignore_ids)
        stats_list.append(stats)
        rows.append({
            "stem": stem,
            "cells_gt": stats["n_gt"],
            "cells_pred": stats["n_pred"],
            "tp@0.5": int(stats["tp"][0]),
            "merged_gt": stats["merged_gt"],
            "split_gt": stats["split_gt"],
        })

        if index < N_EXAMPLES:
            save_example(stem, image, gt, pred,
                         examples_dir / (method_name + "_" + str(index) + ".png"))

    elapsed = time.time() - start
    metrics = aggregate_stats(stats_list)
    metrics["method"] = method_name
    metrics["split"] = args.split
    metrics["seconds_per_image"] = round(elapsed / max(1, len(pairs)), 3)

    name = method_name + "_" + args.split
    with open(out_dir / (name + ".json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / (name + ".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
