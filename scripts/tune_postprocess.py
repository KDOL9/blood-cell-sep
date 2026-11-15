"""Подбор порогов постобработки на val и анализ ложных срабатываний.

Пороги подбираем ТОЛЬКО на val: если подбирать на test, итоговые метрики
окажутся завышенными (мы бы подогнали метод под тестовые снимки).

Запуск из корня репозитория:
    python scripts/tune_postprocess.py --checkpoint checkpoints/unet_dist/best.pth

Результат: results/tuning/<run>_val.json — таблица всех комбинаций и лучшие пороги.
Потом evaluate.py с флагом --tuned сам возьмёт лучшие пороги из этого файла.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import border_labels, find_suspected_merges, list_pairs, load_image, load_mask, mask_to_instances
from src.metrics import aggregate_stats, apply_ignore, image_stats, iou_matrix, overlap_matrix
from src.model import load_checkpoint
from src.postprocess import components, distance_watershed
from src.predict import predict_maps
from src.utils import get_device

DATA_ROOT = ROOT / "data" / "raw" / "bbbc041seg"
SPLIT_PATH = ROOT / "configs" / "split_bbbc041seg.json"
OUT_DIR = ROOT / "results" / "tuning"
GT_MIN_AREA = 200

FG_THRESHOLDS = [0.4, 0.5, 0.6]
SEED_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def run_postprocess(postprocess, fg_prob, dist, fg_threshold, seed_threshold):
    if postprocess == "components":
        return components(fg_prob, fg_threshold)
    return distance_watershed(fg_prob, dist, fg_threshold, seed_threshold)


def analyze_false_positives(gt, pred, ignore_ids):
    """Где ложные срабатывания: у края снимка или внутри, и какого они размера.

    Ложное срабатывание (FP) — предсказанная клетка, у которой ни с одной
    размеченной клеткой IoU не больше 0.5.
    """
    gt, pred = apply_ignore(gt, pred, ignore_ids)
    inter, gt_area, pred_area = overlap_matrix(gt, pred)
    iou = iou_matrix(inter, gt_area, pred_area)

    n_pred = len(pred_area)
    if n_pred == 0:
        return {"fp": 0, "fp_border": 0, "fp_small": 0, "fp_no_overlap": 0}

    best_iou = np.zeros(n_pred)
    overlap_share = np.zeros(n_pred)
    if len(gt_area) > 0:
        best_iou = iou.max(axis=0)
        # Какая доля площади предсказания лежит хоть на какой-то размеченной клетке
        overlap_share = inter.sum(axis=0) / np.maximum(pred_area, 1)

    border = border_labels(pred)
    median_gt_area = 0.0
    if len(gt_area) > 0:
        median_gt_area = float(np.median(gt_area))

    result = {"fp": 0, "fp_border": 0, "fp_small": 0, "fp_no_overlap": 0}
    for j in range(n_pred):
        if best_iou[j] > 0.5:
            continue
        result["fp"] += 1
        label = j + 1
        if label in border:
            result["fp_border"] += 1
        if pred_area[j] < 0.3 * median_gt_area:
            result["fp_small"] += 1
        if overlap_share[j] < 0.1:
            # Почти не пересекается с разметкой: либо мусор, либо неразмеченная клетка
            result["fp_no_overlap"] += 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N снимков val")
    args = parser.parse_args()

    device = get_device()
    model, config = load_checkpoint(args.checkpoint, device)
    run_name = config["run_name"]
    postprocess = config["postprocess"]

    with open(SPLIT_PATH) as f:
        val_stems = set(json.load(f)["val"])
    pairs = []
    for stem, image_path, mask_path in list_pairs(DATA_ROOT):
        if stem in val_stems:
            pairs.append((stem, image_path, mask_path))
    if args.limit > 0:
        pairs = pairs[:args.limit]

    # Шаг 1: прогоняем сеть по всем снимкам ОДИН раз и запоминаем карты.
    # Перебор порогов потом не требует сети — только постобработку.
    # Карты храним в uint8 (0..255), чтобы 199 снимков влезли в память.
    cache = []
    for stem, image_path, mask_path in tqdm(pairs, desc="сеть"):
        image = load_image(image_path)
        gt, _ = mask_to_instances(load_mask(mask_path), min_area=GT_MIN_AREA)
        fg_prob, dist = predict_maps(model, image, device)
        cache.append({
            "gt": gt,
            "ignore": find_suspected_merges(gt),
            "fg": (fg_prob * 255).astype(np.uint8),
            "dist": (dist * 255).astype(np.uint8),
        })

    # Шаг 2: перебираем пороги
    seed_values = SEED_THRESHOLDS
    if postprocess == "components":
        seed_values = [0.5]  # для модели «только маска» карта расстояний не используется

    table = []
    for fg_threshold in FG_THRESHOLDS:
        for seed_threshold in seed_values:
            stats_list = []
            for item in tqdm(cache, desc="fg=" + str(fg_threshold) + " seed=" + str(seed_threshold), leave=False):
                pred = run_postprocess(postprocess, item["fg"] / 255.0, item["dist"] / 255.0,
                                       fg_threshold, seed_threshold)
                stats_list.append(image_stats(item["gt"], pred, item["ignore"]))
            metrics = aggregate_stats(stats_list)
            row = {"fg_threshold": fg_threshold, "seed_threshold": seed_threshold}
            for key in ["f1@0.5", "precision@0.5", "recall@0.5", "ap@[0.5:0.95]", "aji", "merge_rate", "split_rate"]:
                row[key] = metrics[key]
            table.append(row)
            print(row)

    best = table[0]
    for row in table:
        if row["f1@0.5"] > best["f1@0.5"]:
            best = row

    # Шаг 3: анализ ложных срабатываний при лучших порогах
    fp_totals = {"fp": 0, "fp_border": 0, "fp_small": 0, "fp_no_overlap": 0}
    for item in cache:
        pred = run_postprocess(postprocess, item["fg"] / 255.0, item["dist"] / 255.0,
                               best["fg_threshold"], best["seed_threshold"])
        fp = analyze_false_positives(item["gt"], pred, item["ignore"])
        for key in fp_totals:
            fp_totals[key] += fp[key]

    result = {
        "run_name": run_name,
        "postprocess": postprocess,
        "split": "val",
        "images": len(cache),
        "best": best,
        "false_positives_at_best": fp_totals,
        "table": table,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (run_name + "_val.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("Лучшие пороги:", best)
    print("Ложные срабатывания при них:", fp_totals)
    print("Сохранено:", out_path)


if __name__ == "__main__":
    main()
