"""Метрики качества instance-сегментации клеток.

Все функции принимают маски экземпляров: массивы int (H, W),
где 0 — фон, 1..N — номера клеток.

Основа всех метрик — матрица пересечений: сколько пикселей у каждой
размеченной клетки (GT, ground truth) общих с каждой предсказанной.
"""
import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)  # 0.50, 0.55, ..., 0.95


def relabel(labels):
    """Перенумеровывает клетки подряд: 1, 2, 3, ... (без пропусков)."""
    values = np.unique(labels)
    values = values[values > 0]
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[values] = np.arange(1, len(values) + 1)
    return lookup[labels]


def overlap_matrix(gt, pred):
    """Считает пересечения и площади.

    Возвращает:
      inter[i, j] — число общих пикселей у GT-клетки i+1 и предсказанной j+1;
      gt_area[i], pred_area[j] — площади клеток.

    Трюк: кодируем пару номеров (g, p) одним числом g * (P + 1) + p
    и считаем, сколько раз встретилось каждое число (np.bincount).
    """
    n_gt = int(gt.max())
    n_pred = int(pred.max())
    pair_index = gt.astype(np.int64) * (n_pred + 1) + pred.astype(np.int64)
    counts = np.bincount(pair_index.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    counts = counts.reshape(n_gt + 1, n_pred + 1)

    inter = counts[1:, 1:]
    gt_area = counts[1:, :].sum(axis=1)
    pred_area = counts[:, 1:].sum(axis=0)
    return inter, gt_area, pred_area


def iou_matrix(inter, gt_area, pred_area):
    """IoU каждой GT-клетки с каждой предсказанной: пересечение / объединение."""
    union = gt_area[:, None] + pred_area[None, :] - inter
    iou = np.zeros(inter.shape, dtype=np.float64)
    nonzero = union > 0
    iou[nonzero] = inter[nonzero] / union[nonzero]
    return iou


def apply_ignore(gt, pred, ignore_ids):
    """Убирает из оценки области, где разметка ненадёжна (слитые клетки).

    - GT-клетки из ignore_ids удаляются из разметки;
    - предсказанные клетки, у которых больше половины площади лежит
      в этих областях, удаляются из предсказания.
    Так модель не штрафуется за то, что правильно разрезала то,
    что в разметке ошибочно слито.
    """
    if len(ignore_ids) == 0:
        return relabel(gt), relabel(pred)

    ignore_region = np.isin(gt, list(ignore_ids))

    gt_clean = gt.copy()
    gt_clean[ignore_region] = 0

    pred_area = np.bincount(pred.ravel())
    pred_in_ignore = np.bincount(pred[ignore_region].ravel(), minlength=len(pred_area))
    drop = np.zeros(len(pred_area), dtype=bool)
    has_area = pred_area > 0
    drop[has_area] = pred_in_ignore[has_area] / pred_area[has_area] > 0.5
    drop[0] = False

    pred_clean = pred.copy()
    pred_clean[drop[pred]] = 0

    return relabel(gt_clean), relabel(pred_clean)


BORDER_MARGIN = 3  # клетка «у края», если подходит к нему ближе чем на столько пикселей


def border_ids(labels, margin=0):
    """Номера клеток, касающихся края снимка (или подходящих к нему ближе чем на margin пикселей)."""
    m = margin + 1
    edge = np.concatenate([
        labels[:m, :].ravel(), labels[-m:, :].ravel(),
        labels[:, :m].ravel(), labels[:, -m:].ravel(),
    ])
    ids = np.unique(edge)
    return ids[ids > 0]


def remove_border_objects(gt, pred):
    """Исключает из оценки клетки у края снимка.

    Клетки, обрезанные краем кадра, в BBBC041Seg размечены непоследовательно:
    часть есть в разметке, часть нет. Поэтому:
      - из разметки удаляем все клетки, касающиеся края;
      - предсказанную клетку считаем «краевой», если она подходит к краю ближе чем на
        BORDER_MARGIN пикселей (разные методы по-разному обводят клетку у самого края)
        или больше чем наполовину лежит на удалённой краевой клетке разметки;
      - краевую предсказанную клетку удаляем, только если она не накрывает больше
        половины ни одной из оставшихся (внутренних) размеченных клеток.
    Так удаляются неразмеченные и обрезанные клетки у края, но большое склеенное пятно,
    которое задело край и при этом накрывает внутренние клетки, остаётся
    и честно учитывается как склейка.
    """
    gt_border = np.isin(gt, border_ids(gt))
    gt_clean = gt.copy()
    gt_clean[gt_border] = 0

    pred_area = np.bincount(pred.ravel())
    on_removed_gt = np.bincount(pred[gt_border].ravel(), minlength=len(pred_area))
    candidates = set(border_ids(pred, BORDER_MARGIN).tolist())
    for label in range(1, len(pred_area)):
        if pred_area[label] > 0 and on_removed_gt[label] > 0.5 * pred_area[label]:
            candidates.add(label)

    inter, gt_area, _ = overlap_matrix(gt_clean, pred)
    drop = []
    for label in sorted(candidates):
        covers_interior_cell = False
        if inter.shape[0] > 0 and label - 1 < inter.shape[1]:
            covers_interior_cell = bool((inter[:, label - 1] > 0.5 * gt_area).any())
        if not covers_interior_cell:
            drop.append(label)

    pred_clean = pred.copy()
    pred_clean[np.isin(pred, drop)] = 0
    return gt_clean, pred_clean


def image_stats(gt, pred, ignore_ids=(), exclude_border=False):
    """Все «сырые» счётчики для одного снимка.

    Метрики по всему тесту потом считаются суммированием этих счётчиков
    (aggregate_stats) — так каждая клетка весит одинаково, а не каждый снимок.
    """
    if exclude_border:
        gt, pred = remove_border_objects(gt, pred)
    gt, pred = apply_ignore(gt, pred, ignore_ids)
    inter, gt_area, pred_area = overlap_matrix(gt, pred)
    iou = iou_matrix(inter, gt_area, pred_area)
    n_gt = len(gt_area)
    n_pred = len(pred_area)

    # --- TP / FP / FN при каждом пороге IoU ---
    # При IoU > 0.5 у клетки может быть не больше одной пары:
    # две разные клетки не могут перекрывать одну и ту же больше чем наполовину.
    # Поэтому сопоставление однозначное, и достаточно посчитать пары выше порога.
    tp = []
    for threshold in IOU_THRESHOLDS:
        tp.append(int((iou > threshold).sum()))
    tp = np.array(tp)
    fp = n_pred - tp
    fn = n_gt - tp

    # --- AJI: для каждой GT-клетки берём предсказанную с максимальным IoU ---
    aji_inter = 0
    aji_union = 0
    used = np.zeros(n_pred, dtype=bool)
    for i in range(n_gt):
        if n_pred > 0 and iou[i].max() > 0:
            j = int(iou[i].argmax())
            aji_inter += int(inter[i, j])
            aji_union += int(gt_area[i] + pred_area[j] - inter[i, j])
            used[j] = True
        else:
            aji_union += int(gt_area[i])
    # Предсказанные клетки, которые ни к кому не подошли, добавляются в знаменатель
    aji_union += int(pred_area[~used].sum())

    # --- Склеенные и разрезанные клетки ---
    # Предсказание «покрывает» GT-клетку, если накрывает больше половины её площади.
    # Склейка: одно предсказание покрывает 2+ GT-клеток.
    merged_gt = 0
    split_gt = 0
    if n_gt > 0 and n_pred > 0:
        covers = inter > 0.5 * gt_area[:, None]
        gt_per_pred = covers.sum(axis=0)
        for i in range(n_gt):
            covering = np.where(covers[i])[0]
            if len(covering) > 0 and gt_per_pred[covering[0]] >= 2:
                merged_gt += 1
        # Разрез: у GT-клетки 2+ предсказаний, каждое из которых больше чем
        # наполовину лежит внутри неё
        belongs = inter > 0.5 * pred_area[None, :]
        split_gt = int((belongs.sum(axis=1) >= 2).sum())

    # --- Пиксельный Dice маски «клетка/фон» ---
    gt_fg = gt > 0
    pred_fg = pred > 0
    dice_inter = int((gt_fg & pred_fg).sum())
    dice_total = int(gt_fg.sum() + pred_fg.sum())

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "aji_inter": aji_inter,
        "aji_union": aji_union,
        "merged_gt": merged_gt,
        "split_gt": split_gt,
        "dice_inter": dice_inter,
        "dice_total": dice_total,
    }


def aggregate_stats(stats_list):
    """Сводит счётчики всех снимков в итоговые метрики."""
    tp = np.zeros(len(IOU_THRESHOLDS))
    fp = np.zeros(len(IOU_THRESHOLDS))
    fn = np.zeros(len(IOU_THRESHOLDS))
    totals = {
        "n_gt": 0, "n_pred": 0, "aji_inter": 0, "aji_union": 0,
        "merged_gt": 0, "split_gt": 0, "dice_inter": 0, "dice_total": 0,
    }
    count_errors = []
    aji_per_image = []

    for s in stats_list:
        tp += s["tp"]
        fp += s["fp"]
        fn += s["fn"]
        for key in totals:
            totals[key] += s[key]
        if s["n_gt"] > 0:
            count_errors.append(abs(s["n_pred"] - s["n_gt"]) / s["n_gt"])
        if s["aji_union"] > 0:
            aji_per_image.append(s["aji_inter"] / s["aji_union"])

    def safe_div(a, b):
        if b == 0:
            return 0.0
        return float(a / b)

    # Индекс порога 0.5 в списке IOU_THRESHOLDS
    i50 = 0
    i75 = 5

    # AP в духе Data Science Bowl: TP / (TP + FP + FN), усреднённое по порогам
    ap_per_threshold = []
    for k in range(len(IOU_THRESHOLDS)):
        ap_per_threshold.append(safe_div(tp[k], tp[k] + fp[k] + fn[k]))

    # AJI считаем по каждому снимку и усредняем (как принято в литературе).
    # Сумма по всем снимкам сразу дала бы огромный вес нескольким плотным
    # снимкам, где одно гигантское пятно накрывает сотню клеток.
    aji = 0.0
    if len(aji_per_image) > 0:
        aji = float(np.mean(aji_per_image))

    count_error = 0.0
    if len(count_errors) > 0:
        count_error = float(np.mean(count_errors))

    precision = safe_div(tp[i50], tp[i50] + fp[i50])
    recall = safe_div(tp[i50], tp[i50] + fn[i50])

    return {
        "images": len(stats_list),
        "cells_gt": totals["n_gt"],
        "cells_pred": totals["n_pred"],
        "precision@0.5": round(precision, 4),
        "recall@0.5": round(recall, 4),
        "f1@0.5": round(safe_div(2 * tp[i50], 2 * tp[i50] + fp[i50] + fn[i50]), 4),
        "f1@0.75": round(safe_div(2 * tp[i75], 2 * tp[i75] + fp[i75] + fn[i75]), 4),
        "ap@0.5": round(ap_per_threshold[i50], 4),
        "ap@[0.5:0.95]": round(float(np.mean(ap_per_threshold)), 4),
        "aji": round(aji, 4),
        "aji_pooled": round(safe_div(totals["aji_inter"], totals["aji_union"]), 4),
        "merge_rate": round(safe_div(totals["merged_gt"], totals["n_gt"]), 4),
        "split_rate": round(safe_div(totals["split_gt"], totals["n_gt"]), 4),
        "count_error": round(count_error, 4),
        "dice": round(safe_div(2 * totals["dice_inter"], totals["dice_total"]), 4),
        "f1_per_threshold": [round(safe_div(2 * tp[k], 2 * tp[k] + fp[k] + fn[k]), 4)
                             for k in range(len(IOU_THRESHOLDS))],
    }
