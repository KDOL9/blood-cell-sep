"""Целевые карты для обучения: что именно сеть должна научиться предсказывать.

Из маски экземпляров (каждая клетка со своим номером) делаем три карты:
  fg    — 1 на клетке, 0 на фоне (маска «клетка/фон»);
  dist  — нормированная карта расстояний: в центре каждой клетки 1, к краю падает до 0;
  valid — 1 там, где лосс считается; 0 на ненадёжных областях (слитые в разметке клетки).
"""
import numpy as np
from scipy import ndimage as ndi


def normalized_distance(labels):
    """Карта расстояний, нормированная отдельно для каждой клетки.

    distance_transform_edt даёт для каждого пикселя клетки расстояние до ближайшего
    пикселя фона. Касающиеся клетки в разметке разделены линией фона толщиной 1 пиксель,
    поэтому у границы между ними расстояние тоже падает до нуля — каждая клетка
    получает свой собственный «холм».

    Делим на максимум внутри каждой клетки: и у маленькой, и у большой клетки
    вершина холма = 1. Тогда центры клеток можно находить одним порогом
    (например, dist > 0.5), не зная заранее их размер.
    """
    dist = ndi.distance_transform_edt(labels > 0)
    n_cells = int(labels.max())
    if n_cells == 0:
        return dist.astype(np.float32)

    # Максимум расстояния внутри каждой клетки — одной векторной операцией
    peak = ndi.maximum(dist, labels=labels, index=np.arange(1, n_cells + 1))
    peak = np.concatenate([[1.0], np.asarray(peak, dtype=np.float64)])
    peak[peak == 0] = 1.0

    normalized = dist / peak[labels]
    normalized[labels == 0] = 0.0
    return normalized.astype(np.float32)


def make_targets(labels, ignore_ids=()):
    """Строит fg, dist и valid из маски экземпляров."""
    fg = (labels > 0).astype(np.uint8)
    dist = normalized_distance(labels)

    valid = np.ones(labels.shape, dtype=np.uint8)
    if len(ignore_ids) > 0:
        ignore_region = np.isin(labels, list(ignore_ids))
        # Расширяем на пару пикселей: граница слитой области тоже ненадёжна
        ignore_region = ndi.binary_dilation(ignore_region, iterations=2)
        valid[ignore_region] = 0

    return fg, dist, valid
