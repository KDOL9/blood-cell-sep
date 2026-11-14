"""Постобработка: из карт, предсказанных сетью, получаем отдельные клетки."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

MIN_AREA = 200


def components(fg_prob, fg_threshold=0.5):
    """Без разделения: каждое связное пятно маски = одна клетка.

    Для модели «только маска» (бейзлайн «U-Net без карты расстояний»).
    """
    mask = fg_prob > fg_threshold
    mask = remove_small_objects(mask, max_size=MIN_AREA - 1)
    labels, _ = ndi.label(mask)
    return labels.astype(np.int32)


def distance_watershed(fg_prob, dist, fg_threshold=0.5, seed_threshold=0.5):
    """Наш метод: маркеры из карты расстояний + watershed.

    1. Маска клеток: fg_prob > fg_threshold.
    2. Маркеры (центры клеток): dist > seed_threshold.
       Карта нормирована так, что в центре каждой клетки 1, а у границы 0.
       Значит, порог 0.5 оставляет от каждой клетки отдельное «ядро»,
       и у двух слипшихся клеток ядра не соприкасаются — получаются два маркера.
    3. Watershed разрастает каждый маркер по карте расстояний до границ маски.
    """
    mask = fg_prob > fg_threshold
    mask = remove_small_objects(mask, max_size=MIN_AREA - 1)

    seeds = (dist > seed_threshold) & mask
    markers, _ = ndi.label(seeds)

    labels = watershed(-dist, markers, mask=mask)
    # Куски маски, куда не попал ни один маркер, watershed оставляет нулём — это нормально:
    # обычно это мусор или обрывки клеток у края
    labels = remove_small_objects(labels, max_size=MIN_AREA - 1)
    return labels.astype(np.int32)
