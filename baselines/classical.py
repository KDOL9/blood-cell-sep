"""Классические методы без нейросетей — точка отсчёта для сравнения.

1. otsu_components: порог Otsu + связные области.
   Каждое пятно = одна клетка, слипшиеся клетки НЕ разделяются.
2. otsu_watershed: порог Otsu + карта расстояний + watershed.
   Классический способ разрезать слипшиеся клетки.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk, remove_small_objects
from skimage.segmentation import watershed

MIN_AREA = 200


def foreground_mask(image):
    """Маска «клетка/фон» по порогу Otsu.

    Otsu сам выбирает порог яркости, который лучше всего делит пиксели
    на две группы (тёмные клетки и светлый фон).
    """
    gray = rgb2gray(image)
    threshold = threshold_otsu(gray)
    mask = gray < threshold  # клетки темнее фона

    # У эритроцитов светлый центр — Otsu может сделать в нём «дырку». Заливаем.
    mask = ndi.binary_fill_holes(mask)
    # Размыкание убирает тонкие «мостики» и мелкий мусор
    mask = binary_opening(mask, disk(3))
    mask = remove_small_objects(mask, min_size=MIN_AREA)
    return mask


def otsu_components(image):
    """Otsu + связные области: каждое пятно считается одной клеткой."""
    mask = foreground_mask(image)
    labels, _ = ndi.label(mask)
    return labels.astype(np.int32)


def estimate_cell_radius(mask):
    """Типичный радиус клетки на этом снимке.

    Берём медианную площадь пятен (большинство пятен — одиночные клетки)
    и считаем радиус круга такой же площади: r = sqrt(S / pi).
    """
    labels, count = ndi.label(mask)
    if count == 0:
        return 10.0
    areas = np.bincount(labels.ravel())[1:]
    median_area = float(np.median(areas))
    return float(np.sqrt(median_area / np.pi))


def otsu_watershed(image, min_distance_factor=0.7):
    """Otsu + карта расстояний + watershed.

    1. Маска клеток по Otsu.
    2. Карта расстояний: в каждом пикселе клетки — расстояние до ближайшего фона.
       В центре клетки значение максимальное, к краям падает.
    3. Локальные максимумы карты = центры клеток (маркеры).
       min_distance не даёт поставить два маркера в одну клетку.
    4. Watershed «заливает» карту от каждого маркера; где заливки встречаются —
       проходит граница между клетками.
    """
    mask = foreground_mask(image)
    distance = ndi.distance_transform_edt(mask)
    # Сглаживание убирает мелкие ложные пики от неровных краёв
    distance = ndi.gaussian_filter(distance, sigma=2)

    radius = estimate_cell_radius(mask)
    min_distance = max(3, int(radius * min_distance_factor))

    blobs, _ = ndi.label(mask)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=blobs, exclude_border=False)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, (y, x) in enumerate(peaks):
        markers[y, x] = index + 1

    # Watershed ищет «впадины», поэтому подаём карту со знаком минус:
    # центры клеток становятся самыми глубокими точками
    labels = watershed(-distance, markers, mask=mask)
    return labels.astype(np.int32)


METHODS = {
    "otsu_components": otsu_components,
    "otsu_watershed": otsu_watershed,
}
