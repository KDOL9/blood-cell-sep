"""Чтение датасета BBBC041Seg: пары снимок–маска и разбиение маски на клетки."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Связность 4: пиксели считаются соседями только по горизонтали и вертикали.
# Касающиеся клетки в разметке разделены линией толщиной в 1 пиксель.
# При связности 8 (с диагоналями) две клетки могли бы «перешагнуть» через
# эту линию по диагонали и склеиться в одну.
FOUR_CONNECTIVITY = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
)


def find_folder(root, name):
    """Ищет внутри root папку с заданным именем (на любой глубине)."""
    for path in Path(root).rglob(name):
        if path.is_dir():
            return path
    raise FileNotFoundError("Не нашёл папку '" + name + "' внутри " + str(root))


def list_pairs(root):
    """Возвращает отсортированный список пар (stem, путь к снимку, путь к маске).

    stem — имя файла без расширения. По нему сопоставляются снимок и маска,
    потому что у снимка может быть .jpg, а у маски .png.
    """
    image_dir = find_folder(root, "Original")
    mask_dir = find_folder(root, "Mask")

    images = {}
    for path in image_dir.iterdir():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.stem] = path

    masks = {}
    for path in mask_dir.iterdir():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            masks[path.stem] = path

    pairs = []
    for stem in sorted(images):
        if stem in masks:
            pairs.append((stem, images[stem], masks[stem]))

    missing = len(images) + len(masks) - 2 * len(pairs)
    if missing > 0:
        print("Предупреждение: файлов без пары:", missing)

    return pairs


def load_image(path):
    """Читает снимок как RGB-массив uint8 формы (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path):
    """Читает маску как булев массив (H, W): True — клетка, False — фон."""
    mask = np.array(Image.open(path).convert("L"))
    return mask > 127


def mask_to_instances(mask, min_area=0):
    """Превращает бинарную маску в маску экземпляров.

    Каждая связная белая область получает свой номер 1, 2, 3, ...; фон — 0.
    Области меньше min_area пикселей (обычно шум) удаляются.

    Возвращает (labels, count): массив номеров int32 и число клеток.
    """
    labels, count = ndi.label(mask, structure=FOUR_CONNECTIVITY)

    if min_area > 0:
        areas = np.bincount(labels.ravel())
        too_small = areas < min_area
        too_small[0] = False
        labels[too_small[labels]] = 0
        labels, count = ndi.label(labels > 0, structure=FOUR_CONNECTIVITY)

    return labels.astype(np.int32), count
