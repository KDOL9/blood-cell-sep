"""Синтетические слипания: вставляем на снимок копии одиночных клеток вплотную к другим.

Зачем: комки — самый трудный случай, а ошибки сети почти все в них.
Синтетика даёт сети больше примеров касающихся и чуть перекрывающихся клеток,
причём с идеальной разметкой: мы точно знаем, где кончается каждая вставленная клетка.

Как:
  1. Находим на снимке «хорошие» одиночные клетки: обычного размера, выпуклые,
     не у края и не слитые в разметке. Они служат «спрайтами» — их пиксели и маску копируем.
  2. Каждый спрайт случайно поворачиваем/отражаем и ставим рядом с другой клеткой так,
     чтобы расстояние между центрами было 75–100% от суммы радиусов:
     клетки касаются или немного перекрываются.
  3. Вставленная клетка закрывает собой то, что под ней (как в реальном мазке, где одна
     клетка лежит поверх другой). Если от закрытой клетки осталось меньше половины —
     её помечаем как «игнорировать»: такие огрызки неинформативны.
"""
import numpy as np
from skimage.measure import regionprops

from src.data_io import border_labels


def find_sprites(labels, ignore_ids):
    """Одиночные клетки, пригодные для копирования."""
    props = regionprops(labels)
    if len(props) == 0:
        return []

    areas = []
    for prop in props:
        areas.append(prop.area)
    median_area = float(np.median(areas))

    border = border_labels(labels)
    ignore = set(ignore_ids)
    sprites = []
    for prop in props:
        if prop.label in border or prop.label in ignore:
            continue
        if prop.area < 0.6 * median_area or prop.area > 1.4 * median_area:
            continue
        if prop.solidity < 0.93:
            continue
        sprites.append(prop)
    return sprites


def cut_sprite(image, labels, prop):
    """Вырезает пиксели и маску клетки по её прямоугольнику."""
    y0, x0, y1, x1 = prop.bbox
    mask = labels[y0:y1, x0:x1] == prop.label
    pixels = image[y0:y1, x0:x1].copy()
    return pixels, mask


def random_orientation(pixels, mask, rng):
    """Случайный поворот на 0/90/180/270 градусов и отражение — без интерполяции."""
    k = int(rng.integers(0, 4))
    pixels = np.rot90(pixels, k)
    mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        pixels = pixels[:, ::-1]
        mask = mask[:, ::-1]
    return pixels, mask


def add_synthetic_clumps(image, labels, ignore_ids, rng, min_cells=3, max_cells=10):
    """Вставляет от min_cells до max_cells клеток. Возвращает новые image, labels, ignore_ids."""
    sprites = find_sprites(labels, ignore_ids)
    if len(sprites) < 2:
        return image, labels, list(ignore_ids)

    # Спрайты вырезаем из исходного снимка, а вставляем в копию —
    # иначе могли бы скопировать клетку вместе с уже вставленной поверх неё
    source_image = image
    source_labels = labels
    image = image.copy()
    labels = labels.copy()
    ignore = set(ignore_ids)
    height, width = labels.shape

    original_area = np.bincount(labels.ravel())
    next_id = int(labels.max()) + 1

    # Центры клеток, к которым можно «прилепить» новую: и исходные, и уже вставленные
    anchors = []
    for prop in sprites:
        cy, cx = prop.centroid
        anchors.append((cy, cx, prop.equivalent_diameter_area / 2))

    n_cells = int(rng.integers(min_cells, max_cells + 1))
    for _ in range(n_cells):
        sprite = sprites[int(rng.integers(0, len(sprites)))]
        pixels, mask = cut_sprite(source_image, source_labels, sprite)
        pixels, mask = random_orientation(pixels, mask, rng)
        sprite_radius = sprite.equivalent_diameter_area / 2

        anchor_y, anchor_x, anchor_radius = anchors[int(rng.integers(0, len(anchors)))]
        angle = rng.uniform(0, 2 * np.pi)
        gap = rng.uniform(0.75, 1.0) * (anchor_radius + sprite_radius)
        center_y = anchor_y + gap * np.sin(angle)
        center_x = anchor_x + gap * np.cos(angle)

        sprite_h, sprite_w = mask.shape
        top = int(round(center_y - sprite_h / 2))
        left = int(round(center_x - sprite_w / 2))
        if top < 1 or left < 1 or top + sprite_h > height - 1 or left + sprite_w > width - 1:
            continue  # не влезает в кадр целиком — пропускаем

        region_labels = labels[top:top + sprite_h, left:left + sprite_w]
        region_image = image[top:top + sprite_h, left:left + sprite_w]
        region_labels[mask] = next_id
        region_image[mask] = pixels[mask]

        anchors.append((center_y, center_x, sprite_radius))
        next_id += 1

    # Закрытые больше чем наполовину клетки — в игнор
    new_area = np.bincount(labels.ravel(), minlength=len(original_area))
    for cell in range(1, len(original_area)):
        if original_area[cell] > 0 and new_area[cell] < 0.5 * original_area[cell]:
            ignore.add(cell)

    return image, labels, sorted(ignore)
