"""PyTorch Dataset: выдаёт снимок и целевые карты для обучения.

Обучение — на случайных кропах 512×512 с аугментациями.
Валидация — на целых снимках (кропы и аугментации не нужны).
"""
import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset

from src.data_io import find_suspected_merges, load_image, load_mask, mask_to_instances
from src.synthetic import add_synthetic_clumps
from src.targets import make_targets

GT_MIN_AREA = 200

# Средние и стандартные отклонения ImageNet: энкодер предобучен на таких входах
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_train_augmentations(crop_size):
    """Аугментации: каждый раз немного другой вариант того же снимка.

    Геометрия (поворот, отражение, масштаб) применяется одинаково к снимку и ко всем картам.
    Цвет — только к снимку: окраска мазков сильно различается между лабораториями,
    и модель должна быть к этому устойчива.
    """
    return A.Compose([
        A.Affine(scale=(0.75, 1.25), p=0.5),
        A.PadIfNeeded(min_height=crop_size, min_width=crop_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(height=crop_size, width=crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ])


def normalize_image(image):
    """uint8 (H, W, 3) -> float32 тензор (3, H, W), нормированный как в ImageNet."""
    x = image.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class CellDataset(Dataset):
    """pairs — список (stem, путь к снимку, путь к маске)."""

    def __init__(self, pairs, crop_size=512, train=True, samples_per_image=1, synthetic=None):
        self.pairs = pairs
        self.train = train
        # synthetic — словарь из конфига: {"enabled": ..., "prob": ..., "min_cells": ..., "max_cells": ...}
        self.synthetic = synthetic
        # Снимок большой, а кроп маленький: за эпоху берём из каждого снимка несколько кропов
        self.samples_per_image = samples_per_image
        self.augment = None
        if train:
            self.augment = build_train_augmentations(crop_size)

    def __len__(self):
        return len(self.pairs) * self.samples_per_image

    def load_raw(self, index):
        """Снимок, маска экземпляров и номера ненадёжных клеток."""
        stem, image_path, mask_path = self.pairs[index % len(self.pairs)]
        image = load_image(image_path)
        labels, _ = mask_to_instances(load_mask(mask_path), min_area=GT_MIN_AREA)
        ignore_ids = find_suspected_merges(labels)
        return stem, image, labels, ignore_ids

    def load(self, index):
        """Снимок и целевые карты для одного снимка целиком (без синтетики — для валидации)."""
        stem, image, labels, ignore_ids = self.load_raw(index)
        fg, dist, valid = make_targets(labels, ignore_ids)
        return stem, image, labels, fg, dist, valid

    def use_synthetic(self):
        if not self.train or self.synthetic is None:
            return False
        if not self.synthetic.get("enabled", False):
            return False
        # Случайность берём из torch: у каждого процесса-загрузчика свой seed
        return float(torch.rand(1)) < self.synthetic["prob"]

    def __getitem__(self, index):
        stem, image, labels, ignore_ids = self.load_raw(index)

        if self.use_synthetic():
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            image, labels, ignore_ids = add_synthetic_clumps(
                image, labels, ignore_ids, rng,
                min_cells=self.synthetic["min_cells"],
                max_cells=self.synthetic["max_cells"],
            )

        # Цели считаем на ЦЕЛОМ снимке и только потом режем кроп:
        # иначе у клеток, обрезанных краем кропа, карта расстояний была бы неправильной
        fg, dist, valid = make_targets(labels, ignore_ids)

        if self.augment is not None:
            out = self.augment(image=image, masks=[fg, dist, valid])
            image = out["image"]
            fg, dist, valid = out["masks"]

        sample = {
            "image": normalize_image(image),
            "fg": torch.from_numpy(np.asarray(fg, dtype=np.float32))[None],
            "dist": torch.from_numpy(np.asarray(dist, dtype=np.float32))[None],
            "valid": torch.from_numpy(np.asarray(valid, dtype=np.float32))[None],
        }
        if not self.train:
            sample["stem"] = stem
        return sample
