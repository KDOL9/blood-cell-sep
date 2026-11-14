"""Проверка окружения: запускать после установки на Mac и на ПК.

Запуск из корня репозитория:
    python scripts/check_env.py
"""
import sys
from pathlib import Path

# Добавляем корень репозитория в путь поиска модулей, чтобы работал import src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.utils import get_device, set_seed


def main():
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)

    device = get_device()
    print("Устройство:", device)

    if device.type == "cuda":
        print("Видеокарта:", torch.cuda.get_device_name(0))

    # Маленький тест: перемножаем две матрицы на выбранном устройстве
    set_seed(42)
    a = torch.rand(1000, 1000, device=device)
    b = torch.rand(1000, 1000, device=device)
    c = a @ b
    print("Тест матричного умножения пройден, размер результата:", tuple(c.shape))

    # Проверяем, что ставятся ключевые библиотеки
    import cv2
    import skimage
    import segmentation_models_pytorch as smp
    import albumentations as A

    print("OpenCV:", cv2.__version__)
    print("scikit-image:", skimage.__version__)
    print("segmentation-models-pytorch:", smp.__version__)
    print("albumentations:", A.__version__)
    print("Всё готово к работе.")


if __name__ == "__main__":
    main()
