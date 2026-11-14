"""Применение обученной модели к целому снимку."""
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import normalize_image
from src.postprocess import components, distance_watershed


@torch.no_grad()
def predict_maps(model, image, device):
    """Снимок (H, W, 3) uint8 -> (fg_prob, dist), обе карты (H, W) float32.

    U-Net пять раз уменьшает изображение вдвое, поэтому стороны должны делиться на 32.
    Дополняем снимок нулями справа и снизу, а результат обрезаем обратно.
    """
    height, width = image.shape[:2]
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32

    x = normalize_image(image)[None].to(device)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    output = torch.sigmoid(model(x))[0, :, :height, :width]
    output = output.float().cpu().numpy()
    return output[0], output[1]


def segment(model, image, device, postprocess="watershed", fg_threshold=0.5, seed_threshold=0.5):
    """Снимок -> маска экземпляров (каждая клетка со своим номером)."""
    fg_prob, dist = predict_maps(model, image, device)
    if postprocess == "components":
        return components(fg_prob, fg_threshold)
    return distance_watershed(fg_prob, dist, fg_threshold, seed_threshold)
