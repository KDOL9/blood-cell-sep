"""Функция потерь: насколько предсказание сети отличается от целевых карт.

Итоговый лосс = лосс маски + dist_weight * лосс карты расстояний.
Все слагаемые считаются только там, где valid = 1 (без ненадёжных областей и паддинга).
"""
import torch
import torch.nn.functional as F


def masked_bce(logits, target, valid):
    """Бинарная кросс-энтропия: штраф за каждый неверно классифицированный пиксель."""
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def masked_dice(logits, target, valid, eps=1.0):
    """Dice-лосс: 1 - перекрытие предсказанной и настоящей маски.

    BCE смотрит на пиксели по отдельности, Dice — на маску целиком.
    Вместе они обучаются устойчивее, чем каждый по отдельности.
    """
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    inter = (prob * target).sum()
    total = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (total + eps)


def masked_mse(logits, target, valid):
    """Среднеквадратичная ошибка карты расстояний."""
    pred = torch.sigmoid(logits)
    loss = (pred - target) ** 2
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def compute_loss(output, batch, dist_weight):
    """output: (B, 2, H, W). Возвращает общий лосс и словарь слагаемых для логов."""
    fg_logits = output[:, 0:1]
    dist_logits = output[:, 1:2]
    valid = batch["valid"]

    bce = masked_bce(fg_logits, batch["fg"], valid)
    dice = masked_dice(fg_logits, batch["fg"], valid)
    mse = masked_mse(dist_logits, batch["dist"], valid)

    total = bce + dice + dist_weight * mse
    parts = {
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "mse": float(mse.detach()),
    }
    return total, parts
