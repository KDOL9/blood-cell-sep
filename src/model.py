"""Модель: U-Net с двумя выходными каналами.

Канал 0 — логиты маски «клетка/фон» (после sigmoid — вероятность клетки).
Канал 1 — логиты карты расстояний (после sigmoid — значение от 0 до 1).

U-Net = энкодер (сжимает снимок и извлекает признаки) + декодер (разворачивает
обратно до исходного размера) + skip-соединения (передают мелкие детали —
например, тонкие границы между клетками — из энкодера прямо в декодер).
"""
import segmentation_models_pytorch as smp
import torch


def build_model(encoder="resnet34", pretrained=True):
    weights = None
    if pretrained:
        weights = "imagenet"
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=2,
    )


def save_checkpoint(path, model, config, epoch, metrics):
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"]["encoder"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, config
