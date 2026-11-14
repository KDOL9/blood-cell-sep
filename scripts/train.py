"""Обучение U-Net.

Запуск из корня репозитория:
    python scripts/train.py --config configs/unet_dist.yaml
    python scripts/train.py --config configs/unet_dist.yaml --smoke   # быстрая проверка, что всё работает

Чекпоинты: checkpoints/<run_name>/best.pth и last.pth
Логи для TensorBoard: runs/<run_name>   (смотреть: tensorboard --logdir runs)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_io import find_suspected_merges, list_pairs
from src.dataset import CellDataset
from src.losses import compute_loss
from src.metrics import aggregate_stats, image_stats
from src.model import build_model, save_checkpoint
from src.predict import segment
from src.utils import get_device, set_seed

DATA_ROOT = ROOT / "data" / "raw" / "bbbc041seg"
SPLIT_PATH = ROOT / "configs" / "split_bbbc041seg.json"


def load_split_pairs(split_name):
    with open(SPLIT_PATH) as f:
        stems = set(json.load(f)[split_name])
    pairs = []
    for stem, image_path, mask_path in list_pairs(DATA_ROOT):
        if stem in stems:
            pairs.append((stem, image_path, mask_path))
    return pairs


def to_device(batch, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def train_one_epoch(model, loader, optimizer, scaler, device, dist_weight, use_amp, max_steps):
    model.train()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "mse": 0.0}
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        # Смешанная точность (float16) на видеокарте NVIDIA: быстрее и меньше памяти
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(batch["image"])
            loss, parts = compute_loss(output.float(), batch, dist_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += float(loss.detach())
        for key in parts:
            totals[key] += parts[key]
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    for key in totals:
        totals[key] /= max(1, steps)
    return totals


@torch.no_grad()
def validate(model, dataset, device, postprocess, max_images):
    """Сегментирует целые снимки val и считает те же метрики, что и на тесте."""
    model.eval()
    stats_list = []
    count = len(dataset.pairs)
    if max_images > 0:
        count = min(count, max_images)

    for index in tqdm(range(count), desc="val", leave=False):
        stem, image, labels, _, _, _ = dataset.load(index)
        pred = segment(model, image, device, postprocess=postprocess)
        stats_list.append(image_stats(labels, pred, find_suspected_merges(labels)))

    return aggregate_stats(stats_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="пара шагов обучения и пара снимков валидации — проверить, что всё запускается")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    use_amp = device.type == "cuda"
    print("Устройство:", device)

    run_name = config["run_name"]
    max_steps = 0
    max_val_images = 0
    epochs = config["train"]["epochs"]
    if args.smoke:
        run_name = run_name + "_smoke"
        max_steps = 3
        max_val_images = 2
        epochs = 1

    ckpt_dir = ROOT / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(ROOT / "runs" / run_name))

    train_set = CellDataset(load_split_pairs("train"), crop_size=config["train"]["crop_size"],
                            train=True, samples_per_image=config["train"]["samples_per_image"])
    val_set = CellDataset(load_split_pairs("val"), train=False)
    print("train снимков:", len(train_set.pairs), "| val снимков:", len(val_set.pairs))

    loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=use_amp,
        drop_last=True,
        persistent_workers=config["train"]["num_workers"] > 0,
    )

    model = build_model(config["model"]["encoder"], pretrained=config["model"]["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"],
                                  weight_decay=config["train"]["weight_decay"])
    # Косинусное расписание: скорость обучения плавно падает от lr почти до нуля к концу обучения
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    dist_weight = config["loss"]["dist_weight"]
    postprocess = config["postprocess"]
    val_every = config["train"]["val_every"]
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        losses = train_one_epoch(model, loader, optimizer, scaler, device, dist_weight, use_amp, max_steps)
        scheduler.step()

        for key, value in losses.items():
            writer.add_scalar("train/" + key, value, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        message = "эпоха " + str(epoch) + "/" + str(epochs) + " | loss " + str(round(losses["loss"], 4))

        is_val_epoch = epoch % val_every == 0 or epoch == epochs
        if is_val_epoch:
            metrics = validate(model, val_set, device, postprocess, max_val_images)
            for key in ["f1@0.5", "ap@[0.5:0.95]", "aji", "merge_rate", "split_rate", "count_error", "dice"]:
                writer.add_scalar("val/" + key, metrics[key], epoch)
            score = metrics["f1@0.5"]
            message += " | val F1@0.5 " + str(metrics["f1@0.5"]) + " | merge " + str(metrics["merge_rate"])

            if score > best_score:
                best_score = score
                save_checkpoint(ckpt_dir / "best.pth", model, config, epoch, metrics)
                message += "  ← лучшая"

        save_checkpoint(ckpt_dir / "last.pth", model, config, epoch, {})
        message += " | " + str(round(time.time() - start)) + " с"
        print(message)

    writer.close()
    print("Готово. Лучший val F1@0.5:", best_score, "| чекпоинт:", ckpt_dir / "best.pth")


if __name__ == "__main__":
    main()
