"""Train a compact CPU U-Net baseline or evaluate its checkpoint on test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.data.dataset import FootpathDamageDataset  # noqa: E402
from cv.evaluation.metrics import (  # noqa: E402
    dice_coefficient,
    f1_score,
    intersection_over_union,
    precision,
    recall,
)
from cv.models.unet import UNet  # noqa: E402


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss, with smoothing so empty masks remain finite."""
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.to(dtype=logits.dtype)
    if logits.shape != targets.shape:
        raise ValueError(f"Logit and target shapes differ: {logits.shape} vs {targets.shape}")
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, logits.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - score.mean()


def combined_segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Equal-weight BCE-with-logits and soft Dice loss."""
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return 0.5 * (bce + dice_loss(logits, targets))


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy, PyTorch and the shuffled training sampler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Initialize Python and NumPy RNGs from each PyTorch worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_loader(dataset: FootpathDamageDataset, batch_size: int, workers: int, *, shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def _run_epoch(model: UNet, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    val_probs, val_targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = combined_segmentation_loss(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_examples += batch_size
            if not training:
                val_probs.append(torch.sigmoid(logits).detach().cpu())
                val_targets.append(masks.detach().cpu().unsqueeze(1))
    mean_loss = total_loss / max(total_examples, 1)
    if training:
        return mean_loss, float("nan"), float("nan")
    probabilities = torch.cat(val_probs, dim=0)
    targets = torch.cat(val_targets, dim=0)
    return mean_loss, dice_coefficient(probabilities, targets), intersection_over_union(probabilities, targets)


def _draw_plot(path: Path, history: list[dict]) -> None:
    """Draw compact loss and validation-metric plots using Pillow only."""
    width, height = 1200, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def panel(box, title: str, series: list[tuple[str, str, list[float]]]) -> None:
        x0, y0, x1, y1 = box
        draw.text((x0, y0 - 24), title, fill="black")
        draw.rectangle(box, outline=(60, 60, 60), width=1)
        values = [v for _, _, seq in series for v in seq if np.isfinite(v)]
        if not values:
            return
        low, high = min(values), max(values)
        if high - low < 1e-8:
            low -= 0.05
            high += 0.05
        else:
            pad = (high - low) * 0.08
            low -= pad
            high += pad
        for i in range(1, 5):
            y = y1 - int((y1 - y0) * i / 5)
            draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
        epochs = len(history)
        for label, color, seq in series:
            points = []
            for i, value in enumerate(seq):
                x = x0 + int((x1 - x0) * i / max(epochs - 1, 1))
                y = y1 - int((value - low) / (high - low) * (y1 - y0))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text((x0 + 8 + series.index((label, color, seq)) * 135, y0 + 6), label, fill=color)
        draw.text((x0, y1 + 5), "Epoch 1", fill="black")
        draw.text((x1 - 55, y1 + 5), f"{epochs}", fill="black")

    panel((75, 75, 565, 430), "Training and validation loss", [
        ("train", (30, 100, 200), [x["train_loss"] for x in history]),
        ("validation", (220, 70, 50), [x["validation_loss"] for x in history]),
    ])
    panel((635, 75, 1125, 430), "Validation metrics", [
        ("Dice", (20, 140, 70), [x["validation_dice"] for x in history]),
        ("IoU", (140, 70, 180), [x["validation_iou"] for x in history]),
    ])
    image.save(path)


def _save_validation_predictions(model: UNet, dataset: FootpathDamageDataset, output: Path, count: int = 4) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    tile_w, tile_h, label_h = 256, 256, 28
    chosen = min(count, len(dataset))
    sheet = Image.new("RGB", (tile_w * 4, (tile_h + label_h) * chosen), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ("Original image", "Ground truth", "Prediction", "Prediction overlay")
    with torch.no_grad():
        for row in range(chosen):
            image_tensor, mask_tensor = dataset[row]
            logits = model(image_tensor.unsqueeze(0))
            predicted = (torch.sigmoid(logits[0, 0]) >= 0.5).cpu().numpy().astype(np.uint8)
            rgb = np.clip(image_tensor.permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
            original = Image.fromarray(rgb, "RGB").resize((tile_w, tile_h), Image.Resampling.BILINEAR)
            gt = Image.fromarray((mask_tensor.numpy().astype(np.uint8) * 255), "L").resize((tile_w, tile_h), Image.Resampling.NEAREST).convert("RGB")
            pred = Image.fromarray(predicted * 255, "L").resize((tile_w, tile_h), Image.Resampling.NEAREST).convert("RGB")
            overlay_arr = np.asarray(original).copy()
            pred_small = Image.fromarray(predicted * 255, "L").resize((tile_w, tile_h), Image.Resampling.NEAREST)
            active = np.asarray(pred_small) > 0
            overlay_arr[active] = (0.55 * overlay_arr[active] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
            panels = (original, gt, pred, Image.fromarray(overlay_arr, "RGB"))
            for col, (label, panel_image) in enumerate(zip(labels, panels)):
                x, y = col * tile_w, row * (tile_h + label_h)
                draw.text((x + 5, y + 6), label, fill="black")
                sheet.paste(panel_image, (x, y + label_h))
    sheet.save(output / "validation_predictions.png")


def _save_history(output: Path, history: list[dict]) -> None:
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _draw_plot(output / "training_curves.png", history)


def _collect_metrics(model: UNet, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> tuple[float, dict[str, float]]:
    model.eval()
    losses, probs_all, targets_all, count = 0.0, [], [], 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = combined_segmentation_loss(logits, masks)
            batch = images.shape[0]
            losses += float(loss.item()) * batch
            count += batch
            probs_all.append(torch.sigmoid(logits).cpu())
            targets_all.append(masks.unsqueeze(1).cpu())
    probs, targets = torch.cat(probs_all), torch.cat(targets_all)
    metrics = {
        "dice": dice_coefficient(probs, targets, threshold),
        "iou": intersection_over_union(probs, targets, threshold),
        "precision": precision(probs, targets, threshold),
        "recall": recall(probs, targets, threshold),
        "f1": f1_score(probs, targets, threshold),
    }
    return losses / max(count, 1), metrics


def _evaluate_checkpoint(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    model = UNet(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = checkpoint["training_config"]
    dataset = FootpathDamageDataset(args.dataset, "test", size=(config["input_size"], config["input_size"]), manifest=args.manifest)
    loader = _make_loader(dataset, config["batch_size"], args.num_workers, shuffle=False)
    loss, metrics = _collect_metrics(model, loader, torch.device("cpu"), config["metric_threshold"])
    result = {"split": "test", "loss": loss, **metrics, "checkpoint": str(args.evaluate_checkpoint)}
    output_path = Path(args.evaluate_checkpoint).parent / "test_metrics.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Test metrics:", json.dumps(result, indent=2))
    dataset.close()


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.input_size < 16:
        raise ValueError("Epochs, batch size and input size must be positive; workers cannot be negative")
    args.output.mkdir(parents=True, exist_ok=True)
    generator = seed_everything(args.seed)
    device = torch.device("cpu")
    training_config = {
        "device": "cpu",
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "loss": "0.5 * (BCEWithLogitsLoss + soft Dice loss)",
        "metric_threshold": 0.5,
        "num_workers": args.num_workers,
        "random_seed": args.seed,
        "horizontal_flip_probability": 0.5,
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
    }
    model_config = {"in_channels": 3, "out_channels": 1, "base_channels": args.base_channels}
    train_ds = FootpathDamageDataset(args.dataset, "train", horizontal_flip=True, size=(args.input_size, args.input_size), manifest=args.manifest)
    val_ds = FootpathDamageDataset(args.dataset, "validation", size=(args.input_size, args.input_size), manifest=args.manifest)
    train_loader = _make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True, generator=generator)
    val_loader = _make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    model = UNet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict] = []
    best_dice = -1.0
    best_epoch = 0
    started = time.perf_counter()

    print(f"Training on CPU: {len(train_ds)} train, {len(val_ds)} validation samples; {args.epochs} epochs")
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, _, _ = _run_epoch(model, train_loader, optimizer, device)
            val_loss, val_dice, val_iou = _run_epoch(model, val_loader, None, device)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_dice": val_dice,
                "validation_iou": val_iou,
            }
            if val_dice > best_dice:
                best_dice, best_epoch = val_dice, epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_dice": best_dice,
                    "epoch": epoch,
                    "model_config": model_config,
                    "training_config": training_config,
                    "random_seed": args.seed,
                }, args.output / "best_model.pth")
            record["epoch_seconds"] = time.perf_counter() - epoch_started
            history.append(record)
            _save_history(args.output, history)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | Dice {val_dice:.4f} | IoU {val_iou:.4f} | "
                f"{record['epoch_seconds']:.1f}s",
                flush=True,
            )

        checkpoint = torch.load(args.output / "best_model.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_checkpoint_val_loss, best_checkpoint_metrics = _collect_metrics(model, val_loader, device)
        _save_validation_predictions(model, val_ds, args.output / "samples")
        total_seconds = time.perf_counter() - started
        last_epoch = history[-1]
        summary = {
            "best_epoch": best_epoch,
            "best_validation_dice": best_dice,
            "best_validation_iou": next(x["validation_iou"] for x in history if x["epoch"] == best_epoch),
            "best_checkpoint_validation_loss": best_checkpoint_val_loss,
            "best_checkpoint_validation_metrics": best_checkpoint_metrics,
            "last_epoch": last_epoch["epoch"],
            "last_epoch_validation_loss": last_epoch["validation_loss"],
            "last_epoch_validation_dice": last_epoch["validation_dice"],
            "last_epoch_validation_iou": last_epoch["validation_iou"],
            "total_training_seconds": total_seconds,
            "average_epoch_seconds": float(np.mean([x["epoch_seconds"] for x in history])),
            "training_config": training_config,
        }
        (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Training complete:", json.dumps(summary, indent=2))
    finally:
        train_ds.close()
        val_ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Datasets" / "Indian Footpath Damage Segmentation Dataset.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "cv" / "data" / "experiment_manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "unet_baseline")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Evaluate a saved checkpoint on the test split instead of training")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    if options.evaluate_checkpoint:
        _evaluate_checkpoint(options)
    else:
        train(options)
