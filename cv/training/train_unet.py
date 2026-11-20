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


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    batch_reduction: bool = False,
) -> torch.Tensor:
    """Soft Dice loss, with smoothing so empty masks remain finite.

    When ``batch_reduction=True``, intersection and denominator are summed
    across the entire mini-batch rather than computed independently per sample,
    preventing near-empty masks from producing noisy per-image gradients.
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.to(dtype=logits.dtype)
    if logits.shape != targets.shape:
        raise ValueError(f"Logit and target shapes differ: {logits.shape} vs {targets.shape}")
    probs = torch.sigmoid(logits)
    dims = tuple(range(logits.ndim)) if batch_reduction else tuple(range(2, logits.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - score.mean()


def combined_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    batch_dice: bool = False,
) -> torch.Tensor:
    """Weighted combination of BCE-with-logits and soft Dice loss."""
    if bce_weight < 0.0 or dice_weight < 0.0 or (bce_weight + dice_weight) <= 0.0:
        raise ValueError("bce_weight and dice_weight must be non-negative and have positive sum")
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.to(dtype=logits.dtype)
    total = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if bce_weight > 0.0:
        total = total + bce_weight * F.binary_cross_entropy_with_logits(logits, targets)
    if dice_weight > 0.0:
        total = total + dice_weight * dice_loss(logits, targets, batch_reduction=batch_dice)
    return total


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


def _run_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    batch_dice: bool = False,
    epoch: int = 0,
    total_epochs: int = 0,
    progress_every: int = 0,
    progress_file: Path | None = None,
    progress_callback=None,
    best_dice: float = -1.0,
    best_epoch: int = 0,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    val_probs, val_targets = [], []
    total_batches = len(loader)
    t_epoch0 = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step_idx, (images, masks) in enumerate(loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = combined_segmentation_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                batch_dice=batch_dice,
            )
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_examples += batch_size
            if training and progress_every > 0 and (step_idx % progress_every == 0 or step_idx == total_batches):
                running_loss = total_loss / max(total_examples, 1)
                elapsed = time.perf_counter() - t_epoch0
                pct = 100.0 * step_idx / max(total_batches, 1)
                print(
                    f"  [Epoch {epoch:02d}/{total_epochs}] batch {step_idx:03d}/{total_batches:03d} "
                    f"({pct:5.1f}%) | running loss {running_loss:.4f} | {elapsed:.1f}s",
                    flush=True,
                )
                if progress_file is not None:
                    progress_file.write_text(
                        json.dumps({
                            "status": "training",
                            "epoch": epoch,
                            "total_epochs": total_epochs,
                            "batch": step_idx,
                            "total_batches": total_batches,
                            "epoch_percent": round(pct, 1),
                            "running_train_loss": round(running_loss, 5),
                            "epoch_elapsed_seconds": round(elapsed, 1),
                            "best_validation_dice_so_far": round(best_dice, 5) if best_dice >= 0 else None,
                            "best_epoch_so_far": best_epoch if best_epoch > 0 else None,
                        }, indent=2),
                        encoding="utf-8",
                    )
                if progress_callback is not None:
                    progress_callback()
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


def _save_validation_predictions(
    model: UNet,
    dataset: FootpathDamageDataset,
    output: Path,
    count: int = 4,
    threshold: float = 0.5,
) -> None:
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
            predicted = (torch.sigmoid(logits[0, 0]) >= threshold).cpu().numpy().astype(np.uint8)
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


def _read_json_safe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_safe(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except Exception:
        return []


def update_progress_markdown(extra_paths: list[Path] | None = None) -> None:
    """Render live Markdown experiment progress tracker to artifacts and optional extra paths."""
    exps = [
        ("unet_baseline", "128×128", "Original reference (batch=2, 15 ep, 0.5 BCE + 0.5 Dice, H-flip)", ROOT / "artifacts" / "unet_baseline"),
        ("exp1_res256", "256×256", "Higher resolution (256×256), nearest-neighbour mask resize, disk cache", ROOT / "artifacts" / "unet_experiments" / "exp1_res256"),
        ("exp2_train_improvements", "256×256", "batch=4, ±10° rotation + H-flip, ReduceLROnPlateau, early stopping (patience=5)", ROOT / "artifacts" / "unet_experiments" / "exp2_train_improvements"),
        ("exp3a_dice_focused", "256×256", "Exp 2 setup + Dice-focused loss (0.20 BCE + 0.80 batch Dice)", ROOT / "artifacts" / "unet_experiments" / "exp3a_dice_focused"),
        ("exp3b_modest_weight", "256×256", "Exp 2 setup + Modestly weighted loss (0.35 BCE + 0.65 batch Dice)", ROOT / "artifacts" / "unet_experiments" / "exp3b_modest_weight"),
    ]
    lines = [
        "# PathWise — Phase 1 Live Experiment Progress Tracker",
        "",
        f"*Last auto-updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`*",
        "",
        "## 1. Experiment Summary Table",
        "",
        "| Experiment ID | Status | Resolution | Configuration Highlights | Best Epoch | Val Dice (`t=0.50`) | Val IoU (`t=0.50`) | Selected `t*` | Val Dice (`t*`) | Val IoU (`t*`) | Runtime |",
        "|---|---|---|---|:---:|---:|---:|:---:|---:|---:|---:|",
    ]
    active_section: list[str] = []
    for name, res, desc, folder in exps:
        summary = _read_json_safe(folder / "training_summary.json")
        progress = _read_json_safe(folder / "progress.json")
        sweep = _read_json_safe(folder / "threshold_sweep.json")
        history = _read_csv_safe(folder / "history.csv")
        if summary is not None:
            status = "✅ Completed"
            best_ep = f"{summary.get('best_epoch')} / {summary.get('last_epoch', summary.get('training_config', {}).get('epochs', 15))}"
            d05 = f"`{summary['best_validation_dice']:.4f}`"
            iou05 = f"`{summary['best_validation_iou']:.4f}`"
            if "selected_validation_threshold" in summary:
                t_star = f"`{summary['selected_validation_threshold']:.2f}`"
                dt = f"`{summary['selected_threshold_validation_metrics']['dice']:.4f}`"
                iout = f"`{summary['selected_threshold_validation_metrics']['iou']:.4f}`"
            elif sweep is not None:
                t_star = f"`{sweep['selected_threshold']:.2f}`"
                dt = f"`{sweep['selected_metrics']['dice']:.4f}`"
                iout = f"`{sweep['selected_metrics']['iou']:.4f}`"
            else:
                t_star = "`0.40`" if name == "unet_baseline" else "—"
                dt = "`0.4345`" if name == "unet_baseline" else "—"
                iout = "`0.2775`" if name == "unet_baseline" else "—"
            secs = float(summary.get("total_training_seconds", 0.0))
            runtime = f"`{int(secs // 60)}m {int(secs % 60):02d}s` (`{secs:.1f}s`)"
        elif progress is not None or history:
            cur_ep = progress.get("epoch", len(history)) if progress else len(history)
            tot_ep = progress.get("total_epochs", 18) if progress else 18
            pct = progress.get("epoch_percent", 0.0) if progress else 0.0
            status = f"🔄 Ep {cur_ep}/{tot_ep} ({pct:.0f}%)"
            if history:
                best_row = max(history, key=lambda r: float(r["validation_dice"]))
                best_ep = f"{best_row['epoch']} (so far)"
                d05 = f"`{float(best_row['validation_dice']):.4f}`"
                iou05 = f"`{float(best_row['validation_iou']):.4f}`"
                elapsed = sum(float(r.get("epoch_seconds", 0.0)) for r in history)
                runtime = f"`{int(elapsed // 60)}m {int(elapsed % 60):02d}s` elapsed"
            else:
                best_ep, d05, iou05, runtime = "—", "—", "—", "starting..."
            t_star, dt, iout = "—", "—", "—"
            active_section.append(f"## 2. Live Telemetry: `{name}`")
            active_section.append("")
            if progress:
                bar_len = 20
                filled = int(round(bar_len * (progress.get("epoch_percent", 0.0) / 100.0)))
                bar = "█" * filled + "░" * (bar_len - filled)
                active_section.append(
                    f"- **Current Epoch:** `{progress.get('epoch')} / {progress.get('total_epochs')}`  \n"
                    f"- **Current Batch:** `{progress.get('batch')} / {progress.get('total_batches')}` "
                    f"`[{bar}] {progress.get('epoch_percent', 0.0):.1f}%`  \n"
                    f"- **Running Train Loss (current epoch):** `{progress.get('running_train_loss', 0.0):.4f}`  \n"
                    f"- **Current Epoch Elapsed:** `{progress.get('epoch_elapsed_seconds', 0.0):.1f}s`  \n"
                    f"- **Best Validation Dice So Far:** `{progress.get('best_validation_dice_so_far')}` (Epoch `{progress.get('best_epoch_so_far')}`)"
                )
                active_section.append("")
            if history:
                active_section.append("### Completed Epochs in Current Run")
                active_section.append("")
                active_section.append("| Epoch | LR | Train Loss | Val Loss | Val Dice (`t=0.50`) | Val IoU (`t=0.50`) | Epoch Time |")
                active_section.append("|---:|---:|---:|---:|---:|---:|---:|")
                for r in history:
                    lr_val = f"{float(r['learning_rate']):.5f}" if "learning_rate" in r else "0.00100"
                    active_section.append(
                        f"| {r['epoch']} | `{lr_val}` | `{float(r['train_loss']):.4f}` | "
                        f"`{float(r['validation_loss']):.4f}` | **`{float(r['validation_dice']):.4f}`** | "
                        f"`{float(r['validation_iou']):.4f}` | `{float(r['epoch_seconds']):.1f}s` |"
                    )
                active_section.append("")
        else:
            status, best_ep, d05, iou05, t_star, dt, iout, runtime = "⏳ Queued", "—", "—", "—", "—", "—", "—", "—"
        lines.append(f"| **`{name}`** | {status} | `{res}` | {desc} | {best_ep} | {d05} | {iou05} | {t_star} | {dt} | {iout} | {runtime} |")
    lines.append("")
    if active_section:
        lines.extend(active_section)
    else:
        lines.extend(["## 2. Live Telemetry", "", "*No experiment is currently mid-epoch.*", ""])
    content = "\n".join(lines)
    targets = [ROOT / "artifacts" / "unet_experiments" / "EXPERIMENT_PROGRESS.md"]
    if extra_paths:
        targets.extend(extra_paths)
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except Exception:
            pass


def _collect_probabilities(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    batch_dice: bool = False,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    losses, probs_all, targets_all, count = 0.0, [], [], 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = combined_segmentation_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                batch_dice=batch_dice,
            )
            batch = images.shape[0]
            losses += float(loss.item()) * batch
            count += batch
            probs_all.append(torch.sigmoid(logits).cpu())
            targets_all.append(masks.unsqueeze(1).cpu())
    return losses / max(count, 1), torch.cat(probs_all), torch.cat(targets_all)


def _metrics_at_threshold(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    return {
        "dice": dice_coefficient(probs, targets, threshold),
        "iou": intersection_over_union(probs, targets, threshold),
        "precision": precision(probs, targets, threshold),
        "recall": recall(probs, targets, threshold),
        "f1": f1_score(probs, targets, threshold),
    }


def sweep_validation_thresholds(
    probs: torch.Tensor,
    targets: torch.Tensor,
    thresholds: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
) -> dict:
    """Evaluate candidate probability thresholds on validation predictions and choose by Dice."""
    results = []
    for th in thresholds:
        m = _metrics_at_threshold(probs, targets, float(th))
        results.append({"threshold": round(float(th), 4), **m})
    best = max(results, key=lambda item: (item["dice"], -abs(item["threshold"] - 0.5)))
    fixed_05 = next((item for item in results if abs(item["threshold"] - 0.5) < 1e-6), _metrics_at_threshold(probs, targets, 0.5))
    return {
        "selected_threshold": best["threshold"],
        "selected_metrics": best,
        "fixed_0.5_metrics": fixed_05,
        "candidates": results,
    }


def _collect_metrics(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    batch_dice: bool = False,
) -> tuple[float, dict[str, float]]:
    mean_loss, probs, targets = _collect_probabilities(
        model,
        loader,
        device,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        batch_dice=batch_dice,
    )
    return mean_loss, _metrics_at_threshold(probs, targets, threshold)


def _evaluate_checkpoint(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    model = UNet(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = checkpoint["training_config"]
    dataset = FootpathDamageDataset(
        args.dataset,
        "test",
        size=(config["input_size"], config["input_size"]),
        manifest=args.manifest,
        cache_dir=args.cache_dir,
    )
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
    loss_desc = (
        f"{args.bce_weight:.2f} * BCEWithLogitsLoss + {args.dice_weight:.2f} * "
        f"{'batch ' if args.batch_dice else ''}soft Dice loss"
    )
    training_config = {
        "experiment_name": args.experiment_name or args.output.name,
        "device": "cpu",
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": loss_desc,
        "bce_weight": args.bce_weight,
        "dice_weight": args.dice_weight,
        "batch_dice": args.batch_dice,
        "lr_scheduler": "ReduceLROnPlateau" if args.lr_scheduler else "none",
        "lr_patience": args.lr_patience if args.lr_scheduler else None,
        "lr_factor": args.lr_factor if args.lr_scheduler else None,
        "early_stopping_patience": args.early_stopping_patience,
        "metric_threshold": 0.5,
        "num_workers": args.num_workers,
        "random_seed": args.seed,
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.5 if args.vertical_flip else 0.0,
        "max_rotation_degrees": args.rotation_degrees,
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
    }
    model_config = {"in_channels": 3, "out_channels": 1, "base_channels": args.base_channels}
    train_ds = FootpathDamageDataset(
        args.dataset,
        "train",
        horizontal_flip=True,
        vertical_flip=args.vertical_flip,
        max_rotation_degrees=args.rotation_degrees,
        size=(args.input_size, args.input_size),
        manifest=args.manifest,
        cache_dir=args.cache_dir,
    )
    val_ds = FootpathDamageDataset(
        args.dataset,
        "validation",
        size=(args.input_size, args.input_size),
        manifest=args.manifest,
        cache_dir=args.cache_dir,
    )
    train_loader = _make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True, generator=generator)
    val_loader = _make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    model = UNet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=1e-5,
        )
        if args.lr_scheduler
        else None
    )
    history: list[dict] = []
    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    start_epoch = 1
    prior_elapsed = 0.0
    last_ckpt_path = args.output / "last_checkpoint.pth"
    if args.resume and last_ckpt_path.is_file() and (args.output / "history.json").is_file():
        resume_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if scheduler is not None and resume_ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        history = json.loads((args.output / "history.json").read_text(encoding="utf-8"))
        best_dice = float(resume_ckpt["best_validation_dice"])
        best_epoch = int(resume_ckpt["best_epoch"])
        epochs_without_improvement = int(resume_ckpt.get("epochs_without_improvement", 0))
        start_epoch = int(resume_ckpt["epoch"]) + 1
        prior_elapsed = sum(float(x.get("epoch_seconds", 0.0)) for x in history)
        print(f"Resuming from epoch {start_epoch} (best Dice so far: {best_dice:.4f} at epoch {best_epoch})", flush=True)
    started = time.perf_counter()

    total_batches = len(train_loader)
    progress_every = max(1, total_batches // 4)
    extra_md = [args.progress_markdown] if args.progress_markdown else None
    print(f"Training on CPU: {len(train_ds)} train, {len(val_ds)} validation samples; {args.epochs} epochs")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_started = time.perf_counter()
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, _, _ = _run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                batch_dice=args.batch_dice,
                epoch=epoch,
                total_epochs=args.epochs,
                progress_every=progress_every,
                progress_file=args.output / "progress.json",
                progress_callback=lambda: update_progress_markdown(extra_md),
                best_dice=best_dice,
                best_epoch=best_epoch,
            )
            val_loss, val_dice, val_iou = _run_epoch(
                model,
                val_loader,
                None,
                device,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                batch_dice=args.batch_dice,
            )
            record = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_dice": val_dice,
                "validation_iou": val_iou,
            }
            if val_dice > best_dice:
                best_dice, best_epoch = val_dice, epoch
                epochs_without_improvement = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_dice": best_dice,
                    "epoch": epoch,
                    "model_config": model_config,
                    "training_config": training_config,
                    "random_seed": args.seed,
                }, args.output / "best_model.pth")
            else:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step(val_dice)
            record["epoch_seconds"] = time.perf_counter() - epoch_started
            history.append(record)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_validation_dice": best_dice,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch": epoch,
                "model_config": model_config,
                "training_config": training_config,
                "random_seed": args.seed,
            }, last_ckpt_path)
            _save_history(args.output, history)
            update_progress_markdown(extra_md)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | lr {current_lr:.5f} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | Dice {val_dice:.4f} | IoU {val_iou:.4f} | "
                f"{record['epoch_seconds']:.1f}s",
                flush=True,
            )
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                stopped_early = True
                print(
                    f"Early stopping triggered at epoch {epoch} (no validation Dice improvement for "
                    f"{args.early_stopping_patience} epochs; best epoch {best_epoch}, Dice {best_dice:.4f}).",
                    flush=True,
                )
                break

        checkpoint = torch.load(args.output / "best_model.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_checkpoint_val_loss, val_probs, val_targets = _collect_probabilities(
            model,
            val_loader,
            device,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            batch_dice=args.batch_dice,
        )
        best_checkpoint_metrics = _metrics_at_threshold(val_probs, val_targets, 0.5)
        threshold_sweep = sweep_validation_thresholds(val_probs, val_targets)
        (args.output / "threshold_sweep.json").write_text(json.dumps(threshold_sweep, indent=2), encoding="utf-8")
        _save_validation_predictions(model, val_ds, args.output / "samples", threshold=0.5)
        total_seconds = prior_elapsed + (time.perf_counter() - started)
        if last_ckpt_path.is_file():
            last_ckpt_path.unlink()
        last_epoch = history[-1]
        summary = {
            "experiment_name": training_config["experiment_name"],
            "best_epoch": best_epoch,
            "best_validation_dice": best_dice,
            "best_validation_iou": next(x["validation_iou"] for x in history if x["epoch"] == best_epoch),
            "best_checkpoint_validation_loss": best_checkpoint_val_loss,
            "best_checkpoint_validation_metrics": best_checkpoint_metrics,
            "selected_validation_threshold": threshold_sweep["selected_threshold"],
            "selected_threshold_validation_metrics": threshold_sweep["selected_metrics"],
            "last_epoch": last_epoch["epoch"],
            "stopped_early": stopped_early,
            "last_epoch_validation_loss": last_epoch["validation_loss"],
            "last_epoch_validation_dice": last_epoch["validation_dice"],
            "last_epoch_validation_iou": last_epoch["validation_iou"],
            "total_training_seconds": total_seconds,
            "average_epoch_seconds": float(np.mean([x["epoch_seconds"] for x in history])),
            "training_config": training_config,
        }
        (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.output / "progress.json").write_text(
            json.dumps({
                "status": "completed",
                "best_epoch": best_epoch,
                "best_validation_dice": round(best_dice, 5),
                "selected_validation_threshold": threshold_sweep["selected_threshold"],
                "selected_threshold_validation_dice": round(threshold_sweep["selected_metrics"]["dice"], 5),
                "total_training_seconds": round(total_seconds, 1),
            }, indent=2),
            encoding="utf-8",
        )
        update_progress_markdown(extra_md)
        print("Training complete:", json.dumps(summary, indent=2))
    finally:
        train_ds.close()
        val_ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Datasets" / "Indian Footpath Damage Segmentation Dataset.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "cv" / "data" / "experiment_manifest.csv")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional directory to cache resized PNG pairs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "unet_baseline")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--batch-dice", action="store_true", help="Aggregate soft Dice over the mini-batch")
    parser.add_argument("--vertical-flip", action="store_true", help="Enable random vertical flip augmentation")
    parser.add_argument("--rotation-degrees", type=float, default=0.0, help="Max random rotation angle in degrees")
    parser.add_argument("--lr-scheduler", action="store_true", help="Enable ReduceLROnPlateau on validation Dice")
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after N epochs without val Dice improvement (0 disables)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pth if interrupted")
    parser.add_argument("--progress-markdown", type=Path, default=None, help="Optional extra Markdown file to auto-update with live progress")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Evaluate a saved checkpoint on the test split instead of training")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    if options.evaluate_checkpoint:
        _evaluate_checkpoint(options)
    else:
        train(options)
