"""Final held-out test split evaluation and visual comparison for PathWise U-Net checkpoints."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.data.dataset import FootpathDamageDataset  # noqa: E402
from cv.evaluation.metrics import (  # noqa: E402
    dice_coefficient,
    f1_score,
    intersection_over_union,
    macro_dice_coefficient,
    macro_intersection_over_union,
    precision,
    recall,
)
from cv.models.unet import UNet  # noqa: E402

DATASET_ZIP = ROOT / "Datasets" / "Indian Footpath Damage Segmentation Dataset.zip"
MANIFEST_CSV = ROOT / "cv" / "data" / "experiment_manifest.csv"
CACHE_DIR = ROOT / "artifacts" / "dataset_cache"
OUTPUT_DIR = ROOT / "artifacts" / "final_evaluation"

CHECKPOINT_SPECS = [
    {
        "id": "unet_baseline",
        "label": "Baseline (128x128)",
        "checkpoint": ROOT / "artifacts" / "unet_baseline" / "best_model.pth",
        "selected_threshold": 0.40,
        "selected_threshold_source": "Baseline diagnosis validation sweep (docs/BASELINE_DIAGNOSIS.md & docs/MODEL_EXPERIMENTS.md)",
        "validation_dice_050": 0.432351,
        "validation_iou_050": 0.275796,
        "validation_dice_selected": 0.434517,
        "validation_iou_selected": 0.277559,
    },
    {
        "id": "exp2_train_improvements",
        "label": "Exp 2 (256x256, 0.50 BCE + 0.50 Dice)",
        "checkpoint": ROOT / "artifacts" / "unet_experiments" / "exp2_train_improvements" / "best_model.pth",
        "selected_threshold": 0.30,
        "selected_threshold_source": "artifacts/unet_experiments/exp2_train_improvements/threshold_sweep.json",
        "validation_dice_050": 0.477547,
        "validation_iou_050": 0.313669,
        "validation_dice_selected": 0.488830,
        "validation_iou_selected": 0.323478,
    },
    {
        "id": "exp3b_modest_weight",
        "label": "Exp 3b (256x256, 0.35 BCE + 0.65 Batch Dice)",
        "checkpoint": ROOT / "artifacts" / "unet_experiments" / "exp3b_modest_weight" / "best_model.pth",
        "selected_threshold": 0.40,
        "selected_threshold_source": "artifacts/unet_experiments/exp3b_modest_weight/threshold_sweep.json",
        "validation_dice_050": 0.484471,
        "validation_iou_050": 0.319671,
        "validation_dice_selected": 0.486557,
        "validation_iou_selected": 0.321490,
    },
]


def _load_model_and_test_predictions(spec: dict) -> tuple[dict, FootpathDamageDataset, torch.Tensor, torch.Tensor, list[str], float]:
    t0 = time.perf_counter()
    ckpt = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_config"]
    train_cfg = ckpt["training_config"]
    input_size = int(train_cfg["input_size"])
    model = UNet(**model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = FootpathDamageDataset(
        DATASET_ZIP,
        "test",
        size=(input_size, input_size),
        manifest=MANIFEST_CSV,
        cache_dir=CACHE_DIR,
    )
    if len(ds) != 82:
        raise RuntimeError(f"Expected 82 included test samples, found {len(ds)}")

    probs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    image_ids: list[str] = []

    with torch.no_grad():
        for idx in range(len(ds)):
            image_ref, _ = ds.pairs[idx]
            stem = PurePosixPath(str(image_ref).replace("\\", "/")).stem
            image_ids.append(stem)
            img, msk = ds[idx]
            logits = model(img.unsqueeze(0))
            prob = torch.sigmoid(logits[0, 0]).cpu()
            probs_list.append(prob)
            targets_list.append(msk.cpu())

    elapsed = time.perf_counter() - t0
    probs_tensor = torch.stack(probs_list, dim=0)    # (82, H, W)
    targets_tensor = torch.stack(targets_list, dim=0)  # (82, H, W)
    return ckpt, ds, probs_tensor, targets_tensor, image_ids, elapsed


def _evaluate_at_threshold(probs: torch.Tensor, targets: torch.Tensor, threshold: float) -> tuple[dict[str, float], list[dict[str, float]]]:
    micro_metrics = {
        "threshold": round(float(threshold), 4),
        "global_micro_dice": dice_coefficient(probs, targets, threshold),
        "macro_dice": macro_dice_coefficient(probs, targets, threshold),
        "global_micro_iou": intersection_over_union(probs, targets, threshold),
        "macro_iou": macro_intersection_over_union(probs, targets, threshold),
        "precision": precision(probs, targets, threshold),
        "recall": recall(probs, targets, threshold),
        "f1": f1_score(probs, targets, threshold),
    }
    per_image: list[dict[str, float]] = []
    for i in range(probs.shape[0]):
        p_i = probs[i]
        t_i = targets[i]
        total_px = int(t_i.numel())
        gt_pos = int((t_i >= 0.5).sum().item())
        pred_pos = int((p_i >= threshold).sum().item())
        per_image.append({
            "dice": dice_coefficient(p_i, t_i, threshold),
            "iou": intersection_over_union(p_i, t_i, threshold),
            "precision": precision(p_i, t_i, threshold),
            "recall": recall(p_i, t_i, threshold),
            "f1": f1_score(p_i, t_i, threshold),
            "gt_positive_pixels": gt_pos,
            "pred_positive_pixels": pred_pos,
            "total_pixels": total_px,
            "gt_damage_ratio": gt_pos / total_px,
            "pred_damage_ratio": pred_pos / total_px,
        })
    return micro_metrics, per_image


def _render_prediction_comparison(
    output_path: Path,
    sample_indices: list[int],
    ds_256: FootpathDamageDataset,
    preds_by_model: dict[str, dict],
) -> list[str]:
    """Render deterministic side-by-side comparison sheet across all 3 checkpoints."""
    tile_w, tile_h = 220, 220
    header_h = 52
    row_label_h = 26
    cols = [
        ("Original Image", "orig"),
        ("Ground Truth (256x256)", "gt"),
        ("Baseline (128x128)\nt = 0.50 (fixed)", "base_050"),
        ("Exp 2 (256x256)\nt = 0.50 (fixed)", "exp2_050"),
        ("Exp 3b (256x256)\nt = 0.50 (fixed)", "exp3b_050"),
        ("Baseline (128x128)\nt* = 0.40 (val-sel)", "base_sel"),
        ("Exp 2 (256x256)\nt* = 0.30 (val-sel)", "exp2_sel"),
        ("Exp 3b (256x256)\nt* = 0.40 (val-sel)", "exp3b_sel"),
    ]
    n_cols = len(cols)
    n_rows = len(sample_indices)
    sheet_w = n_cols * tile_w
    sheet_h = header_h + n_rows * (tile_h + row_label_h)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for c_idx, (col_title, _) in enumerate(cols):
        x0 = c_idx * tile_w
        draw.rectangle((x0, 0, x0 + tile_w - 1, header_h - 1), fill=(240, 244, 248), outline=(180, 185, 190))
        draw.multiline_text((x0 + 8, 8), col_title, fill=(15, 20, 25), spacing=4)

    selected_ids: list[str] = []

    def make_overlay(rgb_im: Image.Image, prob_2d: torch.Tensor, threshold: float) -> Image.Image:
        mask_arr = (prob_2d.numpy() >= threshold).astype(np.uint8) * 255
        mask_im = Image.fromarray(mask_arr, mode="L").resize((tile_w, tile_h), Image.Resampling.NEAREST)
        base_arr = np.asarray(rgb_im, dtype=np.float32).copy()
        active = np.asarray(mask_im) > 0
        base_arr[active] = 0.52 * base_arr[active] + 0.48 * np.array([245.0, 40.0, 40.0], dtype=np.float32)
        # Also draw binary mask inset in bottom-right corner (56x56) so both raw mask and overlay are clear
        inset_size = 58
        inset = mask_im.resize((inset_size, inset_size), Image.Resampling.NEAREST).convert("RGB")
        out_im = Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8), mode="RGB")
        out_draw = ImageDraw.Draw(out_im)
        ix0, iy0 = tile_w - inset_size - 4, tile_h - inset_size - 4
        out_draw.rectangle((ix0 - 1, iy0 - 1, ix0 + inset_size, iy0 + inset_size), outline=(255, 200, 0), width=1)
        out_im.paste(inset, (ix0, iy0))
        return out_im

    for r_idx, ds_idx in enumerate(sample_indices):
        image_ref, _ = ds_256.pairs[ds_idx]
        image_id = PurePosixPath(str(image_ref).replace("\\", "/")).stem
        selected_ids.append(image_id)
        img_tensor, mask_tensor = ds_256[ds_idx]
        rgb_np = np.clip(img_tensor.permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
        orig_tile = Image.fromarray(rgb_np, mode="RGB").resize((tile_w, tile_h), Image.Resampling.BILINEAR)

        gt_mask_np = (mask_tensor.numpy().astype(np.uint8)) * 255
        gt_mask_im = Image.fromarray(gt_mask_np, mode="L").resize((tile_w, tile_h), Image.Resampling.NEAREST)
        gt_arr = np.asarray(orig_tile, dtype=np.float32).copy()
        gt_active = np.asarray(gt_mask_im) > 0
        gt_arr[gt_active] = 0.52 * gt_arr[gt_active] + 0.48 * np.array([30.0, 200.0, 90.0], dtype=np.float32)
        gt_tile = Image.fromarray(np.clip(gt_arr, 0, 255).astype(np.uint8), mode="RGB")
        inset_size = 58
        gt_inset = gt_mask_im.resize((inset_size, inset_size), Image.Resampling.NEAREST).convert("RGB")
        gt_draw = ImageDraw.Draw(gt_tile)
        ix0, iy0 = tile_w - inset_size - 4, tile_h - inset_size - 4
        gt_draw.rectangle((ix0 - 1, iy0 - 1, ix0 + inset_size, iy0 + inset_size), outline=(30, 200, 90), width=1)
        gt_tile.paste(gt_inset, (ix0, iy0))

        base_p = preds_by_model["unet_baseline"]["probs"][ds_idx]
        exp2_p = preds_by_model["exp2_train_improvements"]["probs"][ds_idx]
        exp3b_p = preds_by_model["exp3b_modest_weight"]["probs"][ds_idx]

        tiles = [
            orig_tile,
            gt_tile,
            make_overlay(orig_tile, base_p, 0.50),
            make_overlay(orig_tile, exp2_p, 0.50),
            make_overlay(orig_tile, exp3b_p, 0.50),
            make_overlay(orig_tile, base_p, 0.40),
            make_overlay(orig_tile, exp2_p, 0.30),
            make_overlay(orig_tile, exp3b_p, 0.40),
        ]

        y0 = header_h + r_idx * (tile_h + row_label_h)
        gt_ratio = float((mask_tensor >= 0.5).float().mean().item()) * 100.0
        for c_idx, tile in enumerate(tiles):
            x0 = c_idx * tile_w
            draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + row_label_h - 1), fill=(250, 250, 250), outline=(210, 210, 210))
            if c_idx == 0:
                caption = f"Test #{ds_idx:02d} (ID {image_id})"
            elif c_idx == 1:
                caption = f"GT area: {gt_ratio:.1f}%"
            else:
                model_key = ["unet_baseline", "exp2_train_improvements", "exp3b_modest_weight"][(c_idx - 2) % 3]
                th_key = "0.50" if c_idx < 5 else "selected"
                d_val = preds_by_model[model_key]["per_image"][th_key][ds_idx]["dice"]
                iou_val = preds_by_model[model_key]["per_image"][th_key][ds_idx]["iou"]
                caption = f"Dice {d_val:.3f} | IoU {iou_val:.3f}"
            draw.text((x0 + 6, y0 + 6), caption, fill=(20, 20, 20))
            sheet.paste(tile, (x0, y0 + row_label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return selected_ids


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_records: list[dict] = []
    per_image_rows: list[dict] = []
    preds_by_model: dict[str, dict] = {}
    ds_256_ref: FootpathDamageDataset | None = None

    for spec in CHECKPOINT_SPECS:
        ckpt, ds, probs, targets, image_ids, eval_secs = _load_model_and_test_predictions(spec)
        if int(ckpt["training_config"]["input_size"]) == 256 and ds_256_ref is None:
            ds_256_ref = ds
        else:
            ds.close()

        metrics_050, per_img_050 = _evaluate_at_threshold(probs, targets, 0.50)
        sel_th = float(spec["selected_threshold"])
        metrics_sel, per_img_sel = _evaluate_at_threshold(probs, targets, sel_th)

        preds_by_model[spec["id"]] = {
            "probs": probs,
            "targets": targets,
            "image_ids": image_ids,
            "per_image": {"0.50": per_img_050, "selected": per_img_sel},
        }

        for mode, th_val, m_dict, p_list in [
            ("fixed_0.50", 0.50, metrics_050, per_img_050),
            ("validation_selected", sel_th, metrics_sel, per_img_sel),
        ]:
            val_dice = spec["validation_dice_050"] if mode == "fixed_0.50" else spec["validation_dice_selected"]
            val_iou = spec["validation_iou_050"] if mode == "fixed_0.50" else spec["validation_iou_selected"]
            comparison_records.append({
                "checkpoint_id": spec["id"],
                "label": spec["label"],
                "checkpoint_path": str(spec["checkpoint"].relative_to(ROOT)).replace("\\", "/"),
                "best_epoch": int(ckpt["epoch"]),
                "input_size": int(ckpt["training_config"]["input_size"]),
                "test_samples": len(image_ids),
                "threshold_mode": mode,
                "threshold": round(th_val, 4),
                "validation_dice": round(val_dice, 6),
                "validation_iou": round(val_iou, 6),
                "test_global_micro_dice": round(m_dict["global_micro_dice"], 6),
                "test_macro_dice": round(m_dict["macro_dice"], 6),
                "test_global_micro_iou": round(m_dict["global_micro_iou"], 6),
                "test_macro_iou": round(m_dict["macro_iou"], 6),
                "test_precision": round(m_dict["precision"], 6),
                "test_recall": round(m_dict["recall"], 6),
                "test_f1": round(m_dict["f1"], 6),
                "evaluation_seconds": round(eval_secs, 2),
            })
            for idx, img_id in enumerate(image_ids):
                per_image_rows.append({
                    "checkpoint_id": spec["id"],
                    "input_size": int(ckpt["training_config"]["input_size"]),
                    "threshold_mode": mode,
                    "threshold": round(th_val, 4),
                    "sample_index": idx,
                    "image_id": img_id,
                    "dice": round(p_list[idx]["dice"], 6),
                    "iou": round(p_list[idx]["iou"], 6),
                    "precision": round(p_list[idx]["precision"], 6),
                    "recall": round(p_list[idx]["recall"], 6),
                    "f1": round(p_list[idx]["f1"], 6),
                    "gt_positive_pixels": p_list[idx]["gt_positive_pixels"],
                    "pred_positive_pixels": p_list[idx]["pred_positive_pixels"],
                    "total_pixels": p_list[idx]["total_pixels"],
                    "gt_damage_ratio": round(p_list[idx]["gt_damage_ratio"], 6),
                    "pred_damage_ratio": round(p_list[idx]["pred_damage_ratio"], 6),
                })

    # Deterministically select 6 evenly spaced test indices across the 82 sorted test samples:
    # indices [0, 16, 32, 48, 64, 80]
    deterministic_indices = [0, 16, 32, 48, 64, 80]
    assert ds_256_ref is not None
    visualized_ids = _render_prediction_comparison(
        OUTPUT_DIR / "prediction_comparison.png",
        deterministic_indices,
        ds_256_ref,
        preds_by_model,
    )
    ds_256_ref.close()

    # Save checkpoint_comparison.csv
    with (OUTPUT_DIR / "checkpoint_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_records[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_records)

    # Save per_image_metrics.csv
    with (OUTPUT_DIR / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    # Save checkpoint_comparison.json
    json_payload = {
        "evaluation_date": "2026-11-20",
        "split": "test",
        "manifest": "cv/data/experiment_manifest.csv",
        "included_test_sample_count": 82,
        "quarantined_test_sample_count": 5,
        "empty_mask_handling": {
            "both_gt_and_prediction_empty": "Dice = 1.0, IoU = 1.0, Precision = 1.0, Recall = 1.0",
            "only_gt_empty_false_positive": "Dice = 0.0, IoU = 0.0, Precision = 0.0, Recall = 0.0",
            "only_prediction_empty_false_negative": "Dice = 0.0, IoU = 0.0, Precision = 0.0, Recall = 0.0",
        },
        "deterministic_visual_sample_indices": deterministic_indices,
        "deterministic_visual_image_ids": visualized_ids,
        "results": comparison_records,
    }
    (OUTPUT_DIR / "checkpoint_comparison.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(json.dumps(json_payload, indent=2))


if __name__ == "__main__":
    main()
