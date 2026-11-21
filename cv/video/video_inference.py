"""U-Net segmentation inference and frame overlay rendering for video frames."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from cv.models.unet import UNet

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "unet_experiments"
    / "exp2_train_improvements"
    / "best_model.pth"
)
DEFAULT_THRESHOLD = 0.30


@dataclass(frozen=True)
class FramePrediction:
    """Result of running U-Net segmentation on a single video frame."""

    binary_mask: np.ndarray  # uint8 (H, W) with values 0 or 1 at original frame resolution
    damaged_pixel_count: int
    total_pixel_count: int
    damaged_percentage: float
    threshold: float
    inference_seconds: float


class VideoSegmentationModel:
    """Load a trained U-Net checkpoint once and run reproducible frame-by-frame inference.

    Reproduces the exact training/validation preprocessing from ``FootpathDamageDataset``:
    1. Convert OpenCV BGR frame to RGB ``PIL.Image``.
    2. Resize to ``(input_size, input_size)`` using ``Image.Resampling.BILINEAR``.
    3. Scale uint8 values to ``float32`` in ``[0, 1]`` by dividing by ``255.0``.
    4. Transpose to ``(1, 3, H, W)`` tensor, run ``model(tensor)``, apply ``torch.sigmoid``,
       threshold at ``threshold``, and resize the binary mask back to original frame
       dimensions using ``Image.Resampling.NEAREST`` (or ``cv2.INTER_NEAREST``).
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        *,
        threshold: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint is missing required model_config or model_state_dict: {self.checkpoint_path}")

        self.model_config: dict = dict(checkpoint["model_config"])
        self.training_config: dict = dict(checkpoint.get("training_config", {}))
        self.input_size: int = int(self.training_config.get("input_size", 256))
        self.checkpoint_epoch: int | None = checkpoint.get("epoch")

        if threshold is None:
            threshold = self._resolve_default_threshold()
        if not (0.0 < float(threshold) < 1.0) or not np.isfinite(float(threshold)):
            raise ValueError(f"Probability threshold must lie strictly between 0.0 and 1.0, got {threshold}")
        self.threshold = float(threshold)

        self.model = UNet(**self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _resolve_default_threshold(self) -> float:
        sweep_path = self.checkpoint_path.parent / "threshold_sweep.json"
        if sweep_path.is_file():
            try:
                data = json.loads(sweep_path.read_text(encoding="utf-8"))
                if "selected_threshold" in data:
                    return float(data["selected_threshold"])
            except Exception:
                pass
        summary_path = self.checkpoint_path.parent / "training_summary.json"
        if summary_path.is_file():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if "selected_validation_threshold" in data:
                    return float(data["selected_validation_threshold"])
            except Exception:
                pass
        return DEFAULT_THRESHOLD

    def preprocess_bgr(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR uint8 frame into a normalized (1, 3, input_size, input_size) float32 tensor."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR frame, got shape {frame_bgr.shape}")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, mode="RGB").resize(
            (self.input_size, self.input_size),
            Image.Resampling.BILINEAR,
        )
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        chw = arr.transpose(2, 0, 1).copy()
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    def predict_frame(self, frame_bgr: np.ndarray, *, threshold: float | None = None) -> FramePrediction:
        """Run U-Net segmentation on a single BGR video frame and return full-resolution binary mask."""
        active_threshold = self.threshold if threshold is None else float(threshold)
        if not (0.0 < active_threshold < 1.0):
            raise ValueError(f"Threshold must lie strictly between 0.0 and 1.0, got {active_threshold}")

        orig_h, orig_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        t0 = time.perf_counter()
        input_tensor = self.preprocess_bgr(frame_bgr)
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits[0, 0]).cpu().numpy()

        small_mask = (probs >= active_threshold).astype(np.uint8)
        if (orig_h, orig_w) != (self.input_size, self.input_size):
            full_mask = cv2.resize(small_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            full_mask = small_mask

        elapsed = time.perf_counter() - t0
        damaged_px = int(full_mask.sum())
        total_px = int(orig_h * orig_w)
        pct = (100.0 * damaged_px / total_px) if total_px > 0 else 0.0
        return FramePrediction(
            binary_mask=full_mask,
            damaged_pixel_count=damaged_px,
            total_pixel_count=total_px,
            damaged_percentage=pct,
            threshold=active_threshold,
            inference_seconds=elapsed,
        )


def annotate_frame(
    frame_bgr: np.ndarray,
    binary_mask: np.ndarray,
    *,
    frame_index: int,
    timestamp_seconds: float,
    threshold: float,
    damaged_percentage: float | None = None,
    skipped_as_duplicate: bool = False,
) -> np.ndarray:
    """Render translucent damage overlay, boundary contour, and compact HUD onto a BGR frame."""
    if frame_bgr.shape[:2] != binary_mask.shape[:2]:
        raise ValueError(
            f"Frame and mask spatial dimensions differ: {frame_bgr.shape[:2]} vs {binary_mask.shape[:2]}"
        )
    output = frame_bgr.copy()
    mask_bool = binary_mask > 0
    if np.any(mask_bool):
        # Translucent red fill (BGR: 40, 40, 245)
        overlay_color = np.array([40.0, 40.0, 245.0], dtype=np.float32)
        output[mask_bool] = np.clip(
            0.55 * output[mask_bool].astype(np.float32) + 0.45 * overlay_color,
            0,
            255,
        ).astype(np.uint8)
        # Bright yellow-orange contour outline (BGR: 0, 215, 255) for clear boundaries
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 215, 255), thickness=2)

    if damaged_percentage is None:
        total_px = max(int(binary_mask.size), 1)
        damaged_percentage = 100.0 * float(mask_bool.sum()) / total_px

    h, w = output.shape[:2]
    dup_tag = " [DUP-REUSED]" if skipped_as_duplicate else ""
    hud_text = (
        f"Frame: {frame_index} | Time: {timestamp_seconds:.2f}s | "
        f"Thr: {threshold:.2f} | Damage: {damaged_percentage:.1f}%{dup_tag}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.42, min(0.85, w / 900.0))
    thickness = 1 if scale < 0.65 else 2
    (text_w, text_h), baseline = cv2.getTextSize(hud_text, font, scale, thickness)
    pad_x, pad_y = 8, 6
    box_w = min(w, text_w + 2 * pad_x)
    box_h = min(h, text_h + baseline + 2 * pad_y)

    roi = output[0:box_h, 0:box_w].astype(np.float32)
    output[0:box_h, 0:box_w] = (0.25 * roi).astype(np.uint8)
    cv2.putText(
        output,
        hud_text,
        (pad_x, pad_y + text_h),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output
