"""Command-line pipeline for offline footpath video segmentation and annotated video output."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.video.frame_sampler import FrameSampler  # noqa: E402
from cv.video.video_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_THRESHOLD,
    FramePrediction,
    VideoSegmentationModel,
    annotate_frame,
)
from cv.video.video_reader import VideoReader  # noqa: E402


def _open_video_writer(
    output_dir: Path,
    stem: str,
    fps: float,
    width: int,
    height: int,
) -> tuple[cv2.VideoWriter, Path, str]:
    """Open an OpenCV VideoWriter using mp4v first, falling back to XVID/MJPG if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_fps = max(1.0, float(fps))
    candidates = [
        ("mp4v", output_dir / f"{stem}_annotated.mp4"),
        ("avc1", output_dir / f"{stem}_annotated.mp4"),
        ("XVID", output_dir / f"{stem}_annotated.avi"),
        ("MJPG", output_dir / f"{stem}_annotated.avi"),
    ]
    for codec, target_path in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(target_path), fourcc, safe_fps, (width, height))
        if writer is not None and writer.isOpened():
            return writer, target_path, codec
        if writer is not None:
            writer.release()
    raise RuntimeError(
        f"Could not open an OpenCV VideoWriter for resolution {width}x{height} at {safe_fps:.2f} FPS "
        f"in directory {output_dir}. Checked codecs: mp4v, avc1, XVID, MJPG."
    )


def run_video_pipeline(
    input_video: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    sample_interval: float = 1.0,
    skip_similar_frames: bool = False,
    max_hamming_distance: int = 4,
    device: str = "cpu",
    progress_every: int = 5,
) -> dict:
    """Execute end-to-end offline video segmentation and write video, CSV, and JSON outputs."""
    input_path = Path(input_video)
    out_dir = Path(output_dir)
    if sample_interval < 0.0 or not np.isfinite(sample_interval):
        raise ValueError("sample_interval must be a non-negative finite number")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create output directory {out_dir}: {exc}") from exc

    pipeline_started = time.perf_counter()

    # Load model once per run
    segmenter = VideoSegmentationModel(
        checkpoint_path,
        threshold=threshold,
        device=device,
    )

    sampler = FrameSampler(
        sample_interval_seconds=sample_interval,
        skip_similar_frames=skip_similar_frames,
        max_hamming_distance=max_hamming_distance,
    )

    frame_records: list[dict] = []
    writer: cv2.VideoWriter | None = None
    output_video_path: Path | None = None
    codec_used: str = ""

    sampled_count = 0
    processed_count = 0
    skipped_duplicate_count = 0
    last_prediction: FramePrediction | None = None

    with VideoReader(input_path) as reader:
        meta = reader.metadata
        output_fps = (
            (1.0 / sample_interval)
            if (sample_interval > 0.0 and (1.0 / sample_interval) <= meta.fps)
            else meta.fps
        )
        writer, output_video_path, codec_used = _open_video_writer(
            out_dir,
            input_path.stem,
            output_fps,
            meta.width,
            meta.height,
        )
        try:
            for sampled in sampler.sample(reader):
                sampled_count += 1
                frame_t0 = time.perf_counter()

                if sampled.skipped_as_duplicate and last_prediction is not None:
                    skipped_duplicate_count += 1
                    prediction = last_prediction
                    infer_secs = 0.0
                else:
                    prediction = segmenter.predict_frame(sampled.image_bgr, threshold=segmenter.threshold)
                    last_prediction = prediction
                    processed_count += 1
                    infer_secs = prediction.inference_seconds

                annotated = annotate_frame(
                    sampled.image_bgr,
                    prediction.binary_mask,
                    frame_index=sampled.frame_index,
                    timestamp_seconds=sampled.timestamp_seconds,
                    threshold=segmenter.threshold,
                    damaged_percentage=prediction.damaged_percentage,
                    skipped_as_duplicate=sampled.skipped_as_duplicate,
                )
                writer.write(annotated)
                frame_total_secs = time.perf_counter() - frame_t0

                frame_records.append({
                    "source_video": input_path.name,
                    "frame_index": sampled.frame_index,
                    "timestamp_seconds": round(sampled.timestamp_seconds, 4),
                    "probability_threshold": round(segmenter.threshold, 4),
                    "damaged_pixel_count": prediction.damaged_pixel_count,
                    "damaged_percentage": round(prediction.damaged_percentage, 4),
                    "skipped_as_duplicate": sampled.skipped_as_duplicate,
                    "hamming_distance_from_kept": (
                        sampled.hamming_distance_from_kept
                        if sampled.hamming_distance_from_kept is not None
                        else ""
                    ),
                    "inference_seconds": round(infer_secs, 5),
                    "processing_seconds": round(frame_total_secs, 5),
                })

                if progress_every > 0 and sampled_count % progress_every == 0:
                    elapsed_so_far = time.perf_counter() - pipeline_started
                    fps_rate = sampled_count / max(elapsed_so_far, 1e-6)
                    print(
                        f"[VideoPipeline] Sampled {sampled_count} frames "
                        f"({processed_count} inferred, {skipped_duplicate_count} dup-skipped) | "
                        f"video t={sampled.timestamp_seconds:.2f}s | "
                        f"damage={prediction.damaged_percentage:.2f}% | "
                        f"{elapsed_so_far:.1f}s elapsed ({fps_rate:.2f} sampled frames/s)",
                        flush=True,
                    )
        finally:
            if writer is not None:
                writer.release()

    total_pipeline_seconds = time.perf_counter() - pipeline_started

    csv_path = out_dir / "frame_results.csv"
    fieldnames = [
        "source_video",
        "frame_index",
        "timestamp_seconds",
        "probability_threshold",
        "damaged_pixel_count",
        "damaged_percentage",
        "skipped_as_duplicate",
        "hamming_distance_from_kept",
        "inference_seconds",
        "processing_seconds",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(frame_records)

    inferred_times = [
        float(r["inference_seconds"])
        for r in frame_records
        if not r["skipped_as_duplicate"]
    ]
    inferred_damage_px = [
        int(r["damaged_pixel_count"])
        for r in frame_records
        if not r["skipped_as_duplicate"]
    ]
    inferred_damage_pct = [
        float(r["damaged_percentage"])
        for r in frame_records
        if not r["skipped_as_duplicate"]
    ]
    all_sampled_damage_pct = [float(r["damaged_percentage"]) for r in frame_records]

    summary = {
        "source_video": input_path.name,
        "source_video_path": str(input_path),
        "video_properties": {
            "width": meta.width,
            "height": meta.height,
            "fps": round(meta.fps, 4),
            "fps_reliable": meta.fps_reliable,
            "reported_total_frames": meta.total_frames,
            "estimated_duration_seconds": (
                round(meta.duration_seconds, 4) if meta.duration_seconds is not None else None
            ),
        },
        "pipeline_configuration": {
            "checkpoint": str(segmenter.checkpoint_path),
            "checkpoint_epoch": segmenter.checkpoint_epoch,
            "model_input_size": segmenter.input_size,
            "probability_threshold": round(segmenter.threshold, 4),
            "sample_interval_seconds": round(sample_interval, 4),
            "skip_similar_frames": skip_similar_frames,
            "max_hamming_distance": max_hamming_distance if skip_similar_frames else None,
            "device": str(segmenter.device),
            "output_video_codec": codec_used,
            "output_video_fps": round(output_fps, 4),
        },
        "frame_counts": {
            "total_frames": meta.total_frames if meta.total_frames is not None else sampled_count,
            "sampled_frames": sampled_count,
            "processed_frames": processed_count,
            "skipped_near_duplicate_frames": skipped_duplicate_count,
        },
        "timing": {
            "average_inference_seconds": (
                round(float(np.mean(inferred_times)), 5) if inferred_times else 0.0
            ),
            "total_processing_seconds": round(total_pipeline_seconds, 4),
            "sampled_frames_per_second": (
                round(sampled_count / total_pipeline_seconds, 3)
                if total_pipeline_seconds > 0
                else 0.0
            ),
        },
        "predicted_damage_area": {
            "average_damaged_pixels_processed": (
                round(float(np.mean(inferred_damage_px)), 2) if inferred_damage_px else 0.0
            ),
            "average_damaged_percentage_processed": (
                round(float(np.mean(inferred_damage_pct)), 4) if inferred_damage_pct else 0.0
            ),
            "average_damaged_percentage_all_sampled": (
                round(float(np.mean(all_sampled_damage_pct)), 4) if all_sampled_damage_pct else 0.0
            ),
            "max_damaged_percentage_sampled": (
                round(float(np.max(all_sampled_damage_pct)), 4) if all_sampled_damage_pct else 0.0
            ),
            "disclaimer": (
                "Predicted damaged pixel counts and percentages represent 2D image-plane proportions "
                "only and must not be interpreted as physical real-world dimensions or engineering severity."
            ),
        },
        "outputs": {
            "annotated_video": str(output_video_path),
            "frame_results_csv": str(csv_path),
            "summary_json": str(out_dir / "summary.json"),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[VideoPipeline] Completed in {total_pipeline_seconds:.2f}s | "
        f"Sampled: {sampled_count}, Processed: {processed_count}, Skipped duplicates: {skipped_duplicate_count} | "
        f"Annotated video: {output_video_path}",
        flush=True,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PathWise offline footpath damage segmentation on a video file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input footpath video file (e.g. .mp4, .avi)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "video_demo",
        help="Directory for annotated video, frame_results.csv, and summary.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to trained U-Net checkpoint (default: exp2_train_improvements/best_model.pth)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for binary damage mask (default: 0.30 for Exp 2)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Time interval in seconds between sampled frames (0 processes every frame; default: 1.0)",
    )
    parser.add_argument(
        "--skip-similar-frames",
        action="store_true",
        help="Suppress redundant inference on near-identical consecutive sampled frames via dHash",
    )
    parser.add_argument(
        "--max-hamming-distance",
        type=int,
        default=4,
        help="Maximum 64-bit dHash Hamming distance for consecutive duplicate suppression (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to run inference on (default: cpu)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_video_pipeline(
            args.input,
            args.output_dir,
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
            sample_interval=args.sample_interval,
            skip_similar_frames=args.skip_similar_frames,
            max_hamming_distance=args.max_hamming_distance,
            device=args.device,
        )
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
