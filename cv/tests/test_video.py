"""Unit tests for the PathWise offline video ingestion, sampling, and segmentation pipeline."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.models.unet import UNet  # noqa: E402
from cv.video.frame_sampler import FrameSampler, compute_dhash, hamming_distance  # noqa: E402
from cv.video.process_video import run_video_pipeline  # noqa: E402
from cv.video.video_inference import VideoSegmentationModel, annotate_frame  # noqa: E402
from cv.video.video_reader import VideoReader  # noqa: E402


def _create_synthetic_video(path: Path, *, fps: float = 10.0, width: int = 160, height: int = 120, num_frames: int = 25) -> None:
    """Write a small synthetic MP4 video with distinct and repeated visual patterns."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic test video at {path}")
    try:
        for idx in range(num_frames):
            frame = np.full((height, width, 3), 90, dtype=np.uint8)
            # First 15 frames (0.0s .. 1.4s): identical checkerboard pattern so frames at 0.0s and 1.0s match
            if idx < 15:
                frame[20:90, 30:110] = (40, 180, 220)
                cv2.rectangle(frame, (40, 30), (100, 80), (240, 50, 50), thickness=-1)
            else:
                # Frames 15..24 (1.5s .. 2.4s): radically different pattern with diagonal stripes
                frame[:, :] = (210, 210, 210)
                cv2.circle(frame, (120, 80), 35, (10, 20, 30), thickness=-1)
                cv2.line(frame, (0, 0), (width - 1, height - 1), (0, 0, 0), thickness=6)
            writer.write(frame)
    finally:
        writer.release()


def _create_synthetic_checkpoint(path: Path, *, input_size: int = 64, base_channels: int = 2) -> None:
    """Save a lightweight synthetic U-Net checkpoint for fast unit testing."""
    torch.manual_seed(42)
    model_config = {"in_channels": 3, "out_channels": 1, "base_channels": base_channels}
    model = UNet(**model_config)
    # Bias final 1x1 conv slightly positive so some pixels exceed 0.30 threshold
    with torch.no_grad():
        model.output.bias.fill_(0.0)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "training_config": {
            "input_size": input_size,
            "metric_threshold": 0.30,
        },
        "epoch": 1,
        "random_seed": 42,
    }
    torch.save(checkpoint, path)


class VideoPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video_path = self.root / "synthetic_footpath.mp4"
        self.ckpt_path = self.root / "tiny_unet.pth"
        _create_synthetic_video(self.video_path, fps=10.0, width=160, height=120, num_frames=25)
        _create_synthetic_checkpoint(self.ckpt_path, input_size=64, base_channels=2)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_open_valid_video_extract_frames_and_timestamps(self) -> None:
        with VideoReader(self.video_path) as reader:
            meta = reader.metadata
            self.assertEqual(meta.width, 160)
            self.assertEqual(meta.height, 120)
            self.assertAlmostEqual(meta.fps, 10.0, places=2)
            self.assertEqual(meta.total_frames, 25)
            frames = list(reader)
            self.assertEqual(len(frames), 25)
            self.assertEqual(frames[0].frame_index, 0)
            self.assertAlmostEqual(frames[0].timestamp_seconds, 0.0, places=5)
            self.assertEqual(frames[10].frame_index, 10)
            self.assertAlmostEqual(frames[10].timestamp_seconds, 1.0, places=5)
            self.assertEqual(tuple(frames[0].image_bgr.shape), (120, 160, 3))
        # Resource cleanup check: iterating after exit raises RuntimeError
        with self.assertRaises(RuntimeError):
            list(reader)

    def test_invalid_missing_and_unsupported_videos_fail_gracefully(self) -> None:
        with self.assertRaises(FileNotFoundError):
            VideoReader(self.root / "nonexistent.mp4")

        unsupported = self.root / "notes.txt"
        unsupported.write_text("not a video", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported video file extension"):
            VideoReader(unsupported)

        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"corrupt video bytes")
        with self.assertRaises(ValueError):
            with VideoReader(corrupt) as reader:
                list(reader)

    def test_time_based_frame_sampling_and_near_duplicate_suppression(self) -> None:
        with VideoReader(self.video_path) as reader:
            sampler_no_skip = FrameSampler(sample_interval_seconds=1.0, skip_similar_frames=False)
            sampled = list(sampler_no_skip.sample(reader))
            # 25 frames at 10 FPS = 0.0s .. 2.4s -> sampled at 0.0s (idx 0), 1.0s (idx 10), 2.0s (idx 20)
            self.assertEqual([s.frame_index for s in sampled], [0, 10, 20])
            self.assertEqual([round(s.timestamp_seconds, 2) for s in sampled], [0.0, 1.0, 2.0])
            self.assertEqual([s.skipped_as_duplicate for s in sampled], [False, False, False])

        with VideoReader(self.video_path) as reader:
            sampler_skip = FrameSampler(
                sample_interval_seconds=1.0,
                skip_similar_frames=True,
                max_hamming_distance=4,
            )
            sampled_skip = list(sampler_skip.sample(reader))
            self.assertEqual([s.frame_index for s in sampled_skip], [0, 10, 20])
            # Frame 0 (0.0s) and Frame 10 (1.0s) have the exact same visual pattern -> Frame 10 is skipped
            # Frame 20 (2.0s) has the distinct circle/stripe pattern -> Frame 20 is NOT skipped
            self.assertFalse(sampled_skip[0].skipped_as_duplicate)
            self.assertTrue(sampled_skip[1].skipped_as_duplicate)
            self.assertFalse(sampled_skip[2].skipped_as_duplicate)
            self.assertGreater(hamming_distance(sampled_skip[0].dhash, sampled_skip[2].dhash), 4)

    def test_inference_output_dimensions_and_end_to_end_pipeline_files(self) -> None:
        segmenter = VideoSegmentationModel(self.ckpt_path, threshold=0.30, device="cpu")
        dummy_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        pred = segmenter.predict_frame(dummy_frame)
        self.assertEqual(tuple(pred.binary_mask.shape), (120, 160))
        self.assertTrue(set(np.unique(pred.binary_mask).tolist()).issubset({0, 1}))

        annotated = annotate_frame(
            dummy_frame,
            pred.binary_mask,
            frame_index=5,
            timestamp_seconds=0.5,
            threshold=0.30,
        )
        self.assertEqual(tuple(annotated.shape), (120, 160, 3))

        out_dir = self.root / "pipeline_out"
        summary = run_video_pipeline(
            self.video_path,
            out_dir,
            checkpoint_path=self.ckpt_path,
            threshold=0.30,
            sample_interval=1.0,
            skip_similar_frames=True,
            device="cpu",
            progress_every=1,
        )
        self.assertEqual(summary["frame_counts"]["total_frames"], 25)
        self.assertEqual(summary["frame_counts"]["sampled_frames"], 3)
        self.assertEqual(summary["frame_counts"]["processed_frames"], 2)
        self.assertEqual(summary["frame_counts"]["skipped_near_duplicate_frames"], 1)

        csv_file = out_dir / "frame_results.csv"
        json_file = out_dir / "summary.json"
        video_file = Path(summary["outputs"]["annotated_video"])
        self.assertTrue(csv_file.is_file())
        self.assertTrue(json_file.is_file())
        self.assertTrue(video_file.is_file())

        with csv_file.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual([int(r["frame_index"]) for r in rows], [0, 10, 20])
        self.assertEqual([r["skipped_as_duplicate"] for r in rows], ["False", "True", "False"])

        loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded_json["source_video"], "synthetic_footpath.mp4")

        # Verify output video matches original spatial dimensions (160x120) and contains 3 sampled frames
        with VideoReader(video_file) as out_reader:
            self.assertEqual(out_reader.width, 160)
            self.assertEqual(out_reader.height, 120)
            out_frames = list(out_reader)
            self.assertEqual(len(out_frames), 3)


if __name__ == "__main__":
    unittest.main()
