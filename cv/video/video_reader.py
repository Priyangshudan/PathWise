"""Video ingestion and sequential frame extraction utilities for PathWise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class VideoMetadata:
    """Properties reported or inferred for an opened video file."""

    path: Path
    fps: float
    fps_reliable: bool
    width: int
    height: int
    total_frames: int | None
    duration_seconds: float | None


@dataclass(frozen=True)
class VideoFrame:
    """Single decoded video frame in BGR format with index and timestamp."""

    frame_index: int
    timestamp_seconds: float
    image_bgr: np.ndarray


class VideoReader:
    """Sequential OpenCV video reader with resource cleanup and fallback metadata handling."""

    def __init__(self, path: str | Path, *, default_fps: float = 30.0) -> None:
        if default_fps <= 0.0 or not np.isfinite(default_fps):
            raise ValueError("default_fps must be a positive finite number")
        self.path = Path(path)
        self.default_fps = float(default_fps)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Video path is not a regular file: {self.path}")
        if self.path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video file extension {self.path.suffix!r} for {self.path.name}; "
                f"supported extensions: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
            )

        self._cap: cv2.VideoCapture | None = cv2.VideoCapture(str(self.path))
        if self._cap is None or not self._cap.isOpened():
            self.close()
            raise ValueError(f"OpenCV could not open video file (corrupt or unsupported codec): {self.path}")

        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if np.isfinite(raw_fps) and 0.1 <= raw_fps <= 240.0:
            self.fps = raw_fps
            self.fps_reliable = True
        else:
            self.fps = self.default_fps
            self.fps_reliable = False

        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._buffered_first_frame: np.ndarray | None = None
        if raw_w <= 0 or raw_h <= 0:
            ok, first_frame = self._cap.read()
            if not ok or first_frame is None or first_frame.size == 0:
                self.close()
                raise ValueError(f"Video file contains no readable frames: {self.path}")
            raw_h, raw_w = int(first_frame.shape[0]), int(first_frame.shape[1])
            self._buffered_first_frame = first_frame

        self.width = raw_w
        self.height = raw_h
        self.total_frames: int | None = raw_count if raw_count > 0 else None
        self.duration_seconds: float | None = (
            self.total_frames / self.fps if (self.total_frames is not None and self.fps > 0) else None
        )

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            path=self.path,
            fps=self.fps,
            fps_reliable=self.fps_reliable,
            width=self.width,
            height=self.height,
            total_frames=self.total_frames,
            duration_seconds=self.duration_seconds,
        )

    def __iter__(self) -> Iterator[VideoFrame]:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoReader is closed and cannot be iterated")

        frame_index = 0
        if self._buffered_first_frame is not None:
            first = self._buffered_first_frame
            self._buffered_first_frame = None
            yield VideoFrame(
                frame_index=0,
                timestamp_seconds=0.0,
                image_bgr=first,
            )
            frame_index = 1
        else:
            # Rewind to frame 0 in case metadata probes moved the cursor
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / self.fps
            yield VideoFrame(
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                image_bgr=frame,
            )
            frame_index += 1

        if frame_index == 0:
            raise ValueError(f"Video file contains zero readable frames: {self.path}")

    def close(self) -> None:
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()
            self._cap = None
        self._buffered_first_frame = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
