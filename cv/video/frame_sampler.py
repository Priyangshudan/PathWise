"""Time-based frame sampling and consecutive near-duplicate frame suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import cv2
import numpy as np

from cv.video.video_reader import VideoFrame


@dataclass(frozen=True)
class SampledFrame:
    """A time-sampled video frame annotated with near-duplicate suppression metadata."""

    frame_index: int
    timestamp_seconds: float
    image_bgr: np.ndarray
    dhash: int
    hamming_distance_from_kept: int | None
    skipped_as_duplicate: bool


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
    """Compute a 64-bit horizontal difference hash (dHash) for a BGR image.

    Converts the image to grayscale, resizes to ``(hash_size + 1, hash_size)``
    with area interpolation, and sets bit ``i`` when pixel ``(r, c + 1) > (r, c)``.
    This measures visual structure independently of model predictions.
    """
    if hash_size < 2 or hash_size > 16:
        raise ValueError("hash_size must be between 2 and 16")
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 2:
        gray = image_bgr
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image_bgr.shape}")
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Bitwise Hamming distance between two integer perceptual hashes."""
    return (hash_a ^ hash_b).bit_count()


class FrameSampler:
    """Sample frames at a configurable time interval and optionally suppress near-identical consecutive frames.

    Note: Near-duplicate suppression compares consecutive sampled frames using perceptual
    dHash similarity only to avoid redundant inference when a camera is stationary. It does
    not perform physical damage tracking or multi-view place recognition.
    """

    def __init__(
        self,
        sample_interval_seconds: float = 1.0,
        *,
        skip_similar_frames: bool = False,
        max_hamming_distance: int = 4,
    ) -> None:
        if sample_interval_seconds < 0.0 or not np.isfinite(sample_interval_seconds):
            raise ValueError("sample_interval_seconds must be a non-negative finite number")
        if max_hamming_distance < 0 or max_hamming_distance > 64:
            raise ValueError("max_hamming_distance must lie between 0 and 64")
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.skip_similar_frames = bool(skip_similar_frames)
        self.max_hamming_distance = int(max_hamming_distance)

    def sample(self, frames: Iterable[VideoFrame]) -> Iterator[SampledFrame]:
        next_target_time = 0.0
        epsilon = 1e-6
        last_kept_hash: int | None = None

        for frame in frames:
            if self.sample_interval_seconds > 0.0 and (frame.timestamp_seconds + epsilon) < next_target_time:
                continue

            current_hash = compute_dhash(frame.image_bgr)
            dist = hamming_distance(current_hash, last_kept_hash) if last_kept_hash is not None else None
            is_duplicate = bool(
                self.skip_similar_frames
                and dist is not None
                and dist <= self.max_hamming_distance
            )
            if not is_duplicate:
                last_kept_hash = current_hash

            yield SampledFrame(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                image_bgr=frame.image_bgr,
                dhash=current_hash,
                hamming_distance_from_kept=dist,
                skipped_as_duplicate=is_duplicate,
            )

            if self.sample_interval_seconds > 0.0:
                while next_target_time <= frame.timestamp_seconds + epsilon:
                    next_target_time += self.sample_interval_seconds
