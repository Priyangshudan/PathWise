"""PyTorch dataset for paired footpath damage images and masks."""

from __future__ import annotations

import random
import csv
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FootpathDamageDataset(Dataset):
    """Read image/mask pairs in an official split, without caching the dataset.

    Expected layout (the archive's verified layout): ``<root>/<split>/<id>.jpg``
    and ``<root>/<split>/<id>.png``. For a ZIP source, the same paths may be
    provided directly and read lazily without extracting or changing it.
    The observed near-black/near-white mask values are thresholded into
    background/damage labels. Nine paired masks have transposed dimensions; the
    audit visually verified that rotating those masks 90 degrees counterclockwise
    aligns them. A synchronized horizontal flip is optional.
    """

    def __init__(self, root: str | Path, split: str, *, horizontal_flip: bool = False, size: tuple[int, int] | None = None, mask_threshold: int = 128, manifest: str | Path | None = None):
        self._zip = None
        self.root = Path(root)
        self.split = {"val": "validation"}.get(split.lower(), split.lower())
        split_map = {"train": "Train", "validation": "Val", "test": "Test"}
        if self.split not in split_map:
            raise ValueError(f"Unknown split {split!r}; expected train, val, or test")
        self.split_dir = split_map[self.split]
        self.horizontal_flip = horizontal_flip
        self.size = size  # (height, width); None preserves original dimensions
        if not 10 <= mask_threshold <= 247:
            raise ValueError("mask_threshold must lie in the observed gap from 10 through 247")
        self.mask_threshold = mask_threshold
        if self.root.is_file() and self.root.suffix.lower() == ".zip":
            import zipfile

            self._zip = zipfile.ZipFile(self.root)
            prefix = self._find_zip_prefix(self._zip.namelist())
            if manifest is not None:
                self.pairs = self._manifest_pairs(manifest, prefix)
            else:
                names = [n for n in self._zip.namelist() if n.startswith(prefix + self.split_dir + "/")]
                image_names = sorted(n for n in names if n.lower().endswith(".jpg"))
                self.pairs = [(n, n[:-4] + ".png") for n in image_names]
        elif self.root.is_dir():
            if manifest is not None:
                self.pairs = self._manifest_pairs(manifest)
            else:
                directory = self.root / self.split_dir
                self.pairs = [(p, p.with_suffix(".png")) for p in sorted(directory.glob("*.jpg"))]
        else:
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.pairs:
            raise ValueError(f"No JPEG images found for split {self.split!r} under {self.root}")
        for image_ref, mask_ref in self.pairs:
            if self._zip:
                if mask_ref not in self._zip.namelist():
                    raise FileNotFoundError(f"Missing mask {mask_ref}")
            elif not mask_ref.is_file():
                raise FileNotFoundError(f"Missing mask {mask_ref}")

    def _manifest_pairs(self, manifest: str | Path, zip_prefix: str = "") -> list[tuple[Path | str, Path | str]]:
        """Select only included rows for this original split from a derived manifest."""
        pairs = []
        expected_folder = self.split_dir
        with Path(manifest).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"official_split", "image_path", "mask_path", "benchmark_status", "benchmark_split"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Manifest must include columns: {', '.join(sorted(required))}")
            for row in reader:
                status = row["benchmark_status"].strip()
                if status not in {"included", "quarantined"}:
                    raise ValueError(f"Unknown manifest benchmark_status: {status!r}")
                if status != "included":
                    continue
                official = {"val": "validation"}.get(row["official_split"].strip().lower(), row["official_split"].strip().lower())
                benchmark = {"val": "validation"}.get(row["benchmark_split"].strip().lower(), row["benchmark_split"].strip().lower())
                if official != benchmark:
                    raise ValueError("Included manifest rows must retain their official split")
                if official != self.split:
                    continue
                image_rel = PurePosixPath(row["image_path"].replace("\\", "/"))
                mask_rel = PurePosixPath(row["mask_path"].replace("\\", "/"))
                if image_rel.is_absolute() or mask_rel.is_absolute() or ".." in image_rel.parts or ".." in mask_rel.parts:
                    raise ValueError("Manifest image and mask paths must be relative and remain under the dataset root")
                if (not image_rel.parts or image_rel.parts[0] != expected_folder or
                        not mask_rel.parts or mask_rel.parts[0] != expected_folder or
                        image_rel.parent != mask_rel.parent or image_rel.stem != mask_rel.stem or
                        image_rel.suffix.lower() != ".jpg" or mask_rel.suffix.lower() != ".png"):
                    raise ValueError("Manifest image/mask paths must be matching JPG/PNG pairs within the official split")
                if self._zip:
                    image_ref = zip_prefix + image_rel.as_posix()
                    mask_ref = zip_prefix + mask_rel.as_posix()
                else:
                    image_ref = self.root.joinpath(*image_rel.parts)
                    mask_ref = self.root.joinpath(*mask_rel.parts)
                pairs.append((image_ref, mask_ref))
        return pairs

    @staticmethod
    def _find_zip_prefix(names: list[str]) -> str:
        candidates = [n.split("/", 1)[0] for n in names if "/" in n and n.split("/", 1)[1].startswith("Train/")]
        if not candidates:
            raise ValueError("Could not find dataset root containing Train/ in ZIP")
        return candidates[0] + "/"

    def __len__(self) -> int:
        return len(self.pairs)

    def _open(self, ref: Path | str) -> Image.Image:
        if self._zip:
            return Image.open(self._zip.open(ref))
        return Image.open(ref)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_ref, mask_ref = self.pairs[index]
        with self._open(image_ref) as im, self._open(mask_ref) as mk:
            im = im.convert("RGB")
            if mk.mode != "L":
                raise ValueError(f"Expected grayscale L mask, got {mk.mode}: {mask_ref}")
            if mk.size != im.size:
                if mk.size == (im.height, im.width) and im.width != im.height:
                    mk = mk.transpose(Image.Transpose.ROTATE_90)
                else:
                    raise ValueError(f"Unexpected image/mask dimensions: {im.size} vs {mk.size} for {mask_ref}")
            if self.size is not None:
                height, width = self.size
                im = im.resize((width, height), Image.Resampling.BILINEAR)
                mk = mk.resize((width, height), Image.Resampling.NEAREST)
            image = np.asarray(im, dtype=np.float32) / 255.0
            raw_mask = np.asarray(mk)
        if not np.all(np.isin(raw_mask, np.r_[0:10, 247:256])):
            raise ValueError(f"Mask contains pixel values outside the audited encoding: {mask_ref}")
        labels = (raw_mask >= self.mask_threshold).astype(np.uint8)
        if self.horizontal_flip and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            labels = np.flip(labels, axis=1).copy()
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(labels.copy()).long()
        return image_tensor, mask_tensor

    def close(self) -> None:
        archive = getattr(self, "_zip", None)
        if archive:
            archive.close()
            self._zip = None

    def __getstate__(self) -> dict:
        """Reopen ZIP handles in spawned Windows DataLoader workers."""
        state = self.__dict__.copy()
        if state.get("_zip") is not None:
            state["_zip"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.root.is_file() and self.root.suffix.lower() == ".zip":
            import zipfile

            self._zip = zipfile.ZipFile(self.root)

    def __del__(self) -> None:
        self.close()
