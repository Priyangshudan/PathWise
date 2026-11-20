from __future__ import annotations

import csv
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cv.data.dataset import FootpathDamageDataset  # noqa: E402


def make_pair(folder: Path, name: str, size=(12, 8)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 120, dtype=np.uint8), "RGB").save(folder / f"{name}.jpg")
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    shift = ord(name[0]) % 3
    mask[2:5, 3 + shift:7 + shift] = 255
    Image.fromarray(mask, "L").save(folder / f"{name}.png")


class FootpathDamageDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_pair(self.root / "Train", "a")
        make_pair(self.root / "Train", "b")
        make_pair(self.root / "Val", "c")
        make_pair(self.root / "Test", "d")

    def tearDown(self):
        self.temp.cleanup()

    def test_pairing_tensor_shapes_types_and_labels(self):
        dataset = FootpathDamageDataset(self.root, "train")
        self.assertEqual(len(dataset), 2)
        image, mask = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 8, 12))
        self.assertEqual(tuple(mask.shape), (8, 12))
        self.assertEqual(str(image.dtype), "torch.float32")
        self.assertEqual(str(mask.dtype), "torch.int64")
        self.assertTrue(set(mask.unique().tolist()).issubset({0, 1}))
        self.assertEqual(int(mask[2, 4]), 1)  # index 0 is a.png / a.png mask
        self.assertEqual(int(mask[2, 3]), 0)

    def test_dataset_indexing_and_split_integrity(self):
        self.assertEqual(len(FootpathDamageDataset(self.root, "train")), 2)
        self.assertEqual(len(FootpathDamageDataset(self.root, "val")), 1)
        self.assertEqual(len(FootpathDamageDataset(self.root, "test")), 1)
        with self.assertRaises(IndexError):
            _ = FootpathDamageDataset(self.root, "train")[2]

    def test_missing_mask_is_reported(self):
        (self.root / "Train" / "b.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing mask"):
            FootpathDamageDataset(self.root, "train")

    def test_invalid_root_and_split_are_reported(self):
        with self.assertRaises(FileNotFoundError):
            FootpathDamageDataset(self.root / "absent", "train")
        with self.assertRaises(ValueError):
            FootpathDamageDataset(self.root, "development")

    def test_corrupt_file_fails_when_read(self):
        (self.root / "Train" / "a.png").write_bytes(b"not an image")
        dataset = FootpathDamageDataset(self.root, "train")
        with self.assertRaises(UnidentifiedImageError):
            _ = dataset[0]

    def test_unexpected_mask_values_are_rejected(self):
        Image.fromarray(np.full((8, 12), 128, dtype=np.uint8), "L").save(self.root / "Train" / "a.png")
        with self.assertRaisesRegex(ValueError, "outside the audited encoding"):
            _ = FootpathDamageDataset(self.root, "train")[0]

    def test_resize_uses_requested_dimensions(self):
        image, mask = FootpathDamageDataset(self.root, "train", size=(4, 6))[0]
        self.assertEqual(tuple(image.shape), (3, 4, 6))
        self.assertEqual(tuple(mask.shape), (4, 6))
        self.assertTrue(set(mask.unique().tolist()).issubset({0, 1}))

    def test_mask_threshold_is_configurable_within_observed_gap(self):
        _, mask = FootpathDamageDataset(self.root, "train", mask_threshold=200)[0]
        self.assertTrue(set(mask.unique().tolist()).issubset({0, 1}))
        with self.assertRaisesRegex(ValueError, "observed gap"):
            FootpathDamageDataset(self.root, "train", mask_threshold=9)

    def test_transposed_pair_is_rotated_consistently(self):
        folder = self.root / "Train"
        Image.fromarray(np.full((8, 12, 3), 100, dtype=np.uint8)).save(folder / "rot.jpg")
        mask = np.zeros((12, 8), dtype=np.uint8)
        mask[2:5, 1:6] = 255
        Image.fromarray(mask, "L").save(folder / "rot.png")
        ds = FootpathDamageDataset(self.root, "train")
        index = next(i for i, pair in enumerate(ds.pairs) if Path(pair[0]).stem == "rot")
        _, labels = ds[index]
        self.assertEqual(tuple(labels.shape), (8, 12))
        expected = np.rot90(mask, 1).copy() >= 128
        np.testing.assert_array_equal(labels.numpy().astype(bool), expected)

    def test_zip_dataset_root(self):
        archive = self.root / "sample.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for file in (self.root / "Train").iterdir():
                z.write(file, f"Fixture/Train/{file.name}")
        ds = FootpathDamageDataset(archive, "train")
        self.assertEqual(len(ds), 2)
        image, mask = ds[0]
        self.assertEqual(tuple(image.shape), (3, 8, 12))
        self.assertEqual(tuple(mask.shape), (8, 12))

    def test_experiment_manifest_filters_quarantined_rows_and_keeps_splits(self):
        manifest = self.root / "experiment_manifest.csv"
        rows = [
            {"image_id": "a", "official_split": "train", "image_path": "Train/a.jpg", "mask_path": "Train/a.png", "benchmark_status": "included", "benchmark_split": "train"},
            {"image_id": "b", "official_split": "train", "image_path": "Train/b.jpg", "mask_path": "Train/b.png", "benchmark_status": "quarantined", "benchmark_split": ""},
            {"image_id": "c", "official_split": "validation", "image_path": "Val/c.jpg", "mask_path": "Val/c.png", "benchmark_status": "included", "benchmark_split": "validation"},
            {"image_id": "d", "official_split": "test", "image_path": "Test/d.jpg", "mask_path": "Test/d.png", "benchmark_status": "included", "benchmark_split": "test"},
        ]
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        train = FootpathDamageDataset(self.root, "train", manifest=manifest)
        val = FootpathDamageDataset(self.root, "val", manifest=manifest)
        test = FootpathDamageDataset(self.root, "test", manifest=manifest)
        self.assertEqual([Path(pair[0]).stem for pair in train.pairs], ["a"])
        self.assertEqual([Path(pair[0]).stem for pair in val.pairs], ["c"])
        self.assertEqual([Path(pair[0]).stem for pair in test.pairs], ["d"])
        image, mask = train[0]
        self.assertEqual(tuple(image.shape), (3, 8, 12))
        self.assertTrue(set(mask.unique().tolist()).issubset({0, 1}))
        archive = self.root / "manifest_sample.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for split_dir in ("Train", "Val", "Test"):
                for file in (self.root / split_dir).glob("*"):
                    z.write(file, f"Fixture/{split_dir}/{file.name}")
        zipped = FootpathDamageDataset(archive, "train", manifest=manifest)
        self.assertEqual(len(zipped), 1)
        zipped_image, zipped_mask = zipped[0]
        self.assertEqual(tuple(zipped_image.shape), (3, 8, 12))
        self.assertTrue(set(zipped_mask.unique().tolist()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
