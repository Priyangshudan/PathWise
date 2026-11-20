from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cv.evaluation.metrics import (  # noqa: E402
    dice_coefficient,
    f1_score,
    intersection_over_union,
    precision,
    recall,
)
from cv.models.unet import UNet  # noqa: E402
from cv.training.train_unet import combined_segmentation_loss, dice_loss  # noqa: E402


class UNetTests(unittest.TestCase):
    def test_forward_shape_channel_and_raw_logits(self):
        model = UNet(base_channels=2)
        inputs = torch.randn(2, 3, 32, 32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(outputs).all())
        odd_outputs = model(torch.randn(1, 3, 35, 37))
        self.assertEqual(tuple(odd_outputs.shape), (1, 1, 35, 37))
        with self.assertRaises(ValueError):
            UNet(out_channels=2)

    def test_combined_loss_backpropagates(self):
        model = UNet(base_channels=2)
        logits = model(torch.randn(2, 3, 32, 32))
        targets = torch.randint(0, 2, (2, 32, 32)).float()
        loss = combined_segmentation_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.0)

    def test_empty_ground_truth_masks_are_finite(self):
        logits = torch.full((2, 1, 16, 16), -4.0, requires_grad=True)
        targets = torch.zeros((2, 16, 16), dtype=torch.long)
        loss = combined_segmentation_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(dice_loss(logits, targets)))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        probs = torch.sigmoid(logits.detach())
        self.assertEqual(dice_coefficient(probs, targets), 1.0)
        self.assertEqual(intersection_over_union(probs, targets), 1.0)
        self.assertEqual(precision(probs, targets), 1.0)
        self.assertEqual(recall(probs, targets), 1.0)
        self.assertEqual(f1_score(probs, targets), 1.0)

    def test_metrics_and_configurable_threshold(self):
        probabilities = torch.tensor([[[[0.9, 0.9, 0.1, 0.1]]]])
        targets = torch.tensor([[[1, 0, 1, 0]]])
        self.assertAlmostEqual(dice_coefficient(probabilities, targets), 0.5)
        self.assertAlmostEqual(intersection_over_union(probabilities, targets), 1 / 3)
        self.assertAlmostEqual(precision(probabilities, targets), 0.5)
        self.assertAlmostEqual(recall(probabilities, targets), 0.5)
        self.assertAlmostEqual(f1_score(probabilities, targets), 0.5)
        self.assertEqual(dice_coefficient(probabilities, targets, threshold=0.95), 0.0)


if __name__ == "__main__":
    unittest.main()
