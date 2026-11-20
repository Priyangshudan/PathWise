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
    macro_dice_coefficient,
    macro_intersection_over_union,
    precision,
    recall,
)
from cv.models.unet import UNet  # noqa: E402
from cv.training.train_unet import (  # noqa: E402
    combined_segmentation_loss,
    dice_loss,
    sweep_validation_thresholds,
)


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

    def test_weighted_losses_and_threshold_sweep(self):
        logits = torch.tensor([[[[2.0, -2.0], [1.5, -1.5]]]], requires_grad=True)
        targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        loss_dice_only = combined_segmentation_loss(logits, targets, bce_weight=0.0, dice_weight=1.0, batch_dice=True)
        self.assertTrue(torch.isfinite(loss_dice_only))
        loss_dice_only.backward()
        self.assertIsNotNone(logits.grad)
        with self.assertRaises(ValueError):
            combined_segmentation_loss(logits, targets, bce_weight=0.0, dice_weight=0.0)

        probs = torch.tensor([[[[0.85, 0.42, 0.10, 0.05]]]])
        gt = torch.tensor([[[[1, 1, 0, 0]]]])
        sweep = sweep_validation_thresholds(probs, gt, thresholds=(0.35, 0.50, 0.70))
        self.assertEqual(sweep["selected_threshold"], 0.35)
        self.assertAlmostEqual(sweep["selected_metrics"]["dice"], 1.0)
        self.assertLess(sweep["fixed_0.5_metrics"]["dice"], 1.0)

    def test_macro_dice_and_macro_iou_differ_from_micro_and_handle_empty_masks(self):
        # Sample 0: perfect match on 1 positive pixel (per-image Dice=1.0, IoU=1.0)
        # Sample 1: 1 TP, 1 FP, 1 FN (per-image Dice=0.5, IoU=1/3)
        # Sample 2: both empty (per-image Dice=1.0, IoU=1.0)
        # Sample 3: empty GT, 1 FP prediction (per-image Dice=0.0, IoU=0.0)
        probs = torch.tensor([
            [[[0.9, 0.1], [0.1, 0.1]]],
            [[[0.9, 0.9], [0.1, 0.1]]],
            [[[0.1, 0.1], [0.1, 0.1]]],
            [[[0.9, 0.1], [0.1, 0.1]]],
        ])
        targets = torch.tensor([
            [[1, 0], [0, 0]],
            [[1, 0], [1, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ])
        self.assertAlmostEqual(macro_dice_coefficient(probs, targets, 0.5), (1.0 + 0.5 + 1.0 + 0.0) / 4.0)
        self.assertAlmostEqual(macro_intersection_over_union(probs, targets, 0.5), (1.0 + (1.0 / 3.0) + 1.0 + 0.0) / 4.0)
        # Micro totals: TP=2, FP=2, FN=1 -> micro Dice = 4 / 7, micro IoU = 2 / 5
        self.assertAlmostEqual(dice_coefficient(probs, targets, 0.5), 4.0 / 7.0)
        self.assertAlmostEqual(intersection_over_union(probs, targets, 0.5), 2.0 / 5.0)


if __name__ == "__main__":
    unittest.main()
