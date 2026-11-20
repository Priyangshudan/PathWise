"""Binary segmentation metrics operating on probabilities and binary targets."""

from __future__ import annotations

import torch


def _counts(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float) -> tuple[int, int, int]:
    probabilities = torch.as_tensor(probabilities)
    targets = torch.as_tensor(targets, device=probabilities.device)
    if probabilities.ndim == targets.ndim + 1 and probabilities.shape[1] == 1:
        probabilities = probabilities.squeeze(1)
    if probabilities.shape != targets.shape:
        raise ValueError(f"Probability and target shapes differ: {probabilities.shape} vs {targets.shape}")
    predicted = probabilities >= threshold
    actual = targets >= 0.5
    tp = int(torch.logical_and(predicted, actual).sum().item())
    fp = int(torch.logical_and(predicted, torch.logical_not(actual)).sum().item())
    fn = int(torch.logical_and(torch.logical_not(predicted), actual).sum().item())
    return tp, fp, fn


def _ratio(numerator: int, denominator: int, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def dice_coefficient(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Global Dice score; returns 1 when prediction and target are both empty."""
    tp, fp, fn = _counts(probabilities, targets, threshold)
    return _ratio(2 * tp, 2 * tp + fp + fn, empty_value=1.0)


def intersection_over_union(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Global binary IoU; returns 1 when prediction and target are both empty."""
    tp, fp, fn = _counts(probabilities, targets, threshold)
    return _ratio(tp, tp + fp + fn, empty_value=1.0)


def precision(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    tp, fp, fn = _counts(probabilities, targets, threshold)
    return _ratio(tp, tp + fp, empty_value=1.0 if fn == 0 else 0.0)


def recall(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    tp, fp, fn = _counts(probabilities, targets, threshold)
    return _ratio(tp, tp + fn, empty_value=1.0 if fp == 0 else 0.0)


def f1_score(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    p = precision(probabilities, targets, threshold)
    r = recall(probabilities, targets, threshold)
    return _ratio(2 * p * r, p + r, empty_value=1.0 if p == 1.0 and r == 1.0 else 0.0)


def _normalize_batch(probabilities: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.as_tensor(probabilities)
    targets = torch.as_tensor(targets, device=probabilities.device)
    if probabilities.ndim == targets.ndim + 1 and probabilities.shape[1] == 1:
        probabilities = probabilities.squeeze(1)
    if targets.ndim == probabilities.ndim + 1 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    if probabilities.shape != targets.shape:
        raise ValueError(f"Probability and target shapes differ: {probabilities.shape} vs {targets.shape}")
    if probabilities.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
        targets = targets.unsqueeze(0)
    elif probabilities.ndim == 4 and probabilities.shape[1] == 1:
        probabilities = probabilities.squeeze(1)
        targets = targets.squeeze(1)
    if probabilities.ndim != 3:
        raise ValueError(f"Expected 2D or 3D spatial batch (N, H, W), got shape {probabilities.shape}")
    return probabilities, targets


def macro_dice_coefficient(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Mean per-image Dice score across a batch.

    Empty ground-truth and empty prediction for the same image return 1.0; if
    only one of ground truth or prediction is empty, that image scores 0.0.
    """
    probs, targs = _normalize_batch(probabilities, targets)
    scores = [dice_coefficient(probs[i], targs[i], threshold) for i in range(probs.shape[0])]
    return sum(scores) / len(scores) if scores else 0.0


def macro_intersection_over_union(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Mean per-image binary IoU across a batch.

    Empty ground-truth and empty prediction for the same image return 1.0; if
    only one of ground truth or prediction is empty, that image scores 0.0.
    """
    probs, targs = _normalize_batch(probabilities, targets)
    scores = [intersection_over_union(probs[i], targs[i], threshold) for i in range(probs.shape[0])]
    return sum(scores) / len(scores) if scores else 0.0
