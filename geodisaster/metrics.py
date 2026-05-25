"""Metrics: pixel-level IoU/F1/AUPRC + calibration + decision-metric hooks.

For the few-shot / cross-domain matrices we need numerically stable accumulators
that work across many small batches; this module wraps ``torchmetrics`` where
possible and adds custom logic for sparse-positive tasks (floods/landslides).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BinaryConfusion:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor, ignore_index: int | None = 255):
        if ignore_index is not None:
            valid = target != ignore_index
            pred = pred[valid]
            target = target[valid]
        pred = pred.bool()
        target = target.bool()
        self.tp += float(torch.logical_and(pred, target).sum().item())
        self.fp += float(torch.logical_and(pred, ~target).sum().item())
        self.fn += float(torch.logical_and(~pred, target).sum().item())
        self.tn += float(torch.logical_and(~pred, ~target).sum().item())

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return float(2 * self.tp / denom) if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "iou": self.iou,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def auprc(scores: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    """Area under the precision-recall curve. Stable for sparse positives."""
    valid = target != ignore_index
    s = scores[valid].detach().float().cpu().numpy().ravel()
    t = (target[valid] == 1).detach().cpu().numpy().ravel().astype(np.int64)
    if t.sum() == 0:
        return 0.0
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(t, s))


def expected_calibration_error(scores: torch.Tensor, target: torch.Tensor, n_bins: int = 10) -> float:
    s = scores.detach().float().cpu().numpy().ravel()
    t = (target == 1).detach().cpu().numpy().ravel().astype(np.float32)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = max(len(s), 1)
    for i in range(n_bins):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if not m.any():
            continue
        conf = float(s[m].mean())
        acc = float(t[m].mean())
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def focal_bce_loss(
    logits: torch.Tensor, target: torch.Tensor,
    alpha: float = 0.75, gamma: float = 2.0, ignore_index: int = 255,
) -> torch.Tensor:
    """Focal binary cross-entropy. Target is {0, 1, ignore_index}."""
    if target.dtype != torch.long:
        target = target.long()
    valid = target != ignore_index
    t = target.float().clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), t, reduction="none")
    p = torch.sigmoid(logits.squeeze(1))
    pt = p * t + (1 - p) * (1 - t)
    w = alpha * t + (1 - alpha) * (1 - t)
    loss = w * ((1 - pt) ** gamma) * bce
    loss = loss[valid]
    return loss.mean() if loss.numel() > 0 else logits.sum() * 0.0
