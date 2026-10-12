"""Batch-wise scale consistency (BSC) losses used by SCD-Det."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

BSC_MODES = ("paper", "gt_scale_center", "log_scale_huber")


class PaperBSCLoss(nn.Module):
    """Compute paper BSC globally or independently within each assigned class.

    The input boxes are decoded foreground predictions in ``xyxy`` coordinates.
    Box widths/heights are therefore in feature-grid units during training; the
    relative-variance form is invariant to a common scale factor.
    """

    def __init__(
        self,
        lambda_cons: float = 0.0,
        eps: float = 1e-6,
        classwise: bool = False,
        mode: str = "paper",
        huber_delta: float = 1.0,
    ):
        super().__init__()
        if lambda_cons < 0:
            raise ValueError(f"lambda_cons must be non-negative, got {lambda_cons}")
        if mode not in BSC_MODES:
            raise ValueError(f"unsupported BSC mode {mode!r}; expected one of {BSC_MODES}")
        if huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta}")
        self.lambda_cons = float(lambda_cons)
        self.eps = float(eps)
        self.classwise = bool(classwise)
        self.mode = mode
        self.huber_delta = float(huber_delta)

    @staticmethod
    def _scales(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    def _relative_variance(self, predictions: torch.Tensor) -> torch.Tensor:
        widths, heights = self._scales(predictions)
        mean_width, mean_height = widths.mean(), heights.mean()
        width_term = (widths - mean_width).square() / (mean_width.square() + self.eps)
        height_term = (heights - mean_height).square() / (mean_height.square() + self.eps)
        return (width_term + height_term).mean()

    def _gt_scale_center(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_widths, pred_heights = self._scales(predictions)
        target_widths, target_heights = self._scales(targets)
        mean_width, mean_height = target_widths.mean(), target_heights.mean()
        width_term = (pred_widths - mean_width).square() / (mean_width.square() + self.eps)
        height_term = (pred_heights - mean_height).square() / (mean_height.square() + self.eps)
        return (width_term + height_term).mean()

    def _log_scale_huber(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_scales = torch.stack(self._scales(predictions), dim=-1).clamp_min(0)
        target_scales = torch.stack(self._scales(targets), dim=-1).clamp_min(0)
        log_ratio = torch.log((pred_scales + self.eps) / (target_scales + self.eps))
        return F.huber_loss(
            log_ratio,
            torch.zeros_like(log_ratio),
            delta=self.huber_delta,
            reduction="mean",
        )

    def _group_loss(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.mode == "paper":
            return self._relative_variance(predictions)
        if targets is None:
            raise ValueError(f"BSC mode {self.mode!r} requires matched target boxes")
        if self.mode == "gt_scale_center":
            return self._gt_scale_center(predictions, targets)
        return self._log_scale_huber(predictions, targets)

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        target_bboxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return weighted BSC for foreground predictions selected by the assigner."""
        zero = pred_bboxes.sum() * 0.0
        if self.lambda_cons == 0:
            return zero

        predictions = pred_bboxes[fg_mask]
        if predictions.shape[0] <= 1:
            return zero

        targets = None
        if self.mode != "paper":
            if target_bboxes is None or target_bboxes.shape != pred_bboxes.shape:
                raise ValueError("non-paper BSC requires target_bboxes shaped like pred_bboxes")
            targets = target_bboxes[fg_mask]

        if not self.classwise:
            return self.lambda_cons * self._group_loss(predictions, targets)

        if class_ids is None or class_ids.shape != fg_mask.shape:
            raise ValueError("classwise BSC requires class_ids shaped like fg_mask")
        positive_classes = class_ids[fg_mask]
        losses = []
        for class_id in positive_classes.unique():
            class_mask = positive_classes == class_id
            if class_mask.sum() > 1:
                class_targets = targets[class_mask] if targets is not None else None
                losses.append(self._group_loss(predictions[class_mask], class_targets))
        return self.lambda_cons * torch.stack(losses).mean() if losses else zero


__all__ = ("BSC_MODES", "PaperBSCLoss")
