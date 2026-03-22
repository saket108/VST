from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils.box_ops import distance_to_boxes, generalized_box_iou
from utils.points import build_points


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
        focal = alpha_factor * focal
    return focal


@dataclass
class LossOutput:
    total: torch.Tensor
    cls: torch.Tensor
    box: torch.Tensor
    center: torch.Tensor
    positives: int


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        strides: tuple[int, ...] = (4, 8, 16, 32),
        size_ranges: tuple[tuple[float, float], ...] | None = None,
        center_radius: float = 1.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.center_radius = center_radius
        self.size_ranges = size_ranges or (
            (0.0, 64.0),
            (64.0, 128.0),
            (128.0, 256.0),
            (256.0, 1e8),
        )
        if len(self.size_ranges) != len(self.strides):
            raise ValueError("size_ranges must match the number of strides.")

    def forward(
        self,
        outputs: dict[str, list[torch.Tensor] | tuple[int, ...]],
        targets: list[dict[str, torch.Tensor]],
    ) -> LossOutput:
        cls_levels = outputs["cls"]  # type: ignore[index]
        box_levels = outputs["box"]  # type: ignore[index]
        center_levels = outputs["center"]  # type: ignore[index]

        batch_size = cls_levels[0].shape[0]
        device = cls_levels[0].device
        dtype = cls_levels[0].dtype

        flat_cls: list[torch.Tensor] = []
        flat_box: list[torch.Tensor] = []
        flat_center: list[torch.Tensor] = []
        flat_points: list[torch.Tensor] = []
        flat_ranges: list[torch.Tensor] = []
        flat_strides: list[torch.Tensor] = []

        for cls_map, box_map, center_map, stride, size_range in zip(
            cls_levels, box_levels, center_levels, self.strides, self.size_ranges
        ):
            _, _, feat_h, feat_w = cls_map.shape
            points = build_points(feat_h, feat_w, stride, device, dtype)

            flat_cls.append(cls_map.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            flat_box.append(box_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) * stride)
            flat_center.append(center_map.permute(0, 2, 3, 1).reshape(batch_size, -1))
            flat_points.append(points)
            flat_ranges.append(
                torch.tensor(size_range, device=device, dtype=dtype).expand(points.shape[0], 2)
            )
            flat_strides.append(
                torch.full((points.shape[0],), stride, device=device, dtype=dtype)
            )

        pred_cls = torch.cat(flat_cls, dim=1)
        pred_box = torch.cat(flat_box, dim=1)
        pred_center = torch.cat(flat_center, dim=1)
        points = torch.cat(flat_points, dim=0)
        size_ranges = torch.cat(flat_ranges, dim=0)
        strides = torch.cat(flat_strides, dim=0)

        cls_loss = pred_cls.new_tensor(0.0)
        box_loss = pred_cls.new_tensor(0.0)
        center_loss = pred_cls.new_tensor(0.0)
        total_pos = 0

        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            assigned = self._assign_targets(points, size_ranges, strides, gt_boxes, gt_labels)

            cls_targets = pred_cls.new_zeros((points.shape[0], self.num_classes))
            pos_mask = assigned["labels"] >= 0
            if pos_mask.any():
                cls_targets[pos_mask, assigned["labels"][pos_mask]] = 1.0

            cls_loss = cls_loss + sigmoid_focal_loss(pred_cls[batch_index], cls_targets).sum()
            total_pos += int(pos_mask.sum().item())

            if pos_mask.any():
                pred_boxes = distance_to_boxes(points[pos_mask], pred_box[batch_index][pos_mask])
                target_boxes = assigned["boxes"][pos_mask]
                giou = generalized_box_iou(pred_boxes, target_boxes)
                box_loss = box_loss + (1.0 - torch.diag(giou)).sum()
                center_loss = center_loss + F.binary_cross_entropy_with_logits(
                    pred_center[batch_index][pos_mask],
                    assigned["centerness"][pos_mask],
                    reduction="sum",
                )

        normalizer = max(total_pos, 1)
        cls_loss = cls_loss / normalizer
        box_loss = box_loss / normalizer
        center_loss = center_loss / normalizer
        total = cls_loss + box_loss + center_loss
        return LossOutput(total=total, cls=cls_loss, box=box_loss, center=center_loss, positives=total_pos)

    def _assign_targets(
        self,
        points: torch.Tensor,
        size_ranges: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = points.device
        num_points = points.shape[0]
        labels = torch.full((num_points,), -1, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((num_points, 4), device=device, dtype=points.dtype)
        centerness = torch.zeros((num_points,), device=device, dtype=points.dtype)

        if gt_boxes.numel() == 0:
            return {"labels": labels, "boxes": assigned_boxes, "centerness": centerness}

        x = points[:, 0][:, None]
        y = points[:, 1][:, None]

        left = x - gt_boxes[:, 0]
        top = y - gt_boxes[:, 1]
        right = gt_boxes[:, 2] - x
        bottom = gt_boxes[:, 3] - y
        reg_targets = torch.stack([left, top, right, bottom], dim=-1)

        inside_box = reg_targets.min(dim=-1).values > 0
        max_reg = reg_targets.max(dim=-1).values
        inside_range = (max_reg >= size_ranges[:, 0:1]) & (max_reg <= size_ranges[:, 1:2])

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        radius = strides[:, None] * self.center_radius
        center_x1 = torch.maximum(gt_boxes[:, 0], gt_centers[:, 0] - radius)
        center_y1 = torch.maximum(gt_boxes[:, 1], gt_centers[:, 1] - radius)
        center_x2 = torch.minimum(gt_boxes[:, 2], gt_centers[:, 0] + radius)
        center_y2 = torch.minimum(gt_boxes[:, 3], gt_centers[:, 1] + radius)
        inside_center = (
            (x >= center_x1)
            & (x <= center_x2)
            & (y >= center_y1)
            & (y <= center_y2)
        )

        matches = inside_box & inside_range & inside_center
        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))[None, :]
        areas = areas.repeat(num_points, 1)
        areas[~matches] = float("inf")

        min_areas, matched_gt = areas.min(dim=1)
        pos_mask = torch.isfinite(min_areas)
        if not pos_mask.any():
            return {"labels": labels, "boxes": assigned_boxes, "centerness": centerness}

        matched_boxes = gt_boxes[matched_gt[pos_mask]]
        matched_regs = reg_targets[pos_mask, matched_gt[pos_mask]]
        labels[pos_mask] = gt_labels[matched_gt[pos_mask]]
        assigned_boxes[pos_mask] = matched_boxes

        lr_min = torch.minimum(matched_regs[:, 0], matched_regs[:, 2])
        lr_max = torch.maximum(matched_regs[:, 0], matched_regs[:, 2]).clamp(min=1e-6)
        tb_min = torch.minimum(matched_regs[:, 1], matched_regs[:, 3])
        tb_max = torch.maximum(matched_regs[:, 1], matched_regs[:, 3]).clamp(min=1e-6)
        centerness[pos_mask] = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))
        return {"labels": labels, "boxes": assigned_boxes, "centerness": centerness}
