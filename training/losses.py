from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils.box_ops import box_iou, distance_to_boxes, generalized_box_iou
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


def varifocal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    pred_sigmoid = logits.sigmoid()
    weight = alpha * pred_sigmoid.pow(gamma) * (targets <= 0).to(logits.dtype)
    weight = weight + targets * (targets > 0).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return loss * weight


@dataclass
class LossOutput:
    total: torch.Tensor
    cls: torch.Tensor
    box: torch.Tensor
    quality: torch.Tensor
    positives: int


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        strides: tuple[int, ...] = (4, 8, 16, 32),
        size_ranges: tuple[tuple[float, float], ...] | None = None,
        assigner: str = "fcos",
        center_radius: float = 1.5,
        topk_candidates: int = 0,
        atss_topk: int = 9,
        atss_anchor_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.assigner = assigner
        self.center_radius = center_radius
        self.topk_candidates = topk_candidates
        self.atss_topk = atss_topk
        self.atss_anchor_scale = atss_anchor_scale
        if self.assigner not in {"fcos", "atss"}:
            raise ValueError("assigner must be either 'fcos' or 'atss'.")
        if size_ranges is None:
            if len(strides) == 5:
                size_ranges = (
                    (0.0, 32.0),
                    (32.0, 64.0),
                    (64.0, 128.0),
                    (128.0, 256.0),
                    (256.0, 1e8),
                )
            else:
                size_ranges = (
                    (0.0, 64.0),
                    (64.0, 128.0),
                    (128.0, 256.0),
                    (256.0, 1e8),
                )
        self.size_ranges = size_ranges
        if len(self.size_ranges) != len(self.strides):
            raise ValueError("size_ranges must match the number of strides.")

    def forward(
        self,
        outputs: dict[str, list[torch.Tensor] | tuple[int, ...]],
        targets: list[dict[str, torch.Tensor]],
    ) -> LossOutput:
        cls_levels = outputs["cls"]  # type: ignore[index]
        box_levels = outputs["box"]  # type: ignore[index]
        quality_levels = outputs.get("quality", outputs["center"])  # type: ignore[index]

        batch_size = cls_levels[0].shape[0]
        device = cls_levels[0].device
        loss_dtype = torch.float32

        flat_cls: list[torch.Tensor] = []
        flat_box: list[torch.Tensor] = []
        flat_quality: list[torch.Tensor] = []
        flat_points: list[torch.Tensor] = []
        flat_ranges: list[torch.Tensor] = []
        flat_strides: list[torch.Tensor] = []

        for cls_map, box_map, quality_map, stride, size_range in zip(
            cls_levels, box_levels, quality_levels, self.strides, self.size_ranges
        ):
            cls_map = cls_map.float()
            box_map = box_map.float()
            quality_map = quality_map.float()
            _, _, feat_h, feat_w = cls_map.shape
            points = build_points(feat_h, feat_w, stride, device, loss_dtype)

            flat_cls.append(cls_map.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            flat_box.append(box_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) * stride)
            flat_quality.append(quality_map.permute(0, 2, 3, 1).reshape(batch_size, -1))
            flat_points.append(points)
            flat_ranges.append(
                torch.tensor(size_range, device=device, dtype=loss_dtype).expand(points.shape[0], 2)
            )
            flat_strides.append(
                torch.full((points.shape[0],), stride, device=device, dtype=loss_dtype)
            )

        pred_cls = torch.cat(flat_cls, dim=1)
        pred_box = torch.cat(flat_box, dim=1)
        pred_quality = torch.cat(flat_quality, dim=1)
        points = torch.cat(flat_points, dim=0)
        size_ranges = torch.cat(flat_ranges, dim=0)
        strides = torch.cat(flat_strides, dim=0)

        cls_loss = pred_cls.new_tensor(0.0)
        box_loss = pred_cls.new_tensor(0.0)
        quality_loss = pred_cls.new_tensor(0.0)
        total_pos = 0

        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes"].to(dtype=loss_dtype)
            gt_labels = target["labels"]
            assigned = self._assign_targets(points, size_ranges, strides, gt_boxes, gt_labels)

            cls_targets = pred_cls.new_zeros((points.shape[0], self.num_classes))
            pos_mask = assigned["labels"] >= 0

            if pos_mask.any():
                pred_boxes = distance_to_boxes(points[pos_mask], pred_box[batch_index][pos_mask])
                target_boxes = assigned["boxes"][pos_mask]
                iou_values = torch.diag(box_iou(pred_boxes, target_boxes)).detach().clamp_(0.0, 1.0)
                cls_targets[pos_mask, assigned["labels"][pos_mask]] = iou_values
                giou = generalized_box_iou(pred_boxes, target_boxes)
                box_loss = box_loss + (1.0 - torch.diag(giou)).sum()
                quality_loss = quality_loss + F.mse_loss(
                    pred_quality[batch_index][pos_mask].sigmoid(),
                    iou_values,
                    reduction="sum",
                )
                total_pos += int(pos_mask.sum().item())

            cls_loss = cls_loss + varifocal_loss(pred_cls[batch_index], cls_targets).sum()

        normalizer = max(total_pos, 1)
        cls_loss = cls_loss / normalizer
        box_loss = box_loss / normalizer
        quality_loss = quality_loss / normalizer
        total = cls_loss + box_loss + quality_loss
        return LossOutput(total=total, cls=cls_loss, box=box_loss, quality=quality_loss, positives=total_pos)

    def _assign_targets(
        self,
        points: torch.Tensor,
        size_ranges: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.assigner == "atss":
            return self._assign_targets_atss(points, strides, gt_boxes, gt_labels)
        return self._assign_targets_fcos(points, size_ranges, strides, gt_boxes, gt_labels)

    def _assign_targets_fcos(
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
        if gt_boxes.numel() == 0:
            return {"labels": labels, "boxes": assigned_boxes}

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
        if self.topk_candidates > 0:
            candidate_mask = inside_box & inside_range
            gt_widths = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
            gt_heights = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
            center_distance = (
                ((x - gt_centers[:, 0]) / gt_widths) ** 2
                + ((y - gt_centers[:, 1]) / gt_heights) ** 2
            )
            center_distance[~candidate_mask] = float("inf")

            topk_matches = torch.zeros_like(candidate_mask)
            topk = min(self.topk_candidates, num_points)
            topk_distance, topk_indices = center_distance.topk(topk, dim=0, largest=False)
            for gt_index in range(gt_boxes.shape[0]):
                valid = torch.isfinite(topk_distance[:, gt_index])
                if valid.any():
                    topk_matches[topk_indices[valid, gt_index], gt_index] = True
            matches = matches | topk_matches

        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))[None, :]
        areas = areas.repeat(num_points, 1)
        areas[~matches] = float("inf")

        min_areas, matched_gt = areas.min(dim=1)
        pos_mask = torch.isfinite(min_areas)
        if not pos_mask.any():
            return {"labels": labels, "boxes": assigned_boxes}

        matched_boxes = gt_boxes[matched_gt[pos_mask]]
        labels[pos_mask] = gt_labels[matched_gt[pos_mask]]
        assigned_boxes[pos_mask] = matched_boxes.to(dtype=assigned_boxes.dtype)
        return {"labels": labels, "boxes": assigned_boxes}

    def _assign_targets_atss(
        self,
        points: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = points.device
        num_points = points.shape[0]
        labels = torch.full((num_points,), -1, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((num_points, 4), device=device, dtype=points.dtype)
        if gt_boxes.numel() == 0:
            return {"labels": labels, "boxes": assigned_boxes}

        x = points[:, 0][:, None]
        y = points[:, 1][:, None]
        left = x - gt_boxes[:, 0]
        top = y - gt_boxes[:, 1]
        right = gt_boxes[:, 2] - x
        bottom = gt_boxes[:, 3] - y
        reg_targets = torch.stack([left, top, right, bottom], dim=-1)
        inside_box = reg_targets.min(dim=-1).values > 0

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        center_dist = (points[:, None, 0] - gt_centers[:, 0]) ** 2 + (
            points[:, None, 1] - gt_centers[:, 1]
        ) ** 2

        half_size = strides * (self.atss_anchor_scale * 0.5)
        anchors = torch.stack(
            [
                points[:, 0] - half_size,
                points[:, 1] - half_size,
                points[:, 0] + half_size,
                points[:, 1] + half_size,
            ],
            dim=-1,
        )
        anchor_ious = box_iou(anchors, gt_boxes)

        matched_gt = torch.full((num_points,), -1, dtype=torch.long, device=device)
        matched_iou = torch.zeros((num_points,), dtype=points.dtype, device=device)

        for gt_index in range(gt_boxes.shape[0]):
            candidate_indices: list[torch.Tensor] = []
            for stride_value in self.strides:
                level_indices = torch.nonzero(strides == float(stride_value), as_tuple=False).squeeze(1)
                if level_indices.numel() == 0:
                    continue
                k = min(self.atss_topk, level_indices.numel())
                level_dist = center_dist[level_indices, gt_index]
                topk_idx = level_dist.topk(k, largest=False).indices
                candidate_indices.append(level_indices[topk_idx])

            if not candidate_indices:
                continue

            candidate_indices = torch.cat(candidate_indices, dim=0)
            candidate_ious = anchor_ious[candidate_indices, gt_index]
            iou_threshold = candidate_ious.mean() + candidate_ious.std(unbiased=False)
            positive_indices = candidate_indices[candidate_ious >= iou_threshold]
            positive_indices = positive_indices[inside_box[positive_indices, gt_index]]

            if positive_indices.numel() == 0:
                inside_indices = torch.nonzero(inside_box[:, gt_index], as_tuple=False).squeeze(1)
                if inside_indices.numel() == 0:
                    continue
                best_inside = inside_indices[anchor_ious[inside_indices, gt_index].argmax()]
                positive_indices = best_inside.unsqueeze(0)

            positive_ious = anchor_ious[positive_indices, gt_index]
            better = positive_ious > matched_iou[positive_indices]
            chosen = positive_indices[better]
            matched_gt[chosen] = gt_index
            matched_iou[chosen] = positive_ious[better]

        pos_mask = matched_gt >= 0
        if not pos_mask.any():
            return {"labels": labels, "boxes": assigned_boxes}

        matched_boxes = gt_boxes[matched_gt[pos_mask]]
        labels[pos_mask] = gt_labels[matched_gt[pos_mask]]
        assigned_boxes[pos_mask] = matched_boxes.to(dtype=assigned_boxes.dtype)
        return {"labels": labels, "boxes": assigned_boxes}
