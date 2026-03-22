from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype
        )

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou

    lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt_inter = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb_inter = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0)
    inter = wh_inter[..., 0] * wh_inter[..., 1]
    union = area1[:, None] + area2 - inter
    return iou - (area - union) / area.clamp(min=1e-6)


def distance_to_boxes(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    x = points[:, 0]
    y = points[:, 1]
    left = distances[:, 0]
    top = distances[:, 1]
    right = distances[:, 2]
    bottom = distances[:, 3]
    return torch.stack([x - left, y - top, x + right, y + bottom], dim=-1)
