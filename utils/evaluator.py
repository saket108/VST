from __future__ import annotations

from collections import defaultdict

import torch
from tqdm.auto import tqdm

from .box_ops import box_iou


def compute_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0)[0], dims=[0])
    samples = torch.linspace(0, 1, 101)
    ap = 0.0
    for sample in samples:
        precision_at_recall = mpre[mrec >= sample].max() if (mrec >= sample).any() else 0.0
        ap += float(precision_at_recall) / 101.0
    return ap


def _evaluate_one_class(
    preds: list[tuple[int, float, torch.Tensor]],
    gt_map: dict[int, torch.Tensor],
    gt_count: int,
    iou_thresholds: torch.Tensor,
) -> dict[str, float]:
    if gt_count == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
        }

    preds = sorted(preds, key=lambda item: item[1], reverse=True)
    aps: list[float] = []
    precision50 = 0.0
    recall50 = 0.0

    for threshold_index, iou_threshold in enumerate(iou_thresholds.tolist()):
        matched = {
            image_id: torch.zeros(len(boxes), dtype=torch.bool)
            for image_id, boxes in gt_map.items()
        }

        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))

        for index, (image_id, _score, pred_box) in enumerate(preds):
            gt_boxes = gt_map.get(image_id)
            if gt_boxes is None or len(gt_boxes) == 0:
                fp[index] = 1
                continue

            ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou >= iou_threshold and not matched[image_id][best_idx]:
                matched[image_id][best_idx] = True
                tp[index] = 1
            else:
                fp[index] = 1

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recall_curve = tp_cum / max(gt_count, 1)
        precision_curve = tp_cum / (tp_cum + fp_cum).clamp(min=1e-6)
        aps.append(compute_ap(recall_curve, precision_curve))

        if threshold_index == 0:
            tp_total = float(tp.sum().item())
            fp_total = float(fp.sum().item())
            precision50 = tp_total / max(tp_total + fp_total, 1.0)
            recall50 = tp_total / max(gt_count, 1)

    return {
        "precision": precision50,
        "recall": recall50,
        "map50": aps[0],
        "map50_95": float(sum(aps) / len(aps)),
    }


@torch.no_grad()
def evaluate_detection_metrics(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    class_names: list[str] | None = None,
    conf_threshold: float = 0.05,
    nms_iou: float = 0.6,
    max_det: int = 300,
    iou_thresholds: torch.Tensor | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, object]:
    if iou_thresholds is None:
        iou_thresholds = torch.arange(0.5, 1.0, 0.05)

    if class_names is None:
        class_names = [f"class_{index}" for index in range(num_classes)]

    model.eval()
    total_images = 0
    records = {
        cls_id: {
            "preds": [],
            "gts": defaultdict(list),
            "gt_count": 0,
            "image_ids": set(),
        }
        for cls_id in range(num_classes)
    }

    iterable = loader
    if show_progress:
        iterable = tqdm(
            loader,
            total=len(loader),
            leave=True,
            dynamic_ncols=True,
            desc=progress_desc or "Validation",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    for images, targets in iterable:
        images = images.to(device)
        predictions = model.predict(
            images,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            max_det=max_det,
        )
        total_images += len(targets)

        for target, prediction in zip(targets, predictions):
            image_id = int(target["image_id"].item())
            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()

            for cls_id in range(num_classes):
                cls_gt = gt_boxes[gt_labels == cls_id]
                if cls_gt.numel() > 0:
                    records[cls_id]["gts"][image_id] = cls_gt
                    records[cls_id]["gt_count"] += int(cls_gt.shape[0])
                    records[cls_id]["image_ids"].add(image_id)

            pred_boxes = prediction["boxes"].cpu()
            pred_scores = prediction["scores"].cpu()
            pred_labels = prediction["labels"].cpu()
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                records[int(label)]["preds"].append((image_id, float(score.item()), box))

    per_class: list[dict[str, float | int | str]] = []
    valid_rows: list[dict[str, float | int | str]] = []

    for cls_id in range(num_classes):
        gt_count = int(records[cls_id]["gt_count"])
        image_count = len(records[cls_id]["image_ids"])
        metrics = _evaluate_one_class(
            preds=records[cls_id]["preds"],
            gt_map=records[cls_id]["gts"],
            gt_count=gt_count,
            iou_thresholds=iou_thresholds,
        )
        row = {
            "class_name": class_names[cls_id],
            "images": image_count,
            "instances": gt_count,
            **metrics,
        }
        per_class.append(row)
        if gt_count > 0:
            valid_rows.append(row)

    if valid_rows:
        summary = {
            "class_name": "All",
            "images": total_images,
            "instances": sum(int(row["instances"]) for row in valid_rows),
            "precision": float(sum(float(row["precision"]) for row in valid_rows) / len(valid_rows)),
            "recall": float(sum(float(row["recall"]) for row in valid_rows) / len(valid_rows)),
            "map50": float(sum(float(row["map50"]) for row in valid_rows) / len(valid_rows)),
            "map50_95": float(sum(float(row["map50_95"]) for row in valid_rows) / len(valid_rows)),
        }
    else:
        summary = {
            "class_name": "All",
            "images": total_images,
            "instances": 0,
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
        }

    return {"summary": summary, "per_class": per_class}


def evaluate_map(*args, **kwargs) -> dict[str, float]:
    report = evaluate_detection_metrics(*args, **kwargs)
    summary = report["summary"]
    return {
        "map50": float(summary["map50"]),
        "map50_95": float(summary["map50_95"]),
    }


def format_metrics_table(report: dict[str, object]) -> str:
    rows = [report["summary"], *report["per_class"]]
    headers = ["Class", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95"]

    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            [
                str(row["class_name"]),
                str(int(row["images"])),
                str(int(row["instances"])),
                f"{float(row['precision']):.3f}",
                f"{float(row['recall']):.3f}",
                f"{float(row['map50']):.3f}",
                f"{float(row['map50_95']):.3f}",
            ]
        )

    widths = [
        max(len(headers[index]), max(len(row[index]) for row in formatted_rows))
        for index in range(len(headers))
    ]

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    ]
    lines.extend(
        "  ".join(column.ljust(widths[index]) for index, column in enumerate(row))
        for row in formatted_rows
    )
    return "\n".join(lines)


def format_summary_row(report: dict[str, object]) -> str:
    row = report["summary"]
    return (
        f"{str(row['class_name']):>20}  "
        f"{int(row['images']):>6}  "
        f"{int(row['instances']):>9}  "
        f"{float(row['precision']):>9.3f}  "
        f"{float(row['recall']):>6.3f}  "
        f"{float(row['map50']):>7.3f}  "
        f"{float(row['map50_95']):>9.3f}"
    )
