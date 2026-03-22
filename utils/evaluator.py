from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from .box_ops import box_iou


def smooth(values: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    if values.size == 0:
        return values
    filter_size = round(len(values) * fraction * 2) // 2 + 1
    padding = np.ones(filter_size // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]), axis=0)
    return np.convolve(padded, np.ones(filter_size) / filter_size, mode="valid")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    samples = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(samples, mrec, mpre), samples))


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> np.ndarray:
    correct = np.zeros((pred_boxes.shape[0], iou_thresholds.numel()), dtype=bool)
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return correct

    iou = box_iou(gt_boxes, pred_boxes)
    correct_class = gt_labels[:, None] == pred_labels[None, :]

    for threshold_index, threshold in enumerate(iou_thresholds):
        matches = torch.where((iou >= threshold) & correct_class)
        if matches[0].numel() == 0:
            continue

        match_data = torch.cat(
            (
                torch.stack(matches, dim=1),
                iou[matches[0], matches[1]].unsqueeze(1),
            ),
            dim=1,
        ).cpu().numpy()

        if match_data.shape[0] > 1:
            match_data = match_data[match_data[:, 2].argsort()[::-1]]
            match_data = match_data[np.unique(match_data[:, 1], return_index=True)[1]]
            match_data = match_data[match_data[:, 2].argsort()[::-1]]
            match_data = match_data[np.unique(match_data[:, 0], return_index=True)[1]]

        correct[match_data[:, 1].astype(int), threshold_index] = True

    return correct


def ap_per_class(
    true_positive: np.ndarray,
    confidence: np.ndarray,
    pred_classes: np.ndarray,
    target_classes: np.ndarray,
    iou_thresholds: torch.Tensor,
    eps: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target_classes.size == 0:
        shape = (0,)
        return np.zeros(shape), np.zeros(shape), np.zeros((0, iou_thresholds.numel())), np.zeros(0, dtype=int)

    order = np.argsort(-confidence)
    true_positive = true_positive[order]
    confidence = confidence[order]
    pred_classes = pred_classes[order]

    unique_classes, target_count = np.unique(target_classes, return_counts=True)
    class_count = unique_classes.shape[0]
    points = np.linspace(0, 1, 1000)
    precision_curve = np.zeros((class_count, points.size))
    recall_curve = np.zeros((class_count, points.size))
    ap = np.zeros((class_count, iou_thresholds.numel()))

    for class_index, class_id in enumerate(unique_classes):
        keep = pred_classes == class_id
        num_labels = target_count[class_index]
        num_predictions = int(keep.sum())
        if num_predictions == 0 or num_labels == 0:
            continue

        false_positive = (1.0 - true_positive[keep]).cumsum(0)
        true_positive_cum = true_positive[keep].cumsum(0)

        recall = true_positive_cum / (num_labels + eps)
        precision = true_positive_cum / (true_positive_cum + false_positive + eps)
        recall_curve[class_index] = np.interp(-points, -confidence[keep], recall[:, 0], left=0)
        precision_curve[class_index] = np.interp(-points, -confidence[keep], precision[:, 0], left=1)

        for iou_index in range(iou_thresholds.numel()):
            ap[class_index, iou_index] = compute_ap(recall[:, iou_index], precision[:, iou_index])

    f1_curve = 2.0 * precision_curve * recall_curve / (precision_curve + recall_curve + eps)
    if f1_curve.size == 0:
        best_index = 0
    else:
        best_index = int(smooth(f1_curve.mean(0), 0.1).argmax())

    precision = precision_curve[:, best_index] if class_count else np.zeros(0)
    recall = recall_curve[:, best_index] if class_count else np.zeros(0)
    return precision, recall, ap, unique_classes.astype(int)


@dataclass
class DetectionMetricsAccumulator:
    num_classes: int
    class_names: list[str]
    iou_thresholds: torch.Tensor = field(
        default_factory=lambda: torch.arange(0.5, 1.0, 0.05, dtype=torch.float32)
    )
    total_images: int = 0
    correct: list[np.ndarray] = field(default_factory=list)
    confidence: list[np.ndarray] = field(default_factory=list)
    pred_classes: list[np.ndarray] = field(default_factory=list)
    target_classes: list[np.ndarray] = field(default_factory=list)
    class_image_ids: list[set[int]] = field(init=False)
    class_instances: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.class_image_ids = [set() for _ in range(self.num_classes)]
        self.class_instances = [0 for _ in range(self.num_classes)]

    def update(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        for prediction, target in zip(predictions, targets):
            self.total_images += 1
            image_id = int(target["image_id"].item())

            gt_boxes = target["boxes"].detach().cpu().float()
            gt_labels = target["labels"].detach().cpu().long()
            if gt_labels.numel() > 0:
                unique_labels, counts = torch.unique(gt_labels, return_counts=True)
                for label, count in zip(unique_labels.tolist(), counts.tolist()):
                    self.class_image_ids[label].add(image_id)
                    self.class_instances[label] += int(count)
                self.target_classes.append(gt_labels.numpy())

            pred_boxes = prediction["boxes"].detach().cpu().float()
            pred_scores = prediction["scores"].detach().cpu().float()
            pred_labels = prediction["labels"].detach().cpu().long()
            if pred_scores.numel() > 0:
                self.correct.append(
                    match_predictions(
                        pred_boxes=pred_boxes,
                        pred_labels=pred_labels,
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels,
                        iou_thresholds=self.iou_thresholds,
                    )
                )
                self.confidence.append(pred_scores.numpy())
                self.pred_classes.append(pred_labels.numpy())

    def compute(self) -> dict[str, object]:
        if self.correct:
            correct = np.concatenate(self.correct, axis=0)
            confidence = np.concatenate(self.confidence, axis=0)
            pred_classes = np.concatenate(self.pred_classes, axis=0)
        else:
            correct = np.zeros((0, self.iou_thresholds.numel()), dtype=bool)
            confidence = np.zeros((0,), dtype=np.float32)
            pred_classes = np.zeros((0,), dtype=np.int64)

        if self.target_classes:
            target_classes = np.concatenate(self.target_classes, axis=0)
        else:
            target_classes = np.zeros((0,), dtype=np.int64)

        precision, recall, ap, evaluated_classes = ap_per_class(
            true_positive=correct.astype(np.float32),
            confidence=confidence.astype(np.float32),
            pred_classes=pred_classes.astype(np.int64),
            target_classes=target_classes.astype(np.int64),
            iou_thresholds=self.iou_thresholds,
        )
        metrics_by_class = {
            int(class_id): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "map50": float(ap[index, 0]),
                "map50_95": float(ap[index].mean()),
            }
            for index, class_id in enumerate(evaluated_classes.tolist())
        }

        per_class: list[dict[str, float | int | str]] = []
        valid_rows: list[dict[str, float | int | str]] = []
        for class_id in range(self.num_classes):
            row_metrics = metrics_by_class.get(
                class_id,
                {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0},
            )
            row = {
                "class_name": self.class_names[class_id],
                "images": len(self.class_image_ids[class_id]),
                "instances": self.class_instances[class_id],
                **row_metrics,
            }
            per_class.append(row)
            if self.class_instances[class_id] > 0:
                valid_rows.append(row)

        if valid_rows:
            summary = {
                "class_name": "All",
                "images": self.total_images,
                "instances": sum(int(row["instances"]) for row in valid_rows),
                "precision": float(np.mean([float(row["precision"]) for row in valid_rows])),
                "recall": float(np.mean([float(row["recall"]) for row in valid_rows])),
                "map50": float(np.mean([float(row["map50"]) for row in valid_rows])),
                "map50_95": float(np.mean([float(row["map50_95"]) for row in valid_rows])),
            }
        else:
            summary = {
                "class_name": "All",
                "images": self.total_images,
                "instances": 0,
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }

        return {"summary": summary, "per_class": per_class}


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
    if class_names is None:
        class_names = [f"class_{index}" for index in range(num_classes)]

    accumulator = DetectionMetricsAccumulator(
        num_classes=num_classes,
        class_names=class_names,
        iou_thresholds=iou_thresholds
        if iou_thresholds is not None
        else torch.arange(0.5, 1.0, 0.05, dtype=torch.float32),
    )

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

    model.eval()
    for images, targets in iterable:
        images = images.to(device)
        predictions = model.predict(
            images,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            max_det=max_det,
        )
        accumulator.update(predictions, targets)

    return accumulator.compute()


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

    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
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
