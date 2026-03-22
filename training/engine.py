from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.detector import VSTDet
from training.losses import DetectionLoss
from utils.evaluator import evaluate_detection_metrics


def move_targets_to_device(
    targets: list[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def format_device_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "0G"
    memory_gb = torch.cuda.memory_reserved(device=device) / (1024**3)
    return f"{memory_gb:.2f}G"


def train_one_epoch(
    model: VSTDet,
    loader: DataLoader,
    criterion: DetectionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    clip_grad: float,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.train()
    loss_sums = {"total": 0.0, "cls": 0.0, "box": 0.0, "center": 0.0}
    total_batches = 0

    progress = tqdm(
        loader,
        total=len(loader),
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            losses = criterion(outputs, targets)

        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        scaler.step(optimizer)
        scaler.update()

        loss_sums["total"] += float(losses.total.item())
        loss_sums["cls"] += float(losses.cls.item())
        loss_sums["box"] += float(losses.box.item())
        loss_sums["center"] += float(losses.center.item())
        total_batches += 1

        avg_box = loss_sums["box"] / total_batches
        avg_cls = loss_sums["cls"] / total_batches
        avg_center = loss_sums["center"] / total_batches
        instances = sum(int(target["labels"].numel()) for target in targets)
        size = int(images.shape[-1])
        progress.set_description(
            f"{epoch:>9}/{epochs:<3} {format_device_memory(device):>9} "
            f"{avg_box:>10.3f} {avg_cls:>10.3f} {avg_center:>10.3f} "
            f"{instances:>10} {size:>10}"
        )

    return {key: value / max(total_batches, 1) for key, value in loss_sums.items()}


@torch.no_grad()
def validate(
    model: VSTDet,
    loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    class_names: list[str],
    conf_threshold: float,
    nms_iou: float,
    max_det: int,
) -> tuple[dict[str, float], dict[str, object]]:
    model.eval()
    loss_sums = {"val_total": 0.0, "val_cls": 0.0, "val_box": 0.0, "val_center": 0.0}
    total_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss_sums["val_total"] += float(losses.total.item())
        loss_sums["val_cls"] += float(losses.cls.item())
        loss_sums["val_box"] += float(losses.box.item())
        loss_sums["val_center"] += float(losses.center.item())
        total_batches += 1

    metrics = {key: value / max(total_batches, 1) for key, value in loss_sums.items()}
    report = evaluate_detection_metrics(
        model=model,
        loader=loader,
        device=device,
        num_classes=model.num_classes,
        class_names=class_names,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou,
        max_det=max_det,
        show_progress=True,
        progress_desc="                 Class     Images  Instances  Precision     Recall      mAP50  mAP50-95",
    )
    summary = report["summary"]
    metrics.update(
        {
            "precision": float(summary["precision"]),
            "recall": float(summary["recall"]),
            "map50": float(summary["map50"]),
            "map50_95": float(summary["map50_95"]),
        }
    )
    return metrics, report


def save_checkpoint(
    path: Path,
    model: VSTDet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "variant": model.variant,
            "backbone_name": model.backbone_name,
            "pretrained_backbone": model.pretrained_backbone,
            "num_classes": model.num_classes,
            "names": names,
        },
        path,
    )


def append_history(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
