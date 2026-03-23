from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import TYPE_CHECKING, Callable

import torch

from utils.torch_utils import format_device_memory

if TYPE_CHECKING:
    from model.detector import VSTDet
    from training.losses import DetectionLoss


@dataclass
class AutoBatchResult:
    batch_size: int
    tried: list[tuple[int, bool]]
    device_memory: str
    trainable_backbone: bool


def _move_targets_to_device(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def _free_cuda_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _build_probe_batch(
    dataset,
    collate: Callable,
    batch_size: int,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    sample_count = len(dataset)
    if sample_count == 0:
        raise ValueError("Dataset is empty; autobatch cannot build a probe batch.")
    samples = [dataset[index % sample_count] for index in range(batch_size)]
    return collate(samples)


def _is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _can_run_batch_size(
    model: VSTDet,
    criterion: DetectionLoss,
    dataset,
    collate: Callable,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
) -> bool:
    images = None
    targets = None
    outputs = None
    losses = None
    was_training = model.training
    model.eval()
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        images, targets = _build_probe_batch(dataset, collate, batch_size)
        images = images.to(device, non_blocking=False)
        targets = _move_targets_to_device(targets, device)

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad = None

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            losses = criterion(outputs, targets)
        losses.total.backward()
        return True
    except RuntimeError as error:
        if not _is_oom_error(error):
            raise
        return False
    finally:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad = None
        del images, targets, outputs, losses
        _free_cuda_memory(device)
        if was_training:
            model.train()


def estimate_autobatch_size(
    model: VSTDet,
    criterion: DetectionLoss,
    dataset,
    collate: Callable,
    device: torch.device,
    max_batch_size: int,
    amp_enabled: bool,
    trainable_backbone: bool,
) -> AutoBatchResult:
    if device.type != "cuda":
        return AutoBatchResult(
            batch_size=1,
            tried=[],
            device_memory=format_device_memory(device),
            trainable_backbone=trainable_backbone,
        )
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1.")

    model.set_backbone_trainable(trainable_backbone)

    tried: list[tuple[int, bool]] = []
    low = 1
    high = 1
    best = 0

    while high <= max_batch_size:
        success = _can_run_batch_size(
            model=model,
            criterion=criterion,
            dataset=dataset,
            collate=collate,
            device=device,
            batch_size=high,
            amp_enabled=amp_enabled,
        )
        tried.append((high, success))
        if not success:
            break
        best = high
        low = high + 1
        high *= 2

    if best == 0:
        raise RuntimeError("Autobatch failed even for batch size 1.")

    upper = min(high - 1, max_batch_size)
    while low <= upper:
        mid = (low + upper) // 2
        success = _can_run_batch_size(
            model=model,
            criterion=criterion,
            dataset=dataset,
            collate=collate,
            device=device,
            batch_size=mid,
            amp_enabled=amp_enabled,
        )
        tried.append((mid, success))
        if success:
            best = mid
            low = mid + 1
        else:
            upper = mid - 1

    return AutoBatchResult(
        batch_size=best,
        tried=tried,
        device_memory=format_device_memory(device),
        trainable_backbone=trainable_backbone,
    )
