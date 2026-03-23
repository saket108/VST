from __future__ import annotations

import random

import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def resolve_device(device_name: str | None = None) -> torch.device:
    requested = (device_name or "").strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def format_device_name(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_name(index)


def format_device_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "0G"
    memory_gb = torch.cuda.memory_reserved(device=device) / (1024**3)
    return f"{memory_gb:.2f}G"


def parameter_count(model: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def model_summary(model: torch.nn.Module) -> str:
    total = parameter_count(model, trainable_only=False)
    trainable = parameter_count(model, trainable_only=True)
    return (
        f"params={format_parameter_count(total)} "
        f"trainable={format_parameter_count(trainable)}"
    )
