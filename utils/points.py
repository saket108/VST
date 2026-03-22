from __future__ import annotations

import torch


def build_points(
    height: int,
    width: int,
    stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    shifts_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
    shifts_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
    yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
