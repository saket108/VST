from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8) -> int:
    return int((value + divisor - 1) // divisor * divisor)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: bool = True,
    ) -> None:
        padding = (kernel_size // 2) * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class EdgeResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 4.0,
    ) -> None:
        super().__init__()
        hidden = make_divisible(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = (
            ConvBNAct(in_channels, hidden, kernel_size=1)
            if hidden != in_channels
            else nn.Identity()
        )
        self.depthwise = ConvBNAct(hidden, hidden, stride=stride, groups=hidden)
        self.se = SqueezeExcite(hidden)
        self.project = ConvBNAct(hidden, out_channels, kernel_size=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.depthwise(y)
        y = self.se(y)
        y = self.project(y)
        if self.use_residual:
            y = y + x
        return F.silu(y, inplace=True)


class ContextBridge(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 16)
        self.local = ConvBNAct(channels, channels, groups=channels)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.mix = ConvBNAct(channels, channels, kernel_size=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        gated = local * self.global_gate(x)
        return F.silu(x + self.mix(gated), inplace=True)


class SpatialChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        padding = kernel_size // 2
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = x * self.channel_gate(x)
        avg = gated.mean(dim=1, keepdim=True)
        mx = gated.max(dim=1, keepdim=True).values
        spatial = self.spatial_gate(torch.cat([avg, mx], dim=1))
        return gated * spatial


class DilatedContextBridge(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = max(channels // 4, 16)
        self.d1 = ConvBNAct(channels, branch_channels, kernel_size=3, groups=branch_channels)
        self.d2 = ConvBNAct(
            channels,
            branch_channels,
            kernel_size=3,
            groups=branch_channels,
            dilation=2,
        )
        self.d4 = ConvBNAct(
            channels,
            branch_channels,
            kernel_size=3,
            groups=branch_channels,
            dilation=4,
        )
        self.fuse = ConvBNAct(branch_channels * 3, channels, kernel_size=1, act=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([self.d1(x), self.d2(x), self.d4(x)], dim=1)
        refined = self.fuse(multi_scale)
        return F.silu(x + refined * self.gate(x), inplace=True)


class DetailStem(nn.Sequential):
    def __init__(self, out_channels: int) -> None:
        hidden = max(out_channels // 2, 16)
        super().__init__(
            ConvBNAct(3, hidden, stride=2),
            ConvBNAct(hidden, out_channels),
            ContextBridge(out_channels),
        )


class WeightedFeatureFusion(nn.Module):
    def __init__(self, channels: int, inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))
        self.project = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-4)
        fused = sum(weight * feature for weight, feature in zip(weights, features))
        return self.project(fused)


class ContextAwareFusion(nn.Module):
    def __init__(self, channels: int, inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))
        hidden = max(channels // 4, 16)
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.local = ConvBNAct(channels, channels, groups=channels)
        self.mix = ConvBNAct(channels, channels, kernel_size=1, act=False)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-4)
        fused = sum(weight * feature for weight, feature in zip(weights, features))
        gated = fused * self.context_gate(fused)
        refined = self.local(gated) + fused
        return F.silu(self.mix(refined) + fused, inplace=True)


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
