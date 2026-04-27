"""Tiny super-resolution model for per-frame upscaling."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Simple residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(x)))
        return x + residual


class TinyUpscaler(nn.Module):
    """Tiny x2/x3 SR model using pixel shuffle."""

    def __init__(self, scale: int = 2, channels: int = 32, num_blocks: int = 4) -> None:
        super().__init__()
        if scale not in {2, 3}:
            raise ValueError("scale must be 2 or 3")

        self.head = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, 3 * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Sigmoid(),
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        features = self.body(features)
        out = self.tail(features)
        return torch.clamp(out, 0.0, 1.0)
