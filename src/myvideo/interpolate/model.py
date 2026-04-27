"""Tiny frame-rate interpolation model for 12fps -> 24/30fps."""

from __future__ import annotations

import torch
from torch import nn


class TinyFrameInterpolator(nn.Module):
    """Predict intermediate frame given adjacent frames and time scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, _, h, w = frame_a.shape
        t_map = t.view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([frame_a, frame_b, t_map], dim=1)
        pred = self.net(x)
        blend = (1.0 - t_map) * frame_a + t_map * frame_b
        return torch.clamp(0.6 * blend + 0.4 * pred, 0.0, 1.0)
