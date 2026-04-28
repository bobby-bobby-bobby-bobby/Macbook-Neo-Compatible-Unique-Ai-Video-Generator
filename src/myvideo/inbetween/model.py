"""Tiny frame inbetweener model.

Input:
- start frame (RGB)
- end frame (RGB)
- normalized time t in [0, 1]
Output:
- predicted intermediate frame
"""

from __future__ import annotations

import torch
from torch import nn


class TinyInbetweener(nn.Module):
    """Small CNN for temporal interpolation between keyframes."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, _, h, w = frame_a.shape
        t_map = t.view(b, 1, 1, 1).expand(b, 1, h, w)

        linear_blend = (1.0 - t_map) * frame_a + t_map * frame_b
        x = torch.cat([frame_a, frame_b, t_map], dim=1)
        residual = self.decoder(self.encoder(x))

        return torch.clamp(0.7 * linear_blend + 0.3 * residual, 0.0, 1.0)
