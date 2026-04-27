"""Tiny latent keyframe generator network.

Design goals:
- Keep parameter count small
- Work at low resolution (e.g., 360p proxy or smaller training crops)
- Support quantization-friendly modules
"""

from __future__ import annotations

import torch
from torch import nn


class TinyTextEmbedder(nn.Module):
    """Hash-based prompt embedder using character ids.

    This avoids large tokenizer dependencies and keeps memory tiny.
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, max_len: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        raw = text.encode("utf-8", errors="ignore")[: self.max_len]
        if not raw:
            raw = b" "
        ids = torch.tensor(list(raw), dtype=torch.long, device=device).clamp_(0, self.vocab_size - 1)
        emb = self.embed(ids)
        pooled = emb.mean(dim=0)
        return self.proj(pooled)


class TinyDenoiser(nn.Module):
    """Small denoiser operating in latent space."""

    def __init__(self, latent_channels: int = 4, cond_dim: int = 64) -> None:
        super().__init__()
        in_channels = latent_channels + cond_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
        )
        self.cond_dim = cond_dim

    def forward(self, noisy_latent: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        b, _, h, w = noisy_latent.shape
        cond_map = text_cond.view(b, self.cond_dim, 1, 1).expand(b, self.cond_dim, h, w)
        x = torch.cat([noisy_latent, cond_map], dim=1)
        return self.net(x)


class TinyDecoder(nn.Module):
    """Decode latent to RGB image."""

    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class TinyKeyframeGenerator(nn.Module):
    """Full tiny latent keyframe model."""

    def __init__(self, latent_channels: int = 4, text_dim: int = 64) -> None:
        super().__init__()
        self.text = TinyTextEmbedder(embed_dim=text_dim)
        self.denoiser = TinyDenoiser(latent_channels=latent_channels, cond_dim=text_dim)
        self.decoder = TinyDecoder(latent_channels=latent_channels)
        self.latent_channels = latent_channels

    def denoise_step(self, noisy_latent: torch.Tensor, prompt: str) -> torch.Tensor:
        text_cond = self.text.encode_text(prompt, noisy_latent.device).unsqueeze(0).expand(noisy_latent.size(0), -1)
        predicted_noise = self.denoiser(noisy_latent, text_cond)
        return noisy_latent - predicted_noise

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 180,
        width: int = 320,
        steps: int = 12,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        device_obj = torch.device(device)
        generator = torch.Generator(device=device_obj)
        generator.manual_seed(seed)

        latent = torch.randn(
            1,
            self.latent_channels,
            height,
            width,
            generator=generator,
            device=device_obj,
        )

        for _ in range(steps):
            latent = self.denoise_step(latent, prompt)

        image = self.decoder(latent).clamp(0.0, 1.0)
        return image
