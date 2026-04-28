"""Hardware and environment detection utilities.

This module centralizes environment and device selection so all training and
inference scripts can rely on the same logic for:
- Google Colab with CUDA GPU
- Google Colab CPU fallback
- Apple Silicon Mac (MPS/CPU)
- Generic Linux/Windows CPU fallback
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardwareProfile:
    """Represents detected runtime environment and preferred device."""

    environment: str
    os_name: str
    machine: str
    has_cuda: bool
    has_mps: bool
    device: str
    free_mem_hint_gb: float



def _is_colab() -> bool:
    """Return True when running in Google Colab."""
    return "COLAB_GPU" in os.environ or "google.colab" in os.environ.get("PYTHONPATH", "")



def _is_apple_silicon() -> bool:
    """Return True when host appears to be Apple Silicon."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    return system == "darwin" and machine in {"arm64", "aarch64"}



def detect_hardware() -> HardwareProfile:
    """Detect runtime hardware and return a normalized profile.

    Device preference order:
    1. CUDA (Colab GPU / NVIDIA)
    2. MPS (Apple Silicon Metal)
    3. CPU
    """
    has_cuda = bool(torch.cuda.is_available())
    has_mps = bool(torch.backends.mps.is_available()) if hasattr(torch.backends, "mps") else False

    if _is_colab() and has_cuda:
        environment = "colab_gpu"
        device = "cuda"
        free_mem_hint_gb = 12.0
    elif _is_colab() and not has_cuda:
        environment = "colab_cpu"
        device = "cpu"
        free_mem_hint_gb = 6.0
    elif _is_apple_silicon() and has_mps:
        environment = "mac_apple_silicon_mps"
        device = "mps"
        free_mem_hint_gb = 4.0
    elif _is_apple_silicon() and not has_mps:
        environment = "mac_apple_silicon_cpu"
        device = "cpu"
        free_mem_hint_gb = 4.0
    else:
        environment = "generic_cpu_or_other_gpu"
        device = "cuda" if has_cuda else "cpu"
        free_mem_hint_gb = 8.0 if has_cuda else 4.0

    return HardwareProfile(
        environment=environment,
        os_name=platform.system(),
        machine=platform.machine(),
        has_cuda=has_cuda,
        has_mps=has_mps,
        device=device,
        free_mem_hint_gb=free_mem_hint_gb,
    )
