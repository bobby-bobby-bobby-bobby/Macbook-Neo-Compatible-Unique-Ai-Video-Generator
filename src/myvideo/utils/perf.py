"""Performance helpers for low-memory inference and training."""

from __future__ import annotations

from contextlib import nullcontext

import torch



def get_inference_dtype(device: str, prefer_fp16: bool = True) -> torch.dtype:
    """Choose best inference dtype for device."""
    if prefer_fp16 and device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32



def maybe_autocast(device: str, dtype: torch.dtype):
    """Return autocast context when supported, else nullcontext."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device == "cpu" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cpu", dtype=dtype)
    return nullcontext()



def clear_device_cache(device: str) -> None:
    """Clear accelerator cache when available."""
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
