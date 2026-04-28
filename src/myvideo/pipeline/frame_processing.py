"""Low-memory frame processing pipeline for upscaling and interpolation."""

from __future__ import annotations

from pathlib import Path

import torch

from myvideo.interpolate.model import TinyFrameInterpolator
from myvideo.upscale.model import TinyUpscaler
from myvideo.utils.hardware import detect_hardware
from myvideo.utils.perf import clear_device_cache, get_inference_dtype, maybe_autocast



def upscale_frames_low_memory(
    input_dir: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    scale: int = 2,
) -> None:
    """Upscale frames one-by-one to minimize memory usage."""
    from PIL import Image
    from torchvision import transforms

    profile = detect_hardware()
    device = torch.device(profile.device)

    infer_dtype = get_inference_dtype(profile.device)
    model = TinyUpscaler(scale=scale).to(device=device, dtype=infer_dtype)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    tf = transforms.ToTensor()
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    with torch.inference_mode(), maybe_autocast(profile.device, infer_dtype):
        for idx, path in enumerate(frames):
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device=device, dtype=infer_dtype)
            y = model(x)
            arr = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            Image.fromarray(arr).save(out_dir / f"{idx:06d}.png")

            clear_device_cache(profile.device)



def interpolate_frames_low_memory(
    input_dir: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    multiplier: int = 2,
) -> None:
    """Interpolate adjacent frames incrementally for low memory footprint."""
    from PIL import Image
    from torchvision import transforms

    profile = detect_hardware()
    device = torch.device(profile.device)

    infer_dtype = get_inference_dtype(profile.device)
    model = TinyFrameInterpolator().to(device=device, dtype=infer_dtype)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    tf = transforms.ToTensor()
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(in_dir.glob("*.png"))

    def save_tensor(x: torch.Tensor, p: Path) -> None:
        arr = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(p)

    with torch.inference_mode(), maybe_autocast(profile.device, infer_dtype):
        out_idx = 0
        for i in range(len(frames) - 1):
            a = tf(Image.open(frames[i]).convert("RGB")).unsqueeze(0).to(device=device, dtype=infer_dtype)
            b = tf(Image.open(frames[i + 1]).convert("RGB")).unsqueeze(0).to(device=device, dtype=infer_dtype)

            save_tensor(a.cpu(), out_dir / f"{out_idx:06d}.png")
            out_idx += 1

            for m in range(1, multiplier):
                t = torch.tensor([m / multiplier], device=device, dtype=infer_dtype)
                mid = model(a, b, t)
                save_tensor(mid.cpu(), out_dir / f"{out_idx:06d}.png")
                out_idx += 1

            clear_device_cache(profile.device)

        if frames:
            last = tf(Image.open(frames[-1]).convert("RGB")).unsqueeze(0)
            save_tensor(last, out_dir / f"{out_idx:06d}.png")
