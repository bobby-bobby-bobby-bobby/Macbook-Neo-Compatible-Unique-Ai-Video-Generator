"""Per-frame upscaling inference with low memory usage."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from myvideo.upscale.model import TinyUpscaler
from myvideo.utils.hardware import detect_hardware
from myvideo.utils.perf import clear_device_cache, get_inference_dtype, maybe_autocast



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale frames directory")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()



def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)



def main() -> None:
    args = parse_args()
    profile = detect_hardware()
    device = torch.device(profile.device)

    infer_dtype = get_inference_dtype(profile.device)
    model = TinyUpscaler(scale=args.scale).to(device=device, dtype=infer_dtype)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.ToTensor()
    frames = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    with torch.inference_mode(), maybe_autocast(profile.device, infer_dtype):
        for idx, frame_path in enumerate(frames):
            frame = Image.open(frame_path).convert("RGB")
            x = tf(frame).unsqueeze(0).to(device=device, dtype=infer_dtype)
            y = model(x)
            _tensor_to_pil(y.detach().cpu()).save(out_dir / f"{idx:06d}.png")

            clear_device_cache(profile.device)

    print(f"Upscaled {len(frames)} frames to {out_dir}")


if __name__ == "__main__":
    main()
