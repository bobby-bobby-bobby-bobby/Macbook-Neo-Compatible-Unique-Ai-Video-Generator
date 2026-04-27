"""Segment-based low-memory inference for tiny inbetweener."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from myvideo.inbetween.model import TinyInbetweener
from myvideo.utils.hardware import detect_hardware



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inbetween frames from keyframe pairs")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with segment_*_key_*.png files")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/inbetween")
    parser.add_argument("--frames-between", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=180)
    return parser.parse_args()



def _load_img(path: Path, image_size: int) -> torch.Tensor:
    tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)



def _save_img(tensor: torch.Tensor, path: Path) -> None:
    arr = (tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)



def main() -> None:
    args = parse_args()
    profile = detect_hardware()
    device = torch.device(profile.device)

    model = TinyInbetweener().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    in_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keyframes = sorted(in_dir.glob("segment_*_key_*.png"))
    if len(keyframes) < 2:
        raise ValueError("Need at least two keyframes to run inbetweening")

    with torch.no_grad():
        for idx in range(len(keyframes) - 1):
            a_path = keyframes[idx]
            b_path = keyframes[idx + 1]
            frame_a = _load_img(a_path, args.image_size).to(device)
            frame_b = _load_img(b_path, args.image_size).to(device)

            pair_dir = output_dir / f"pair_{idx:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            _save_img(frame_a, pair_dir / "0000.png")

            for j in range(1, args.frames_between + 1):
                t_val = j / (args.frames_between + 1)
                t = torch.tensor([t_val], device=device)
                mid = model(frame_a, frame_b, t)
                _save_img(mid, pair_dir / f"{j:04d}.png")

            _save_img(frame_b, pair_dir / f"{args.frames_between + 1:04d}.png")

            if profile.device == "mps":
                torch.mps.empty_cache()
            elif profile.device == "cuda":
                torch.cuda.empty_cache()

    print(f"Saved inbetweened segments to {output_dir}")


if __name__ == "__main__":
    main()
