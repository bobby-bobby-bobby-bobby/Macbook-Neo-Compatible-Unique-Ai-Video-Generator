"""Segment-based low-memory inference for tiny inbetweener."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from myvideo.inbetween.model import TinyInbetweener
from myvideo.utils.hardware import detect_hardware
from myvideo.utils.perf import clear_device_cache, get_inference_dtype, maybe_autocast



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inbetween frames from keyframe pairs")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with segment_*_key_*.png files")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/inbetween")
    parser.add_argument("--frames-between", type=int, default=4)
    parser.add_argument("--height", type=int, default=180, help="Frame height in pixels")
    parser.add_argument("--width", type=int, default=320, help="Frame width in pixels")
    return parser.parse_args()



def _load_img(path: Path, height: int, width: int) -> torch.Tensor:
    tf = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)



def _save_img(tensor: torch.Tensor, path: Path) -> None:
    arr = (tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)



_KEYFRAME_RE = re.compile(r"^segment_(\d+)_key_(\d+)\.png$")


def _group_keyframes_by_segment(in_dir: Path) -> dict[int, list[Path]]:
    """Return keyframes grouped by segment id, sorted by segment id then keyframe index."""
    groups: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for p in in_dir.glob("segment_*_key_*.png"):
        m = _KEYFRAME_RE.match(p.name)
        if m:
            seg_id = int(m.group(1))
            key_idx = int(m.group(2))
            groups[seg_id].append((key_idx, p))
    return {seg_id: [p for _, p in sorted(items)] for seg_id, items in sorted(groups.items())}


def main() -> None:
    args = parse_args()
    profile = detect_hardware()
    device = torch.device(profile.device)

    infer_dtype = get_inference_dtype(profile.device)
    model = TinyInbetweener().to(device=device, dtype=infer_dtype)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    in_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = _group_keyframes_by_segment(in_dir)
    if not segments:
        raise ValueError("No keyframes matching segment_*_key_*.png found in --input-dir")

    pair_global_idx = 0
    with torch.inference_mode(), maybe_autocast(profile.device, infer_dtype):
        for seg_id, keyframes in segments.items():
            if len(keyframes) < 2:
                continue
            for idx in range(len(keyframes) - 1):
                a_path = keyframes[idx]
                b_path = keyframes[idx + 1]
                frame_a = _load_img(a_path, args.height, args.width).to(device=device, dtype=infer_dtype)
                frame_b = _load_img(b_path, args.height, args.width).to(device=device, dtype=infer_dtype)

                pair_dir = output_dir / f"pair_{pair_global_idx:04d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                _save_img(frame_a, pair_dir / "0000.png")

                for j in range(1, args.frames_between + 1):
                    t_val = j / (args.frames_between + 1)
                    t = torch.tensor([t_val], device=device, dtype=infer_dtype)
                    mid = model(frame_a, frame_b, t)
                    _save_img(mid, pair_dir / f"{j:04d}.png")

                _save_img(frame_b, pair_dir / f"{args.frames_between + 1:04d}.png")

                clear_device_cache(profile.device)
                pair_global_idx += 1

    print(f"Saved inbetweened segments to {output_dir}")


if __name__ == "__main__":
    main()
