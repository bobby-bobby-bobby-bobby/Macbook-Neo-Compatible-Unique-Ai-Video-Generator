"""Inference script for tiny keyframe generator optimized for low memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from myvideo.keyframe.model import TinyKeyframeGenerator
from myvideo.utils.hardware import detect_hardware



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate keyframes from timeline segments")
    parser.add_argument("--timeline-json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/keyframes")
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--keyframes-per-segment", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(image_np)



def main() -> None:
    args = parse_args()
    hardware = detect_hardware()
    device = torch.device(hardware.device)

    timeline = json.loads(Path(args.timeline_json).read_text(encoding="utf-8"))
    segments = timeline["segments"]

    model = TinyKeyframeGenerator().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for segment in segments:
            segment_id = segment["segment_id"]
            prompt = f"{segment['style']}. {segment['scene_description']}. Motion: {segment['motion_plan']}"

            for key_idx in range(args.keyframes_per_segment):
                seed = args.seed + segment_id * 100 + key_idx
                image = model.generate(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    steps=args.steps,
                    seed=seed,
                    device=device,
                )
                pil_img = _tensor_to_pil(image)
                out_path = output_dir / f"segment_{segment_id:02d}_key_{key_idx:02d}.png"
                pil_img.save(out_path)

                if hardware.device == "mps":
                    torch.mps.empty_cache()
                elif hardware.device == "cuda":
                    torch.cuda.empty_cache()

    print(f"Generated keyframes in {output_dir}")


if __name__ == "__main__":
    main()
