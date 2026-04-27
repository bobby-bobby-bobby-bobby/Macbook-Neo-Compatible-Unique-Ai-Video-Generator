"""Quantization/export for tiny upscaler."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from myvideo.upscale.model import TinyUpscaler



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize/export tiny upscaler")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp16-output", default="models/quantized/upscaler_tiny_fp16.pt")
    parser.add_argument("--script-output", default="models/exports/upscaler_tiny_scripted.pt")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    scale = int(state.get("scale", 2))

    model = TinyUpscaler(scale=scale).eval()
    model.load_state_dict(state["model"])

    fp16_model = TinyUpscaler(scale=scale).half().eval()
    fp16_model.load_state_dict(state["model"], strict=False)
    fp16_path = Path(args.fp16_output)
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": fp16_model.state_dict(), "dtype": "float16", "scale": scale}, fp16_path)

    example = torch.rand(1, 3, 180, 320)
    scripted = torch.jit.trace(model, example)
    script_path = Path(args.script_output)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(script_path))

    print(f"Saved fp16 checkpoint to {fp16_path}")
    print(f"Saved TorchScript model to {script_path}")


if __name__ == "__main__":
    main()
