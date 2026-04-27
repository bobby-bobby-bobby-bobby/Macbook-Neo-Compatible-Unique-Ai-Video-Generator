"""Quantization/export for tiny frame interpolator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from myvideo.interpolate.model import TinyFrameInterpolator



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tiny frame interpolator")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp16-output", default="models/quantized/interpolator_tiny_fp16.pt")
    parser.add_argument("--script-output", default="models/exports/interpolator_tiny_scripted.pt")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    model = TinyFrameInterpolator().eval()
    model.load_state_dict(state["model"])

    fp16_model = TinyFrameInterpolator().half().eval()
    fp16_model.load_state_dict(state["model"], strict=False)
    fp16_path = Path(args.fp16_output)
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": fp16_model.state_dict(), "dtype": "float16"}, fp16_path)

    ex_a = torch.rand(1, 3, 180, 320)
    ex_b = torch.rand(1, 3, 180, 320)
    ex_t = torch.tensor([0.5])
    scripted = torch.jit.trace(model, (ex_a, ex_b, ex_t))
    script_path = Path(args.script_output)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(script_path))

    print(f"Saved fp16 interpolator to {fp16_path}")
    print(f"Saved scripted interpolator to {script_path}")


if __name__ == "__main__":
    main()
