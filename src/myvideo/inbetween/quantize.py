"""Quantization/export utilities for tiny inbetweener."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from myvideo.inbetween.model import TinyInbetweener



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize tiny inbetweener")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fp16-output", type=str, default="models/quantized/inbetween_tiny_fp16.pt")
    parser.add_argument("--script-output", type=str, default="models/exports/inbetween_tiny_scripted.pt")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")

    model = TinyInbetweener().eval()
    model.load_state_dict(state["model"])

    fp16 = TinyInbetweener().half().eval()
    fp16.load_state_dict(state["model"], strict=False)
    fp16_path = Path(args.fp16_output)
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": fp16.state_dict(), "dtype": "float16"}, fp16_path)

    example_a = torch.rand(1, 3, 180, 180)
    example_b = torch.rand(1, 3, 180, 180)
    example_t = torch.tensor([0.5])
    scripted = torch.jit.trace(model, (example_a, example_b, example_t))
    script_path = Path(args.script_output)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(script_path))

    print(f"Saved fp16 weights to {fp16_path}")
    print(f"Saved TorchScript model to {script_path}")


if __name__ == "__main__":
    main()
