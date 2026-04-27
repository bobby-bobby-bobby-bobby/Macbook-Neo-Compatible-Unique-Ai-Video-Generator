"""Post-training quantization/export for tiny keyframe generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from myvideo.keyframe.model import TinyKeyframeGenerator



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize tiny keyframe model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/quantized/keyframe_tiny_int8.pt")
    parser.add_argument("--fp16-output", type=str, default="models/quantized/keyframe_tiny_fp16.pt")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    model = TinyKeyframeGenerator()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": quantized.state_dict(), "quantized": True}, output_path)

    fp16_model = TinyKeyframeGenerator().half().eval()
    fp16_model.load_state_dict(state["model"], strict=False)
    fp16_output_path = Path(args.fp16_output)
    fp16_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": fp16_model.state_dict(), "dtype": "float16"}, fp16_output_path)

    print(f"Saved int8(dynamic) checkpoint to {output_path}")
    print(f"Saved fp16 checkpoint to {fp16_output_path}")


if __name__ == "__main__":
    main()
