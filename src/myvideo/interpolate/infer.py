"""Frame interpolation inference for frame folders and videos."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from myvideo.interpolate.model import TinyFrameInterpolator
from myvideo.utils.hardware import detect_hardware
from myvideo.utils.perf import clear_device_cache, get_inference_dtype, maybe_autocast



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate frames to higher FPS")
    parser.add_argument("--input-dir", type=str, help="Input frames directory")
    parser.add_argument("--input-video", type=str, help="Optional input video path")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--multiplier", type=int, default=2, help="2 for 12->24fps, 3 for 10->30fps style")
    parser.add_argument("--work-dir", type=str, default="outputs/interpolate_work")
    return parser.parse_args()



def _to_tensor(path: Path) -> torch.Tensor:
    return transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)



def _save_tensor(x: torch.Tensor, path: Path) -> None:
    arr = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)



def _extract_frames(video_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), str(output_dir / "%06d.png")]
    subprocess.run(cmd, check=True)



def main() -> None:
    args = parse_args()
    profile = detect_hardware()
    device = torch.device(profile.device)

    work_dir = Path(args.work_dir)
    frames_dir = Path(args.input_dir) if args.input_dir else work_dir / "input_frames"
    if args.input_video:
        _extract_frames(Path(args.input_video), frames_dir)

    frame_paths = sorted(frames_dir.glob("*.png"))
    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames")

    infer_dtype = get_inference_dtype(profile.device)
    model = TinyFrameInterpolator().to(device=device, dtype=infer_dtype)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode(), maybe_autocast(profile.device, infer_dtype):
        write_index = 0
        for i in range(len(frame_paths) - 1):
            a = _to_tensor(frame_paths[i]).to(device=device, dtype=infer_dtype)
            b = _to_tensor(frame_paths[i + 1]).to(device=device, dtype=infer_dtype)

            _save_tensor(a.cpu(), output_dir / f"{write_index:06d}.png")
            write_index += 1

            for m in range(1, args.multiplier):
                t = torch.tensor([m / args.multiplier], device=device, dtype=infer_dtype)
                mid = model(a, b, t)
                _save_tensor(mid.cpu(), output_dir / f"{write_index:06d}.png")
                write_index += 1

            clear_device_cache(profile.device)

        _save_tensor(_to_tensor(frame_paths[-1]), output_dir / f"{write_index:06d}.png")

    print(f"Saved interpolated frames to {output_dir}")


if __name__ == "__main__":
    main()
