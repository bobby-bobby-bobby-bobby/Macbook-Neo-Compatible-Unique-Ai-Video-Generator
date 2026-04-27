"""FFmpeg helper functions for assembling videos and concatenating segments."""

from __future__ import annotations

import subprocess
from pathlib import Path



def frames_to_video(frames_dir: str | Path, output_video: str | Path, fps: int = 24, codec: str = "libx264") -> None:
    frames_dir = Path(frames_dir)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)



def concat_videos(video_paths: list[str | Path], output_video: str | Path) -> None:
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_video.parent / "concat_list.txt"
    lines = []
    for path in video_paths:
        lines.append(f"file '{Path(path).resolve()}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)



def extract_frames(video_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), str(output_dir / "%06d.png")]
    subprocess.run(cmd, check=True)
