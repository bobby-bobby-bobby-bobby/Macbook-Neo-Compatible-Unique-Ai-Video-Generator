"""End-to-end orchestration utilities for CLI commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from myvideo.assemble.ffmpeg_tools import concat_videos, frames_to_video
from myvideo.config.loader import load_config
from myvideo.pipeline.frame_processing import interpolate_frames_low_memory, upscale_frames_low_memory
from myvideo.planner import build_timeline_plan, save_timeline_plan



def make_timeline(prompt_text: str, output_json: str | Path, duration: float = 60.0) -> Path:
    plan = build_timeline_plan(prompt=prompt_text, total_duration_sec=duration)
    save_timeline_plan(plan, output_json)
    return Path(output_json)



def build_prompt_from_segment(segment: dict) -> str:
    return f"{segment['style']}. {segment['scene_description']}. Motion: {segment['motion_plan']}"



def run_full_pipeline(
    prompt_file: str | Path,
    output_video: str | Path,
    mode: str,
    keyframe_checkpoint: str | Path,
    inbetween_checkpoint: str | Path,
    upscaler_checkpoint: str | Path,
    interpolator_checkpoint: str | Path,
    work_dir: str | Path = "outputs/work",
) -> Path:
    """Run make pipeline (prompt -> timeline -> keyframes -> inbetween -> upscale -> interpolate -> concat)."""
    # lazy imports to avoid circular CLI/module loading
    import subprocess
    import sys

    config = load_config(mode)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = Path(prompt_file).read_text(encoding="utf-8").strip()
    timeline_path = make_timeline(prompt_text, work_dir / "timeline.json", duration=config.video.total_duration_sec)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))

    segment_videos: list[Path] = []
    for segment in timeline["segments"]:
        seg_id = int(segment["segment_id"])
        seg_dir = work_dir / f"segment_{seg_id:02d}"
        key_dir = seg_dir / "keyframes"
        inb_dir = seg_dir / "inbetween"
        up_dir = seg_dir / "upscaled"
        int_dir = seg_dir / "interpolated"
        seg_dir.mkdir(parents=True, exist_ok=True)

        temp_timeline = {"segments": [segment]}
        temp_timeline_path = seg_dir / "timeline_single.json"
        temp_timeline_path.write_text(json.dumps(temp_timeline), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "myvideo.keyframe.infer",
                "--timeline-json",
                str(temp_timeline_path),
                "--checkpoint",
                str(keyframe_checkpoint),
                "--output-dir",
                str(key_dir),
                "--height",
                str(config.video.base_height),
                "--width",
                str(config.video.base_width),
                "--keyframes-per-segment",
                "2",
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "myvideo.inbetween.infer",
                "--input-dir",
                str(key_dir),
                "--checkpoint",
                str(inbetween_checkpoint),
                "--output-dir",
                str(inb_dir),
                "--frames-between",
                "4",
                "--height",
                str(config.video.base_height),
                "--width",
                str(config.video.base_width),
            ],
            check=True,
        )

        # flatten pair folders to ordered frame folder, skipping repeated boundaries
        flat_dir = seg_dir / "frames_flat"
        flat_dir.mkdir(parents=True, exist_ok=True)
        frame_idx = 0
        pair_dirs = sorted(inb_dir.glob("pair_*"))
        for pair_i, pair_dir in enumerate(pair_dirs):
            pair_frames = sorted(pair_dir.glob("*.png"))
            if pair_i > 0:
                pair_frames = pair_frames[1:]
            for frame_path in pair_frames:
                target = flat_dir / f"{frame_idx:06d}.png"
                shutil.copy2(frame_path, target)
                frame_idx += 1

        upscale_scale = max(2, min(3, round(config.video.upscaled_width / max(1, config.video.base_width))))
        fps_multiplier = max(2, round(config.video.target_fps / 12))
        upscale_frames_low_memory(flat_dir, up_dir, upscaler_checkpoint, scale=upscale_scale)
        interpolate_frames_low_memory(up_dir, int_dir, interpolator_checkpoint, multiplier=fps_multiplier)

        seg_video = seg_dir / "segment.mp4"
        frames_to_video(int_dir, seg_video, fps=config.video.target_fps)
        segment_videos.append(seg_video)

    output_video = Path(output_video)
    concat_videos(segment_videos, output_video)
    return output_video
