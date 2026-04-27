"""CLI entrypoints for myvideo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

from myvideo.config.loader import load_config
from myvideo.planner import build_timeline_plan, save_timeline_plan
from myvideo.utils.hardware import detect_hardware


@click.group()
def cli() -> None:
    """myvideo command line tool."""


@cli.command("show-config")
@click.option("--mode", type=click.Choice(["preview", "final", "low_memory"]), default="preview")
def show_config(mode: str) -> None:
    """Print resolved configuration JSON for a given mode."""
    config = load_config(mode=mode)
    hardware = detect_hardware()
    payload = {"hardware": hardware.__dict__, "config": config.model_dump()}
    click.echo(json.dumps(payload, indent=2))


@cli.command("plan")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--duration", type=float, default=60.0)
def plan(prompt_file: Path, output_json: Path, duration: float) -> None:
    """Create timeline segments from a prompt text file."""
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    timeline = build_timeline_plan(prompt=prompt, total_duration_sec=duration)
    save_timeline_plan(timeline, output_json)
    click.echo(f"Saved timeline to {output_json}")


@cli.command("gen")
@click.option("--timeline-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--keyframe-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--inbetween-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def gen(timeline_json: Path, keyframe_checkpoint: Path, inbetween_checkpoint: Path, output_dir: Path) -> None:
    """Generate low-res inbetweened frames from a timeline file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    key_dir = output_dir / "keyframes"
    inb_dir = output_dir / "inbetween"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "myvideo.keyframe.infer",
            "--timeline-json",
            str(timeline_json),
            "--checkpoint",
            str(keyframe_checkpoint),
            "--output-dir",
            str(key_dir),
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
        ],
        check=True,
    )
    click.echo(f"Generated low-res segment data under {output_dir}")


@cli.command("upscale")
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--scale", type=int, default=2)
def upscale(input_dir: Path, checkpoint: Path, output_dir: Path, scale: int) -> None:
    """Upscale a frame directory."""
    from myvideo.pipeline.frame_processing import upscale_frames_low_memory
    upscale_frames_low_memory(input_dir, output_dir, checkpoint, scale=scale)
    click.echo(f"Upscaled frames saved to {output_dir}")


@cli.command("interpolate")
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--multiplier", type=int, default=2)
def interpolate(input_dir: Path, checkpoint: Path, output_dir: Path, multiplier: int) -> None:
    """Interpolate frame directory for higher FPS."""
    from myvideo.pipeline.frame_processing import interpolate_frames_low_memory
    interpolate_frames_low_memory(input_dir, output_dir, checkpoint, multiplier=multiplier)
    click.echo(f"Interpolated frames saved to {output_dir}")


@cli.command("concat")
@click.option("--segments-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-video", type=click.Path(path_type=Path), required=True)
def concat(segments_dir: Path, output_video: Path) -> None:
    """Concatenate segment mp4 files into final video."""
    from myvideo.assemble import concat_videos
    segment_paths = sorted(segments_dir.glob("*.mp4"))
    concat_videos(segment_paths, output_video)
    click.echo(f"Concatenated video saved to {output_video}")


@cli.command("make")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-video", type=click.Path(path_type=Path), required=True)
@click.option("--mode", type=click.Choice(["preview", "final", "low_memory"]), default="low_memory")
@click.option("--keyframe-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--inbetween-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--upscaler-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--interpolator-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--work-dir", type=click.Path(path_type=Path), default=Path("outputs/work"))
def make(
    prompt_file: Path,
    output_video: Path,
    mode: str,
    keyframe_checkpoint: Path,
    inbetween_checkpoint: Path,
    upscaler_checkpoint: Path,
    interpolator_checkpoint: Path,
    work_dir: Path,
) -> None:
    """Run full pipeline from prompt file to final video."""
    from myvideo.pipeline.orchestrator import run_full_pipeline
    final_path = run_full_pipeline(
        prompt_file=prompt_file,
        output_video=output_video,
        mode=mode,
        keyframe_checkpoint=keyframe_checkpoint,
        inbetween_checkpoint=inbetween_checkpoint,
        upscaler_checkpoint=upscaler_checkpoint,
        interpolator_checkpoint=interpolator_checkpoint,
        work_dir=work_dir,
    )
    click.echo(f"Final video generated at {final_path}")


if __name__ == "__main__":
    cli()
