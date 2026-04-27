"""Typed configuration schema for myvideo presets."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """Runtime and performance settings."""

    mode: str = Field(description="Preset mode name: preview, final, or low_memory")
    device_preference: str = Field(default="auto", description="auto, cuda, mps, or cpu")
    batch_size: int = Field(default=1, ge=1)
    seed: int = Field(default=42)
    use_mixed_precision: bool = Field(default=True)
    enable_compile: bool = Field(default=False)


class VideoConfig(BaseModel):
    """Video generation controls."""

    total_duration_sec: int = Field(default=60, ge=10)
    target_fps: int = Field(default=24, ge=8)
    base_width: int = Field(default=640, ge=256)
    base_height: int = Field(default=360, ge=256)
    upscaled_width: int = Field(default=1280, ge=256)
    upscaled_height: int = Field(default=720, ge=256)


class MemoryConfig(BaseModel):
    """Memory-related safeguards for 8GB unified memory."""

    max_frames_in_memory: int = Field(default=8, ge=1)
    stream_chunks: bool = Field(default=True)
    offload_to_cpu_when_idle: bool = Field(default=True)
    clear_cache_between_segments: bool = Field(default=True)


class PipelineConfig(BaseModel):
    """Top-level config object used by all pipeline stages."""

    runtime: RuntimeConfig
    video: VideoConfig
    memory: MemoryConfig
