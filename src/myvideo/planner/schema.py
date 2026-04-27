"""Schemas for timeline planning outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TimelineSegment(BaseModel):
    """Single planned segment for the 1-minute timeline."""

    segment_id: int = Field(ge=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    duration_sec: float = Field(gt=0)
    scene_description: str
    style: str
    motion_plan: str


class TimelinePlan(BaseModel):
    """Timeline plan containing 5-8 segments."""

    total_duration_sec: float = Field(default=60.0, gt=0)
    target_segment_count: int = Field(ge=5, le=8)
    pacing: Literal["slow", "balanced", "fast"] = "balanced"
    segments: list[TimelineSegment]
