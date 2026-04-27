"""Rule-based timeline planner for 1-minute prompt-to-segment conversion.

This planner intentionally avoids heavy LLM dependencies so it can run on
Apple Silicon with minimal RAM.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from myvideo.planner.schema import TimelinePlan, TimelineSegment

STYLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "anime": ("anime", "manga", "cel-shaded", "cel shaded"),
    "watercolor": ("watercolor", "painterly", "brush", "illustration"),
    "pixel_art": ("pixel", "8-bit", "16-bit", "retro"),
    "cinematic": ("cinematic", "film", "dramatic", "epic"),
    "minimal_flat": ("minimal", "flat", "clean", "vector"),
    "comic": ("comic", "panel", "ink", "halftone"),
}

MOTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "slow_pan": ("pan", "glide", "drift", "landscape"),
    "parallax": ("parallax", "depth", "layers", "foreground"),
    "orbit": ("orbit", "circle", "rotate around"),
    "dolly_in": ("zoom in", "push in", "approach", "dolly in"),
    "dolly_out": ("zoom out", "pull back", "retreat", "dolly out"),
    "action_cuts": ("action", "fight", "chase", "fast", "explosive"),
}

SCENE_TEMPLATES: tuple[str, ...] = (
    "Establish the setting and visual tone from the prompt.",
    "Introduce the main subject and a clear focal composition.",
    "Add a transition beat with environmental movement and depth.",
    "Highlight the most important narrative or visual action.",
    "Resolve with a memorable closing shot that matches the prompt mood.",
    "Insert a connector shot to smooth pacing between major moments.",
    "Use a detail close-up that reinforces style and theme.",
    "Finish with a wide shot that feels complete and calm.",
)


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())



def _pick_style(prompt_lc: str) -> str:
    for style, keywords in STYLE_KEYWORDS.items():
        if any(keyword in prompt_lc for keyword in keywords):
            return style
    return "cinematic"



def _pick_motion(prompt_lc: str, segment_index: int, total_segments: int) -> str:
    for motion, keywords in MOTION_KEYWORDS.items():
        if any(keyword in prompt_lc for keyword in keywords):
            if motion == "action_cuts":
                return "fast cut every 1.5-2.0 seconds with directional movement"
            if motion == "parallax":
                return "multi-layer parallax drift with subtle foreground speed offset"
            return motion.replace("_", " ")

    progress = segment_index / max(total_segments - 1, 1)
    if progress < 0.25:
        return "slow pan left-to-right with slight zoom in"
    if progress < 0.5:
        return "gentle dolly in with mild parallax"
    if progress < 0.75:
        return "horizontal tracking move with occasional hold"
    return "slow dolly out for visual resolution"



def _estimate_segment_count(prompt: str, total_duration_sec: float) -> int:
    words = len(prompt.split())
    base = 6
    if words < 40:
        base = 5
    elif words > 120:
        base = 8
    return max(5, min(8, base, int(math.floor(total_duration_sec / 7.0))))



def _choose_pacing(prompt_lc: str) -> str:
    if any(x in prompt_lc for x in ("fast", "intense", "action", "rapid", "chaotic")):
        return "fast"
    if any(x in prompt_lc for x in ("calm", "meditative", "slow", "peaceful", "ambient")):
        return "slow"
    return "balanced"



def build_timeline_plan(prompt: str, total_duration_sec: float = 60.0) -> TimelinePlan:
    """Convert a 1-minute prompt into a 5-8 segment timeline JSON model."""
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt must not be empty")

    prompt_lc = _normalize_prompt(cleaned_prompt)
    segment_count = _estimate_segment_count(cleaned_prompt, total_duration_sec)
    pacing = _choose_pacing(prompt_lc)
    style = _pick_style(prompt_lc)

    base_duration = total_duration_sec / segment_count
    segments: list[TimelineSegment] = []
    cursor = 0.0

    for idx in range(segment_count):
        start_sec = cursor
        if idx == segment_count - 1:
            end_sec = total_duration_sec
        else:
            end_sec = round(min(total_duration_sec, cursor + base_duration), 3)

        scene_hint = SCENE_TEMPLATES[idx % len(SCENE_TEMPLATES)]
        segment = TimelineSegment(
            segment_id=idx + 1,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            duration_sec=round(end_sec - start_sec, 3),
            scene_description=(
                f"{scene_hint} Prompt focus: {cleaned_prompt[:220]}"
            ),
            style=style,
            motion_plan=_pick_motion(prompt_lc, idx, segment_count),
        )
        segments.append(segment)
        cursor = end_sec

    return TimelinePlan(
        total_duration_sec=total_duration_sec,
        target_segment_count=segment_count,
        pacing=pacing,
        segments=segments,
    )



def plan_to_json(plan: TimelinePlan) -> str:
    """Serialize timeline plan to compact JSON."""
    return json.dumps(plan.model_dump(), indent=2)



def save_timeline_plan(plan: TimelinePlan, output_path: str | Path) -> None:
    """Persist the timeline plan as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan_to_json(plan), encoding="utf-8")
