"""Basic tests for timeline planner invariants."""

from myvideo.planner import build_timeline_plan



def test_segment_count_and_duration() -> None:
    plan = build_timeline_plan("A calm cinematic sunrise sequence with gentle movement", total_duration_sec=60.0)
    assert 5 <= len(plan.segments) <= 8
    assert plan.segments[0].start_sec == 0
    assert round(plan.segments[-1].end_sec, 3) == 60.0



def test_segments_have_required_fields() -> None:
    plan = build_timeline_plan("Anime city skyline and reflective nighttime scenes", total_duration_sec=60.0)
    for seg in plan.segments:
        assert seg.style
        assert seg.scene_description
        assert seg.motion_plan
        assert seg.duration_sec > 0
