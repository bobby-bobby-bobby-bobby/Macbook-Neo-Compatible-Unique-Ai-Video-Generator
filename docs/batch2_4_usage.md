# Batch 2-4 Usage Guide

## Batch 2 — Timeline Planner

Generate a structured 1-minute timeline (5-8 segments):

```bash
PYTHONPATH=src python - << 'PY'
from myvideo.planner import build_timeline_plan, save_timeline_plan

prompt = "A cinematic sunrise over a neon city, then street-level exploration with calm motion and reflective mood"
plan = build_timeline_plan(prompt, total_duration_sec=60.0)
save_timeline_plan(plan, "outputs/timeline/plan.json")
print("saved", len(plan.segments), "segments")
PY
```

## Batch 3 — Keyframe Generator

### Train

```bash
PYTHONPATH=src python -m myvideo.keyframe.train \
  --image-dir data/processed/keyframes_train \
  --prompt "cinematic neon city sunrise" \
  --epochs 5 \
  --batch-size 1 \
  --save-path models/checkpoints/keyframe_tiny.pt
```

### Infer keyframes from timeline

```bash
PYTHONPATH=src python -m myvideo.keyframe.infer \
  --timeline-json outputs/timeline/plan.json \
  --checkpoint models/checkpoints/keyframe_tiny.pt \
  --output-dir outputs/keyframes \
  --height 180 \
  --width 320 \
  --steps 12 \
  --keyframes-per-segment 2
```

### Quantize/export

```bash
PYTHONPATH=src python -m myvideo.keyframe.quantize \
  --checkpoint models/checkpoints/keyframe_tiny.pt \
  --output models/quantized/keyframe_tiny_int8.pt \
  --fp16-output models/quantized/keyframe_tiny_fp16.pt
```

## Batch 4 — Tiny Video Diffusion Inbetweener

### Train

```bash
PYTHONPATH=src python -m myvideo.inbetween.train \
  --data-dir data/processed/inbetween_triplets \
  --epochs 6 \
  --batch-size 1 \
  --save-path models/checkpoints/inbetween_tiny.pt
```

### Infer inbetween frames

```bash
PYTHONPATH=src python -m myvideo.inbetween.infer \
  --input-dir outputs/keyframes \
  --checkpoint models/checkpoints/inbetween_tiny.pt \
  --output-dir outputs/inbetween \
  --frames-between 4 \
  --image-size 180
```

### Quantize/export

```bash
PYTHONPATH=src python -m myvideo.inbetween.quantize \
  --checkpoint models/checkpoints/inbetween_tiny.pt \
  --fp16-output models/quantized/inbetween_tiny_fp16.pt \
  --script-output models/exports/inbetween_tiny_scripted.pt
```
