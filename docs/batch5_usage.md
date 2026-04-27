# Batch 5 Usage (Upscale + Interpolate)

## Upscale frames

```bash
PYTHONPATH=src python -m myvideo.upscale.infer \
  --input-dir outputs/lowres_frames \
  --checkpoint models/checkpoints/upscaler_tiny.pt \
  --output-dir outputs/upscaled_frames \
  --scale 2
```

## Interpolate frames to 24fps

```bash
PYTHONPATH=src python -m myvideo.interpolate.infer \
  --input-dir outputs/upscaled_frames \
  --checkpoint models/checkpoints/interpolator_tiny.pt \
  --output-dir outputs/interpolated_frames \
  --multiplier 2
```

## Convert frames to video

```bash
python - << 'PY'
from myvideo.assemble import frames_to_video
frames_to_video('outputs/interpolated_frames', 'outputs/segment_00.mp4', fps=24)
PY
```
