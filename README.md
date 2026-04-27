# myvideo: Hybrid 2D Video Generation for Apple Silicon A18 (8GB)

Terminal-first, low-memory hybrid 2D video generation pipeline.

## Implemented

- ✅ Batch 1: project structure + config system + hardware detection
- ✅ Batch 2: rule-based timeline planner (prompt -> 5-8 structured segments)
- ✅ Batch 3: tiny keyframe generator (model + train + infer + quantize)
- ✅ Batch 4: tiny inbetweener (model + train + infer + quantize)
- ✅ Batch 5: tiny upscaler + tiny frame interpolator (train + infer + quantize)
- ✅ Batch 6: full `myvideo` CLI (`make`, `gen`, `upscale`, `interpolate`, `concat`)
- ✅ Batch 7: Colab notebooks + local Mac setup/inference docs

## Architecture Map

1. **Timeline Planner** (`src/myvideo/planner/`)
2. **Keyframe Generator** (`src/myvideo/keyframe/`)
3. **Tiny Video Inbetweener** (`src/myvideo/inbetween/`)
4. **Upscaler** (`src/myvideo/upscale/`)
5. **Frame Interpolator** (`src/myvideo/interpolate/`)
6. **Assembler (FFmpeg)** (`src/myvideo/assemble/`)
7. **Orchestrator** (`src/myvideo/pipeline/`)

## CLI Commands

```bash
myvideo show-config --mode low_memory
myvideo plan --prompt-file prompt.txt --output-json outputs/timeline.json --duration 60
myvideo gen --timeline-json outputs/timeline.json --keyframe-checkpoint models/checkpoints/keyframe_tiny.pt --inbetween-checkpoint models/checkpoints/inbetween_tiny.pt --output-dir outputs/gen
myvideo upscale --input-dir outputs/frames_lowres --checkpoint models/checkpoints/upscaler_tiny.pt --output-dir outputs/frames_upscaled --scale 2
myvideo interpolate --input-dir outputs/frames_upscaled --checkpoint models/checkpoints/interpolator_tiny.pt --output-dir outputs/frames_interp --multiplier 2
myvideo concat --segments-dir outputs/segments --output-video outputs/final.mp4
myvideo make --prompt-file prompt.txt --output-video outputs/final.mp4 --mode low_memory --keyframe-checkpoint models/checkpoints/keyframe_tiny.pt --inbetween-checkpoint models/checkpoints/inbetween_tiny.pt --upscaler-checkpoint models/checkpoints/upscaler_tiny.pt --interpolator-checkpoint models/checkpoints/interpolator_tiny.pt
```

## Presets

- `preview`: fast, smaller resolution/fps
- `final`: higher quality
- `low_memory`: safest for constrained A18/8GB conditions

## Additional Docs

- Batch 2-4 usage: `docs/batch2_4_usage.md`
- Batch 5 usage: `docs/batch5_usage.md`
- Batch 5-7 setup/instructions: `docs/batch5_7_setup.md`

## Notebook files (Batch 7)

- `notebooks/01_train_keyframe_colab.ipynb`
- `notebooks/02_train_inbetween_colab.ipynb`
- `notebooks/03_train_upscaler_colab.ipynb`
- `notebooks/04_train_interpolator_colab.ipynb`
- `notebooks/05_quantize_export_and_inference_tests.ipynb`
