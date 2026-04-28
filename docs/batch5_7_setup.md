# Batch 5-7 Setup and Execution

## What is included

- Batch 5:
  - Tiny upscaler model (`myvideo.upscale.*`)
  - Tiny frame interpolator model (`myvideo.interpolate.*`)
  - Low-memory frame-by-frame processing utilities
- Batch 6:
  - CLI commands: `make`, `gen`, `upscale`, `interpolate`, `concat`
- Batch 7:
  - Colab notebooks for training, quantization, and inference tests
  - Local Mac instructions and end-to-end commands

## Local Mac (Apple Silicon A18) setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
ffmpeg -version
```

If `ffmpeg` is missing:

```bash
brew install ffmpeg
```

## Environment checks

```bash
myvideo show-config --mode preview
myvideo show-config --mode final
myvideo show-config --mode low_memory
```

## Training on Mac (optional)

```bash
PYTHONPATH=src python -m myvideo.keyframe.train --image-dir data/processed/keyframes_train --prompt "stylized scene" --epochs 3 --batch-size 1 --save-path models/checkpoints/keyframe_tiny.pt
PYTHONPATH=src python -m myvideo.inbetween.train --data-dir data/processed/inbetween_triplets --epochs 3 --batch-size 1 --save-path models/checkpoints/inbetween_tiny.pt
PYTHONPATH=src python -m myvideo.upscale.train --image-dir data/processed/sr_train --scale 2 --epochs 3 --batch-size 1 --save-path models/checkpoints/upscaler_tiny.pt
PYTHONPATH=src python -m myvideo.interpolate.train --data-dir data/processed/interp_triplets --epochs 3 --batch-size 1 --save-path models/checkpoints/interpolator_tiny.pt
```

## Quantization + export

```bash
PYTHONPATH=src python -m myvideo.keyframe.quantize --checkpoint models/checkpoints/keyframe_tiny.pt
PYTHONPATH=src python -m myvideo.inbetween.quantize --checkpoint models/checkpoints/inbetween_tiny.pt
PYTHONPATH=src python -m myvideo.upscale.quantize --checkpoint models/checkpoints/upscaler_tiny.pt
PYTHONPATH=src python -m myvideo.interpolate.quantize --checkpoint models/checkpoints/interpolator_tiny.pt
```

## Inference on Mac (main target)

Create `prompt.txt` with your 1-minute text prompt, then run:

```bash
myvideo make \
  --prompt-file prompt.txt \
  --output-video outputs/final_video.mp4 \
  --mode low_memory \
  --keyframe-checkpoint models/checkpoints/keyframe_tiny.pt \
  --inbetween-checkpoint models/checkpoints/inbetween_tiny.pt \
  --upscaler-checkpoint models/checkpoints/upscaler_tiny.pt \
  --interpolator-checkpoint models/checkpoints/interpolator_tiny.pt \
  --work-dir outputs/work_lowmem
```

## Colab workflow

1. Upload repo + training data to Colab runtime.
2. Run notebooks in order:
   - `notebooks/01_train_keyframe_colab.ipynb`
   - `notebooks/02_train_inbetween_colab.ipynb`
   - `notebooks/03_train_upscaler_colab.ipynb`
   - `notebooks/04_train_interpolator_colab.ipynb`
   - `notebooks/05_quantize_export_and_inference_tests.ipynb`
3. Download exported checkpoints to Mac.
4. Run `myvideo make ...` on Mac for final inference.
