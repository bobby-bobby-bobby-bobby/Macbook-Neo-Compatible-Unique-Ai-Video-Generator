"""Training script for tiny super-resolution model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from myvideo.upscale.dataset import SRImageDataset
from myvideo.upscale.model import TinyUpscaler
from myvideo.utils.hardware import detect_hardware



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny upscaler")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--hr-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="models/checkpoints/upscaler_tiny.pt")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    profile = detect_hardware()
    device = torch.device(profile.device)

    ds = SRImageDataset(args.image_dir, hr_size=args.hr_size, scale=args.scale)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TinyUpscaler(scale=args.scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for lr_frames, hr_frames in pbar:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)

            pred = model(lr_frames)
            loss = 0.8 * F.l1_loss(pred, hr_frames) + 0.2 * F.mse_loss(pred, hr_frames)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "scale": args.scale}, save_path)
    print(f"Saved upscaler model to {save_path}")


if __name__ == "__main__":
    main()
