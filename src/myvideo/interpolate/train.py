"""Training script for tiny frame interpolator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from myvideo.interpolate.dataset import InterpTripletDataset
from myvideo.interpolate.model import TinyFrameInterpolator
from myvideo.utils.hardware import detect_hardware



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny frame interpolator")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--save-path", type=str, default="models/checkpoints/interpolator_tiny.pt")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    profile = detect_hardware()
    device = torch.device(profile.device)

    ds = InterpTripletDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TinyFrameInterpolator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for prev, mid, nxt in pbar:
            prev = prev.to(device)
            mid = mid.to(device)
            nxt = nxt.to(device)
            t = torch.full((prev.size(0),), 0.5, device=device)

            pred = model(prev, nxt, t)
            loss = 0.7 * F.l1_loss(pred, mid) + 0.3 * F.mse_loss(pred, mid)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, save_path)
    print(f"Saved interpolator model to {save_path}")


if __name__ == "__main__":
    main()
