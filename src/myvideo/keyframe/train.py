"""Training script for tiny keyframe generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from myvideo.keyframe.dataset import ImageFolderWithPrompt
from myvideo.keyframe.model import TinyKeyframeGenerator
from myvideo.utils.hardware import detect_hardware



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny keyframe generator")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=180)
    parser.add_argument("--save-path", type=str, default="models/checkpoints/keyframe_tiny.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()



def train() -> None:
    args = parse_args()
    hardware = detect_hardware()
    device = torch.device(hardware.device)

    dataset = ImageFolderWithPrompt(args.image_dir, args.prompt, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TinyKeyframeGenerator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, prompts in pbar:
            images = images.to(device)

            latent_noise = torch.randn(images.size(0), model.latent_channels, images.size(2), images.size(3), device=device)
            prompt = prompts[0]
            denoised = model.denoise_step(latent_noise, prompt)
            reconstructed = model.decoder(denoised)

            loss = F.l1_loss(reconstructed, images)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    train()
