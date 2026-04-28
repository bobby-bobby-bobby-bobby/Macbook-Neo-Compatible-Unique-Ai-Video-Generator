"""Dataset for super-resolution training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SRImageDataset(Dataset):
    """Create LR-HR pairs from HR images."""

    def __init__(self, image_dir: str | Path, hr_size: int = 256, scale: int = 2) -> None:
        self.image_dir = Path(image_dir)
        self.paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not self.paths:
            raise ValueError(f"No images found in {self.image_dir}")

        self.hr_tf = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
        ])
        self.lr_tf = transforms.Compose([
            transforms.Resize((hr_size // scale, hr_size // scale)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        hr = self.hr_tf(image)
        lr = self.lr_tf(image)
        return lr, hr
