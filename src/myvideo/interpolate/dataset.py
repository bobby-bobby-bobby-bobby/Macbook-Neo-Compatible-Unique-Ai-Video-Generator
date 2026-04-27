"""Dataset for frame interpolation training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class InterpTripletDataset(Dataset):
    """Folder-based triplets: prev.png, mid.png, next.png."""

    def __init__(self, data_dir: str | Path, image_size: int = 256) -> None:
        self.root = Path(data_dir)
        self.samples = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not self.samples:
            raise ValueError(f"No interpolation triplets found in {self.root}")
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, index: int):
        item = self.samples[index]
        prev = self._load(item / "prev.png")
        mid = self._load(item / "mid.png")
        nxt = self._load(item / "next.png")
        return prev, mid, nxt
