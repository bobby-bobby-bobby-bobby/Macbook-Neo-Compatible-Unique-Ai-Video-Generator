"""Dataset for inbetween frame training from pre-extracted frame triplets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TripletFrameDataset(Dataset):
    """Load frame triplets from folder layout.

    Expected:
    data_dir/
      sample_000/
        a.png
        mid.png
        b.png
      sample_001/
        a.png
        mid.png
        b.png
    """

    def __init__(self, data_dir: str | Path, image_size: int = 180) -> None:
        self.data_dir = Path(data_dir)
        self.samples = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        if not self.samples:
            raise ValueError(f"No sample folders found in {self.data_dir}")

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _read(self, path: Path):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, index: int):
        root = self.samples[index]
        a = self._read(root / "a.png")
        mid = self._read(root / "mid.png")
        b = self._read(root / "b.png")
        return a, mid, b
