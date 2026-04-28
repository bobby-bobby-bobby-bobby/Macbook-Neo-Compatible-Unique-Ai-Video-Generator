"""Dataset utilities for keyframe model training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolderWithPrompt(Dataset):
    """Loads image files from a directory and pairs each with a prompt string.

    Expected layout:
    - image_dir/*.png|jpg|jpeg
    """

    def __init__(self, image_dir: str | Path, prompt: str, image_size: int = 180) -> None:
        self.image_dir = Path(image_dir)
        self.prompt = prompt
        self.paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.image_dir}")

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple:
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        tensor = self.tf(img)
        return tensor, self.prompt
