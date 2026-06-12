from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """Minimal image dataset for probing on unlabeled image folders."""

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        root_cache: Optional[str] = None,
        **kwargs,
    ):
        del root_cache, kwargs
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.root}")
        self.image_paths = sorted(
            p
            for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.IMG_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "img_path": str(img_path),
        }
