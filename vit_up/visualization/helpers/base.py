from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image

from vit_up.utils import img_transforms
from lightning import LightningModule

# from vit_up.training.lightning_module import ViTUpPL


class VisHelper(ABC):
    """Shared base class for ViTUp visualization helpers."""

    def __init__(
        self,
        img_w: int = 1024,
        img_h: int = 1024,
    ):
        self.img_transform = img_transforms.build_image_transform(
            img_w=img_w, img_h=img_h
        )

        # Cached per-batch values initialized by set_input_images.
        self.imgs: Optional[List[Image.Image]] = None
        self.pixel_values: Optional[torch.Tensor] = None

    def _transform_images(
        self, imgs: List[Image.Image], device: torch.device
    ) -> torch.Tensor:
        return torch.stack([self.img_transform(img) for img in imgs]).to(device)

    @abstractmethod
    def set_input_images(self, imgs: List[Image.Image], pl_module: LightningModule):
        # TODO we should probably change this function set_batch(batch, pl_module) and the
        #   helper class decides on what to do with the batch, since some helpers may need
        #   more than just the images, and it work with more dataloaders.
        #   For simplicity, we keep it like this for now.
        """Initialize helper state from a list of PIL images."""

    @abstractmethod
    def generate_vis(self, img_idx: int, pl_module: LightningModule) -> Image.Image:
        """Generate one visualization image for a given index from initialized state."""
