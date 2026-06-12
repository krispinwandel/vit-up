from importlib import import_module
from typing import Any, Dict, List

from PIL import Image

from lightning import LightningModule
from vit_up.training.lightning_module import ViTUpPL
from vit_up.utils import pil_img_utils

from .base import VisHelper


class StackVisHelper(VisHelper):
    """Compose multiple visualization helpers and stack their outputs vertically."""

    def __init__(
        self,
        vis_helpers: List[VisHelper],
    ):
        super().__init__()
        self.vis_helpers = vis_helpers

    def set_input_images(self, imgs: List[Image.Image], pl_module: LightningModule):
        self.imgs = [img.copy() for img in imgs]
        self.pixel_values = self._transform_images(imgs=imgs, device=pl_module.device)
        for helper in self.vis_helpers:
            helper.set_input_images(imgs=imgs, pl_module=pl_module)

    def generate_vis(self, img_idx: int, pl_module: LightningModule) -> Image.Image:

        helper_images = [
            helper.generate_vis(img_idx=img_idx, pl_module=pl_module)
            for helper in self.vis_helpers
        ]

        return pil_img_utils.concat_images(
            helper_images,
            target_width=helper_images[0].width,
            mode="col",
            pad=2,
            pad_color=(0, 0, 0),
        )
