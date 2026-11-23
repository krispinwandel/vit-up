"""Shared square-padding and pixel-center geometry for correspondence evaluation."""

from dataclasses import dataclass

import torch
from PIL import Image



def square_offset(width: int, height: int) -> tuple[int, int]:
    """Top-left offset of an image centered on a square canvas (odd padding on the bottom/right)."""
    return ((height - width) // 2, 0) if height > width else (0, (width - height) // 2)


def pad_to_square(image: Image.Image) -> Image.Image:
    side = max(image.size)
    canvas = Image.new(image.mode, (side, side), color=0)
    canvas.paste(image, square_offset(*image.size))
    return canvas


@dataclass(frozen=True)
class SquareImageGeometry:
    width: int
    height: int

    def __post_init__(self):
        if min(self.width, self.height) < 1:
            raise ValueError("Image dimensions must be positive.")

    @property
    def side(self):
        return max(self.width, self.height)

    @property
    def offset(self):
        # Any extra pixel goes on the bottom/right of the square canvas.
        return square_offset(self.width, self.height)

    def pad_image(self, image: Image.Image) -> Image.Image:
        if image.size != (self.width, self.height):
            raise ValueError("Image dimensions do not match the annotation geometry.")
        return pad_to_square(image.convert("RGB"))

    def image_to_grid(self, xy, grid_size: int) -> torch.Tensor:
        """Map zero-based pixel centers through padding and align_corners=False resize."""
        xy = torch.as_tensor(xy, dtype=torch.float64)
        return (xy + xy.new_tensor(self.offset) + 0.5) * (grid_size / self.side) - 0.5

    def grid_to_image(self, xy, grid_size: int) -> torch.Tensor:
        """Invert the same transform without rounding away subpixel predictions."""
        xy = torch.as_tensor(xy, dtype=torch.float64)
        return (xy + 0.5) * (self.side / grid_size) - xy.new_tensor(self.offset) - 0.5

    def keypoint_indices(self, xy, grid_size: int) -> torch.Tensor:
        """Select the nearest available feature-grid center for each source point."""
        coords = self.image_to_grid(xy, grid_size)
        return (coords + 0.5).floor().long().clamp(0, grid_size - 1)
