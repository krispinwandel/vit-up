import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn


class ImageNetNormalization(nn.Module):
    """Normalization metadata for probes whose upsampler owns the backbone."""

    def __init__(self, name: str = "imagenet"):
        super().__init__()
        self.name = name
        self.config = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def as_size(size: int | tuple[int, int]) -> tuple[int, int]:
    return (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))


class UpsamplerBase(nn.Module):
    """Map ImageNet-normalized (B, 3, H, W) images to dense (B, h, w, C) features."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @staticmethod
    def resize(images: torch.Tensor, size: int | tuple[int, int] | None) -> torch.Tensor:
        return images if size is None else T.Resize(size=as_size(size))(images)

    @staticmethod
    def backbone_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Last-layer features of a `vit_up.model` backbone, (B, C, h, w)."""
        return backbone.hidden_states(images)[-1].permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def to_bhwc(features: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        """(B, C, h, w) -> (B, H, W, C), bilinearly resized to `size` if it differs."""
        if size is not None and tuple(features.shape[-2:]) != size:
            features = F.interpolate(features, size=size, mode="bilinear", align_corners=False)
        return features.permute(0, 2, 3, 1).contiguous()

    def forward(self, images: torch.Tensor, output_size: int | tuple[int, int],
                input_size: int | None = None) -> torch.Tensor:
        raise NotImplementedError
