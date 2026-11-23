import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def normalize(images: torch.Tensor) -> torch.Tensor:
    """uint8 or [0, 255] (B, 3, H, W) images to ImageNet-normalized floats."""
    mean, std = IMAGENET_MEAN.to(images.device), IMAGENET_STD.to(images.device)
    return (images.float() / 255.0 - mean[:, None, None]) / std[:, None, None]


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalized (B, 3, H, W) images to uint8."""
    mean, std = IMAGENET_MEAN.to(images), IMAGENET_STD.to(images)
    return ((images * std[:, None, None] + mean[:, None, None]) * 255).clamp(0, 255).to(torch.uint8)


def to_pil(image: torch.Tensor) -> Image.Image:
    """An ImageNet-normalized (3, H, W) image as a PIL image."""
    return Image.fromarray(unnormalize(image[None])[0].permute(1, 2, 0).cpu().numpy())


def square_resize(size: int) -> T.Compose:
    """Resize the short side to `size`, then center crop a square."""
    return T.Compose([T.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                      T.CenterCrop((size, size))])


def image_transform(size: int, augmentation: T.Transform | None = None) -> T.Compose:
    """PIL image to an ImageNet-normalized (3, size, size) tensor, optionally augmented."""
    ops = [T.ToImage(), T.Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True)]
    ops += [augmentation] if augmentation is not None else [T.ToDtype(torch.float32, scale=True)]
    return T.Compose(ops + [T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=augmentation is not None)])
