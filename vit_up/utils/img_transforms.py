from typing import List
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision.transforms.v2 import InterpolationMode
import torchvision.transforms.v2 as T

RESNET_IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
RESNET_IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])


def norm_img(img: torch.Tensor, tgt_dtype=torch.float32):
    resnet_mean = RESNET_IMAGE_MEAN.to(dtype=tgt_dtype, device=img.device)
    resnet_std = RESNET_IMAGE_STD.to(dtype=tgt_dtype, device=img.device)
    img = (
        (img.type(tgt_dtype) / 255.0) - resnet_mean[None, :, None, None]
    ) / resnet_std[None, :, None, None]
    return img


def unnorm_img(img: torch.Tensor):
    resnet_mean = RESNET_IMAGE_MEAN.to(dtype=img.dtype, device=img.device)
    resnet_std = RESNET_IMAGE_STD.to(dtype=img.dtype, device=img.device)
    img = (
        (img * resnet_std[None, :, None, None]) + resnet_mean[None, :, None, None]
    ) * 255
    img = torch.clamp(img, 0, 255).type(torch.uint8)
    return img


def pixel_values_to_pil(img_torch: torch.Tensor):
    assert img_torch.ndim == 3
    img_torch = unnorm_img(img_torch[None])[0].permute(
        1, 2, 0
    )  # (C, H, W) -> (H, W, C)
    img_np = img_torch.cpu().numpy()
    return Image.fromarray(img_np)


def save_check_img_np(img_np: np.ndarray):
    # check for negative strides (flips) which cause issues
    if img_np.strides[0] < 0:
        img_np = img_np.copy()
        # force contiguous memory
    img_np = np.ascontiguousarray(img_np)
    return img_np


def img_np_to_tensor(img_np, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    """
    Resizes numpy array and converts to torch tensor.
    """
    # cv2.resize takes (width, height)
    img_resized = cv2.resize(img_np, (target_w, target_h), interpolation=interpolation)
    img_resized = save_check_img_np(img_resized)
    # Convert to Tensor (Does not normalize to 0-1 or change channels to CHW)
    # Result is (256, 256, C) if color, or (256, 256) if grayscale
    return torch.from_numpy(img_resized)


def build_image_transform(
    img_w: int,
    img_h: int,
):
    """
    This function is adapted from AnyUp
    """
    ops: List[T.Transform] = [
        T.ToImage(),
        T.Resize(
            (img_h, img_w), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
    ]
    ops.extend(
        [
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=RESNET_IMAGE_MEAN, std=RESNET_IMAGE_STD),
        ]
    )
    return T.Compose(ops)


def build_image_transform_with_augmentation(
    img_w: int,
    img_h: int,
    strength: float = 1.0,
):
    """
    This function is adapted from AnyUp
    NOTE we merge image transform and augmentation because their operations are interleaved
    """
    ops: List[T.Transform] = [
        T.ToImage(),
        T.Resize(
            (img_h, img_w), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        # T.Grayscale(),
    ]
    augmentation_transforms = [
        T.ToDtype(torch.uint8, scale=True),  # work in uint8 for color/JPEG
        T.RandomApply(
            [
                T.ColorJitter(0.2, 0.2, 0.3, 0.05),
                T.RandomPhotometricDistort(p=0.5),
            ],
            p=0.5 ** (1 / strength),
        ),
        T.RandomGrayscale(p=0.1 ** (1 / strength)),
        # T.Grayscale(),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5))], p=0.35 ** (1 / strength)
        ),
        T.RandomChoice(
            [  # JPEG with a few discrete qualities
                T.JPEG(95),
                T.JPEG(80),
                T.JPEG(60),
                T.JPEG(40),
            ],
            p=[0.25, 0.35, 0.25, 0.15],
        ),
        T.RandomApply([T.RandomAutocontrast()], p=0.2 ** (1 / strength)),
        T.RandomApply([T.RandomEqualize()], p=0.15 ** (1 / strength)),
        T.RandomApply([T.RandomAdjustSharpness(1.5)], p=0.2 ** (1 / strength)),
        T.RandomApply([T.RandomPosterize(bits=5)], p=0.1 ** (1 / strength)),
        T.RandomApply([T.RandomSolarize(threshold=0.9)], p=0.05 ** (1 / strength)),
        T.RandomApply([T.RandomInvert()], p=0.02 ** (1 / strength)),
        T.ToDtype(torch.float32, scale=True),  # now in [0,1] float
        T.RandomApply(
            [T.GaussianNoise(sigma=0.03, clip=True)], p=0.6 ** (1 / strength)
        ),
    ]
    ops.extend(augmentation_transforms)
    ops.append(T.Normalize(mean=RESNET_IMAGE_MEAN, std=RESNET_IMAGE_STD))
    return T.Compose(ops)


def build_bg_transform(h, w, min_scale):
    bg_transform = T.Compose(
        [
            T.ToImage(),
            T.RandomResizedCrop(
                (h, w),
                scale=(min_scale, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),  # Allow slight aspect ratio jitter
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            # ... Convert back to Numpy ...
            T.Lambda(lambda x: x.permute(1, 2, 0).numpy()),
        ]
    )
    return bg_transform
