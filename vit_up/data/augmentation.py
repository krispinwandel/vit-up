"""Photometric training augmentation (adapted from AnyUp).

Torchvision samples every parameter; OpenCV executes the uint8 color and filter
kernels on CPU, which is much faster for large images. Results agree with
torchvision up to rounding.
"""
import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T


def _is_cpu_rgb(image):
    return (isinstance(image, torch.Tensor) and image.device.type == "cpu" and image.dtype == torch.uint8
            and image.ndim == 3 and image.shape[0] == 3 and image.numel() > 0)


def _as_array(image):
    return np.ascontiguousarray(image.permute(1, 2, 0).numpy())


def _as_tensor(array, original):
    result = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tv_tensors.wrap(result, like=original) if isinstance(original, tv_tensors.Image) else result


def _gray(image):
    # Truncate like torchvision's uint8 grayscale (OpenCV's RGB2GRAY rounds).
    gray = image[..., 0].astype(np.float32) * np.float32(.2989)
    gray += image[..., 1].astype(np.float32) * np.float32(.587)
    gray += image[..., 2].astype(np.float32) * np.float32(.114)
    return np.floor(gray)


def _adjust(image, kind, factor):
    if kind == "brightness":
        table = np.clip(np.arange(256, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
        return cv2.LUT(image, table)
    if kind == "contrast":
        mean = _gray(image).mean(dtype=np.float32)
        table = np.arange(256, dtype=np.float32) * factor
        table += mean * (1. - factor)
        return cv2.LUT(image, table.clip(0, 255).astype(np.uint8))
    if kind == "saturation":
        gray = _gray(image)[..., None]
        result = image.astype(np.float32) * factor
        result += gray * (1. - factor)
        return result.clip(0, 255).astype(np.uint8)
    if kind == "hue":
        hsv = cv2.cvtColor(image.astype(np.float32) * np.float32(1. / 255), cv2.COLOR_RGB2HSV)
        hue = hsv[..., 0]
        hue += np.float32(factor * 360.)
        if factor < 0:
            np.add(hue, 360., out=hue, where=hue < 0)
        else:
            np.subtract(hue, 360., out=hue, where=hue >= 360.)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # torchvision converts float to uint8 by scaling with 256 - 1e-3.
        return (rgb * np.float32(255.999)).clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unknown color adjustment: {kind}")


class ColorJitter(T.ColorJitter):
    def transform(self, inpt, params):
        if not _is_cpu_rgb(inpt):
            return super().transform(inpt, params)
        image = _as_array(inpt)
        kinds = ("brightness", "contrast", "saturation", "hue")
        for index in params["fn_idx"].tolist():
            factor = params[f"{kinds[index]}_factor"]
            if factor is not None:
                image = _adjust(image, kinds[index], factor)
        return _as_tensor(image, inpt)


class PhotometricDistort(T.RandomPhotometricDistort):
    def transform(self, inpt, params):
        if not _is_cpu_rgb(inpt):
            return super().transform(inpt, params)
        image = _as_array(inpt)
        order = ["brightness", "saturation", "hue"]
        order.insert(1 if params["contrast_before"] else 3, "contrast")
        for kind in order:
            factor = params[f"{kind}_factor"]
            if factor is not None:
                image = _adjust(image, kind, factor)
        if params["channel_permutation"] is not None:
            image = image[..., params["channel_permutation"].tolist()]
        return _as_tensor(image, inpt)


class GaussianBlur(T.GaussianBlur):
    def transform(self, inpt, params):
        if not _is_cpu_rgb(inpt) or min(inpt.shape[-2:]) <= max(self.kernel_size) // 2:
            return super().transform(inpt, params)
        kx = cv2.getGaussianKernel(self.kernel_size[0], params["sigma"][0], cv2.CV_32F)
        ky = cv2.getGaussianKernel(self.kernel_size[1], params["sigma"][1], cv2.CV_32F)
        result = cv2.sepFilter2D(_as_array(inpt), cv2.CV_32F, kx, ky, borderType=cv2.BORDER_REFLECT_101)
        return _as_tensor(result.round().clip(0, 255).astype(np.uint8), inpt)


class AdjustSharpness(T.RandomAdjustSharpness):
    def transform(self, inpt, params):
        if not _is_cpu_rgb(inpt) or min(inpt.shape[-2:]) <= 2:
            return super().transform(inpt, params)
        image = _as_array(inpt).astype(np.float32)
        kernel = np.full((3, 3), 1. / 13, dtype=np.float32)
        kernel[1, 1] = 5. / 13
        blurred = cv2.filter2D(image, -1, kernel)[1:-1, 1:-1]
        np.rint(blurred, out=blurred)
        interior = image[1:-1, 1:-1]
        blurred -= interior
        blurred *= 1. - self.sharpness_factor
        interior += blurred
        return _as_tensor(image.clip(0, 255).astype(np.uint8), inpt)


class Contiguous(torch.nn.Module):
    def forward(self, image):
        return image.contiguous()


class GaussianNoise(T.GaussianNoise):
    def transform(self, inpt, params):
        if not ((type(inpt) is torch.Tensor or isinstance(inpt, tv_tensors.Image)) and inpt.device.type == "cpu"
                and inpt.ndim == 3 and inpt.dtype == torch.float32 and not inpt.requires_grad and self.sigma >= 0):
            return super().transform(inpt, params)
        noise = torch.randn_like(inpt)
        noise.mul_(self.sigma).add_(self.mean).add_(inpt)
        if self.clip:
            noise.clamp_(0, 1)
        return tv_tensors.wrap(noise, like=inpt) if isinstance(inpt, tv_tensors.Image) else noise


def build_augmentation(strength: float = 1.0) -> T.Compose:
    """uint8 photometric augmentation followed by float Gaussian noise; outputs [0, 1] floats."""
    def p(probability):
        return probability ** (1 / strength)

    return T.Compose([
        T.ToDtype(torch.uint8, scale=True),
        T.RandomApply([ColorJitter(0.2, 0.2, 0.3, 0.05), PhotometricDistort(p=0.5)], p=p(0.5)),
        T.RandomGrayscale(p=p(0.1)),
        T.RandomApply([GaussianBlur(kernel_size=5, sigma=(0.3, 1.5))], p=p(0.35)),
        T.RandomChoice([T.JPEG(95), T.JPEG(80), T.JPEG(60), T.JPEG(40)], p=[0.25, 0.35, 0.25, 0.15]),
        T.RandomApply([T.RandomAutocontrast()], p=p(0.2)),
        T.RandomApply([T.RandomEqualize()], p=p(0.15)),
        T.RandomApply([AdjustSharpness(1.5)], p=p(0.2)),
        T.RandomApply([T.RandomPosterize(bits=5)], p=p(0.1)),
        T.RandomApply([T.RandomSolarize(threshold=0.9 * 255)], p=p(0.05)),
        T.RandomApply([T.RandomInvert()], p=p(0.02)),
        T.ToDtype(torch.float32, scale=True),
        Contiguous(),
        T.RandomApply([GaussianNoise(sigma=0.03, clip=True)], p=p(0.6)),
    ])
