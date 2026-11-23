"""Streamed ImageNet teacher/student pairs for self-supervised distillation.

Every sample derives both views from one augmented square image:

* pad: the teacher sees the full image; the student sees it downscaled and
  pasted at a random position on a noise canvas.
* crop: the teacher sees a random crop (resized to full resolution); the student
  sees the full image, downscaled and pasted onto a noise canvas.

`teacher_box` locates the teacher view inside the student image, in normalized
(x0, y0, x1, y1) coordinates.
"""
import os
import random
from dataclasses import dataclass
from functools import partial
from itertools import islice

import cv2
import lightning as L
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .augmentation import build_augmentation
from .transforms import IMAGENET_MEAN, IMAGENET_STD, image_transform, square_resize


@dataclass
class Views:
    pad_scale: tuple[float, float] = (0.1, 1.0)
    crop_probability: float = 0.0
    crop_scale: float = 0.5
    crop_student_scale: tuple[float, float] = (0.5, 1.0)


def random_crop(image: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A random crop with side `scale` and its normalized (x0, y0, x1, y1) box."""
    height, width = image.shape[-2:]
    crop_w, crop_h = max(1, int(width * scale)), max(1, int(height * scale))
    x = random.randint(0, width - crop_w)
    y = random.randint(0, height - crop_h)
    box = torch.tensor([x / width, y / height, (x + crop_w) / width, (y + crop_h) / height])
    return image[:, y:y + crop_h, x:x + crop_w], box


def paste_on_canvas(image: torch.Tensor, scale: float, size: int, noise: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Downscale a normalized image and paste it at a random position on a (noise) canvas.

    Returns the canvas and the pixel bounds (x0, y0, x1, y1) of the pasted content.
    """
    side = max(1, int(size * scale))
    content = TF.resize(image, [side, side], antialias=True)
    x, y = random.randint(0, size - side), random.randint(0, size - side)
    canvas = (torch.rand if noise else torch.zeros)((3, size, size), dtype=content.dtype)
    canvas.sub_(IMAGENET_MEAN[:, None, None]).div_(IMAGENET_STD[:, None, None])
    canvas[:, y:y + side, x:x + side] = content
    return canvas, torch.tensor([x, y, x + side, y + side], dtype=content.dtype)


class SupervisionPairs(IterableDataset):
    def __init__(self, source, image_size: int, views: Views, augmentation_strength: float = 0.0,
                 background_noise: bool = True):
        self.source = source
        self.image_size = image_size
        self.views = views
        self.background_noise = background_noise
        self.square = square_resize(image_size)
        augmentation = build_augmentation(augmentation_strength) if augmentation_strength > 0 else None
        self.transform = image_transform(image_size, augmentation)

    def __iter__(self):
        for item in self.source:
            yield self.sample(item["image"])

    def sample(self, image) -> dict[str, torch.Tensor]:
        use_crop = self.views.crop_probability > 0 and random.random() < self.views.crop_probability
        source = self.transform(self.square(image.convert("RGB")))
        if use_crop:
            teacher, box = random_crop(source, self.views.crop_scale)
            teacher = TF.resize(teacher, [self.image_size, self.image_size], antialias=True)
            student_scale = random.uniform(*self.views.crop_student_scale)
        else:
            teacher, box = source, torch.tensor([0., 0., 1., 1.])
            student_scale = random.uniform(*self.views.pad_scale)
        student, content = paste_on_canvas(source, student_scale, self.image_size, self.background_noise)
        offset, extent = content[:2].repeat(2), (content[2:] - content[:2]).repeat(2)
        teacher_box = (offset + box * extent) / self.image_size
        return {"teacher_image": teacher, "student_image": student, "teacher_box": teacher_box}


class _CachedPairs(Dataset):
    def __init__(self, pairs: SupervisionPairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs.source)

    def __getitem__(self, index):
        return self.pairs.sample(self.pairs.source[index]["image"])


def _init_worker(worker_id, rank=0):
    cv2.setNumThreads(1)
    if int(os.environ.get("PL_SEED_WORKERS", "0")):
        from lightning.fabric.utilities.seed import pl_worker_init_function
        pl_worker_init_function(worker_id, rank=rank)


class ImageNetDataModule(L.LightningDataModule):
    """Stream ImageNet-1k from the Hugging Face Hub (requires accepting its license).

    Validation keeps `limit_val_batches * batch_size` source images in memory,
    so every validation round uses the same images (with freshly sampled views).
    """

    def __init__(self, batch_size: int, views: dict, path: str = "ILSVRC/imagenet-1k", image_size: int = 896,
                 augmentation_strength: float = 1.0, background_noise: bool = True, num_train_workers: int = 6,
                 num_val_workers: int = 3, pin_memory: bool = True, prefetch_factor: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.views = Views(**views)
        self.path = path
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        self.background_noise = background_noise
        self.num_train_workers = num_train_workers
        self.num_val_workers = num_val_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def _pairs(self, split: str, augmentation_strength: float) -> SupervisionPairs:
        return SupervisionPairs(load_dataset(self.path, split=split, streaming=True), self.image_size, self.views,
                                augmentation_strength, self.background_noise)

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_pairs = self._pairs("train", self.augmentation_strength)
        if not hasattr(self, "val_pairs"):
            self.val_pairs = self._pairs("validation", 0.0)
            limit = getattr(self.trainer, "limit_val_batches", None)
            if isinstance(limit, int):
                count = limit * self.batch_size * self.trainer.world_size
                self.val_pairs.source = [{"image": item["image"].convert("RGB").copy()}
                                         for item in islice(self.val_pairs.source, count)]

    def _loader(self, dataset, num_workers):
        kwargs = dict(batch_size=self.batch_size, num_workers=num_workers, pin_memory=self.pin_memory,
                      drop_last=True)
        if num_workers > 0:
            rank = self.trainer.global_rank if self.trainer is not None else 0
            # Forking the training process, which already runs network and allocator threads, can deadlock workers.
            kwargs |= dict(prefetch_factor=self.prefetch_factor, persistent_workers=True,
                           worker_init_fn=partial(_init_worker, rank=rank), multiprocessing_context="forkserver")
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._loader(self.train_pairs, self.num_train_workers)

    def val_dataloader(self):
        cached = isinstance(self.val_pairs.source, list)
        return self._loader(_CachedPairs(self.val_pairs) if cached else self.val_pairs, self.num_val_workers)
