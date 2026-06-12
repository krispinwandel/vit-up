import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import lightning as pl
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from PIL import Image

from ..utils import img_transforms


def create_img_container(
    img: Image.Image,
    target_size: int,
    n_tokens_per_side: int,
    rel_size_min: float,
    rel_size_max=1.0,
    random_bg_noise=True,
):
    # sample random rel_size for this image
    rel_size_min = max(rel_size_min, 1.0 / target_size)
    rel_size_max = min(rel_size_max, 1.0)
    rel_size = random.uniform(rel_size_min, rel_size_max)
    scaled_size = int(target_size * rel_size)

    # scale image
    img = TF.to_tensor(img)
    img_scaled = F.interpolate(
        img.unsqueeze(0),
        size=(scaled_size, scaled_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # paste onto img_container at random location
    offset_x = random.randint(0, target_size - scaled_size)
    offset_y = random.randint(0, target_size - scaled_size)
    if random_bg_noise:
        img_container = torch.rand((3, target_size, target_size), dtype=img.dtype)
    else:
        img_container = torch.zeros((3, target_size, target_size), dtype=img.dtype)
    img_container[
        :,
        offset_y : offset_y + scaled_size,
        offset_x : offset_x + scaled_size,
    ] = img_scaled

    # query coods
    x_coords = offset_x / target_size + (
        torch.linspace(0.5, n_tokens_per_side - 0.5, n_tokens_per_side)
        * rel_size
        / n_tokens_per_side
    )
    y_coords = offset_y / target_size + (
        torch.linspace(0.5, n_tokens_per_side - 0.5, n_tokens_per_side)
        * rel_size
        / n_tokens_per_side
    )
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    query_xy_normalized = torch.stack([grid_x, grid_y], dim=-1)
    query_size_normalized = torch.tensor(rel_size / n_tokens_per_side).expand_as(
        query_xy_normalized[:, :, 0]
    )  # Normalized size of each patch
    return {
        "img_container": Image.fromarray(
            (img_container * 255).type(torch.uint8).numpy().transpose(1, 2, 0)
        ),
        "query_xy": query_xy_normalized.float(),
        "query_size": query_size_normalized.float(),
    }


class StreamedDatasetLIIF(IterableDataset):
    def __init__(
        self,
        hf_dataset_stream,
        n_tokens_per_side=32,
        target_size=224,
        aug_strength=0.0,
        min_rel_size=0.5,
        img_container_rel_size_min=0.1,
        img_container_rel_size_max=1.0,
    ):
        self.dataset = hf_dataset_stream
        self.target_size = target_size
        self.min_rel_size = min_rel_size
        self.n_tokens_per_side = n_tokens_per_side
        self.img_container_rel_size_min = img_container_rel_size_min
        self.img_container_rel_size_max = img_container_rel_size_max
        self.img_transform = (
            img_transforms.build_image_transform(target_size, target_size)
            if aug_strength == 0.0
            else img_transforms.build_image_transform_with_augmentation(
                target_size, target_size, strength=aug_strength
            )
        )
        self.img_transform_no_aug = img_transforms.build_image_transform(
            target_size, target_size
        )

    def process_item(self, item):
        img = item["image"].convert("RGB")
        w, h = img.size

        # 1. Random Crop to Target Aspect Ratio
        target_ratio = 1.0  # Square aspect ratio for LIIF
        if w / h > target_ratio:
            max_h = h
            max_w = int(h * target_ratio)
        else:
            max_w = w
            max_h = int(w / target_ratio)

        max_orig_edge = max(w, h)
        min_edge_len = self.min_rel_size * max_orig_edge
        max_crop_edge = max(max_w, max_h)

        min_edge_len = min(min_edge_len, max_crop_edge)
        s_min = min_edge_len / max_crop_edge if max_crop_edge > 0 else 1.0

        s = random.uniform(s_min, 1.0)
        crop_w = max(1, int(max_w * s))
        crop_h = max(1, int(max_h * s))

        offset_x = random.randint(0, max(0, w - crop_w))
        offset_y = random.randint(0, max(0, h - crop_h))

        img_crop_pil = TF.crop(img, offset_y, offset_x, crop_h, crop_w)
        pixel_values = self.img_transform(img_crop_pil)
        img_crop_pil_aug = img_transforms.pixel_values_to_pil(pixel_values)

        img_container_data = create_img_container(
            img_crop_pil_aug,
            target_size=self.target_size,
            n_tokens_per_side=self.n_tokens_per_side,
            rel_size_min=self.img_container_rel_size_min,
            rel_size_max=self.img_container_rel_size_max,
        )
        # NOTE important: no double augmentation!
        pixel_values_container = self.img_transform_no_aug(
            img_container_data["img_container"]
        )
        return {
            "pixel_values": pixel_values,
            "pixel_values_container": pixel_values_container,
            "query_xy": img_container_data["query_xy"],
            "query_size": img_container_data["query_size"],
        }

    def __iter__(self):
        # Yield processed items one by one as they stream in
        for item in self.dataset:
            yield self.process_item(item)


class StreamedImageNetLIIFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_train_workers=4,
        num_val_workers=0,
        target_size=224,
        aug_strength=1.0,
        min_rel_size=0.5,
        img_container_rel_size_min=0.1,
        img_container_rel_size_max=1.0,
        n_tokens_per_side=32,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False,
        path=None,
    ):
        super().__init__()
        self.path = path
        if path is None:
            self.path = "ILSVRC/imagenet-1k"
        self.target_size = target_size
        self.aug_strength = aug_strength
        self.batch_size = batch_size
        self.num_train_workers = num_train_workers
        self.num_val_workers = num_val_workers
        self.min_rel_size = min_rel_size
        self.img_container_rel_size_min = img_container_rel_size_min
        self.img_container_rel_size_max = img_container_rel_size_max
        self.n_tokens_per_side = n_tokens_per_side
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)

    def setup(self, stage=None):
        if stage in {"fit", None}:
            hf_train = load_dataset(self.path, split="train", streaming=True)
            # hf_train = hf_train.shuffle(buffer_size=100, seed=42)

            self.train_dataset = StreamedDatasetLIIF(
                hf_train,
                target_size=self.target_size,
                aug_strength=self.aug_strength,
                min_rel_size=self.min_rel_size,
                img_container_rel_size_min=self.img_container_rel_size_min,
                img_container_rel_size_max=self.img_container_rel_size_max,
                n_tokens_per_side=self.n_tokens_per_side,
            )

        if stage in {"fit", "validate", None}:
            # Enable streaming=True here
            hf_val = load_dataset(self.path, split="validation", streaming=True)

            self.val_dataset = StreamedDatasetLIIF(
                hf_val,
                target_size=self.target_size,
                aug_strength=0.0,
                min_rel_size=self.min_rel_size,
                img_container_rel_size_min=self.img_container_rel_size_min,
                img_container_rel_size_max=self.img_container_rel_size_max,
                n_tokens_per_side=self.n_tokens_per_side,
            )

    def train_dataloader(self):
        # IMPORTANT: shuffle=True is strictly forbidden when using IterableDatasets in DataLoader
        # drop_last=True avoids short final batches that would trigger new torch.compile specializations
        loader_kwargs = {
            "dataset": self.train_dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_train_workers,
            "pin_memory": self.pin_memory,
            "drop_last": True,
        }
        if self.num_train_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(**loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = {
            "dataset": self.val_dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_val_workers,
            "pin_memory": self.pin_memory,
            "drop_last": True,
        }
        if self.num_val_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(**loader_kwargs)
