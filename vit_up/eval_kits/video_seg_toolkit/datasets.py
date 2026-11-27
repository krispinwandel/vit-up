# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from google-research/lila (dataload.py): the DAVIS-2017 dataset only,
# with a configurable directory layout. The transform is unchanged.
"""DAVIS-2017 videos for the LILA video segmentation protocol."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- DAVIS-2017

class DavisVideos:
    """DAVIS-2017 semi-supervised sequences at 480p: normalized frames and label maps."""

    def __init__(self, root: str | Path, split_file: str | Path | None = None):
        self.root = Path(root)
        split_file = Path(split_file) if split_file else self.root / "ImageSets" / "2017" / "val.txt"
        self.videos = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.transform = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

    def __len__(self) -> int:
        return len(self.videos)

    def get_sequence(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns frames (T, 3, H, W) and masks (T, H, W)."""
        frames_dir = self.root / "JPEGImages" / "480p" / name
        masks_dir = self.root / "Annotations" / "480p" / name
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        mask_paths = sorted(masks_dir.glob("*.png"))
        if not frame_paths or len(frame_paths) != len(mask_paths):
            raise FileNotFoundError(f"Missing or unmatched DAVIS frames/masks for {name} under {self.root}.")
        frames, masks = [], []
        for frame_path, mask_path in zip(frame_paths, mask_paths):
            out = self.transform(image=_read_rgb(frame_path), mask=np.array(Image.open(mask_path)))
            frames.append(out["image"])
            masks.append(out["mask"])
        return torch.stack(frames), torch.stack(masks)
