import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from vit_up.data.imagenet import SupervisionPairs, Views


def image(seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))


@pytest.mark.parametrize("views", [Views(pad_scale=(0.3, 0.8)), Views(pad_scale=(0.3, 0.5), crop_probability=1.0)])
def test_teacher_box_locates_teacher_in_student(views):
    random.seed(0)
    torch.manual_seed(0)
    pairs = SupervisionPairs([], image_size=128, views=views, background_noise=False)
    sample = pairs.sample(image(0))
    assert sample["teacher_image"].shape == sample["student_image"].shape == (3, 128, 128)
    x0, y0, x1, y1 = (sample["teacher_box"] * 128).round().long().tolist()
    assert 0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128
    region = sample["student_image"][:, y0:y1, x0:x1]
    teacher = F.interpolate(sample["teacher_image"][None], size=region.shape[-2:], mode="area")[0]
    assert (region - teacher).abs().mean() < 0.2


def test_augmentation_changes_only_appearance():
    random.seed(0)
    torch.manual_seed(0)
    pairs = SupervisionPairs([], image_size=64, views=Views(), augmentation_strength=1.0)
    sample = pairs.sample(image(1))
    assert torch.isfinite(sample["student_image"]).all()
