from typing import List, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw

from vit_up.utils import pil_img_utils

TEXTBOX_HEIGHT = 30
FONT_SIZE = 16


def add_cell_desc(img: Image.Image, desc: str, **overwrite_args) -> Image.Image:
    return pil_img_utils.add_description_to_image(
        img,
        description=desc,
        margin=2,
        font_size=FONT_SIZE,
        placement="top",
        text_align="center",
        bg_color="black",
        font_color="white",
        # text_box_height_ratio=0.12,
        overlay=False,
        textbox_height=TEXTBOX_HEIGHT,
        **overwrite_args,
    )


def make_row_image(
    row_cells: List[Image.Image],
    label: str,
    cell_width: Optional[int] = None,
    resample: int = Image.Resampling.NEAREST,
) -> Image.Image:
    row = pil_img_utils.concat_images(
        row_cells,
        # target_width=cell_width,
        mode="row",
        pad=2,
        pad_color=(0, 0, 0),
        interpolate_resample=resample,
    )
    return pil_img_utils.add_description_to_image(
        row,
        description=label,
        margin=4,
        font_size=FONT_SIZE,
        placement="top",
        text_align="center",
        bg_color="black",
        font_color="white",
        # text_box_height_ratio=0.1,
        overlay=False,
        textbox_height=TEXTBOX_HEIGHT,
    )
