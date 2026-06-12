from typing import List
import numpy as np


def numpy_color_palette(n_colors: int) -> List[np.ndarray]:
    palette = [
        (255, 0, 0),
        (0, 102, 255),
        (0, 200, 0),
        (255, 215, 0),
        (170, 0, 255),
        (255, 128, 0),
        (0, 220, 220),
        (255, 0, 170),
    ]
    return [
        np.array(palette[i % len(palette)], dtype=np.float32) / 255.0
        for i in range(max(1, n_colors))
    ]
