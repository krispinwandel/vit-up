import torch
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Tuple, Union

from .base import SupervisionStrategy
from vit_up.utils.grid_coords import build_regular_grid_centers_xy


class PointwiseTokenSupervision(SupervisionStrategy):
    def __init__(
        self,
        embd_size: int,
        patch_size: int,
        block_chunk_size: int,
    ):
        self.embd_size = embd_size
        super().__init__(block_chunk_size=block_chunk_size, patch_size=patch_size)

    def gt_img_sizes(self) -> Dict[int, int]:
        return {0: self.patch_size * self.embd_size}

    def build_query_xy_grid(self):
        return build_regular_grid_centers_xy(self.embd_size)

    def block_size_query(self):
        return 1

    def block_size_hidden(self):
        return {0: 1}

    def query_features_to_hidden(self, q_ft_chunk):
        """
        Args:
            q_ft_chunk: (b, n_blocks, block_size_q, block_size_q, c)
        Returns:
            Dict[int, torch.Tensor] where tensors have shape (b, n_blocks, block_size_h, block_size_h, c)
        """
        return {0: q_ft_chunk}
