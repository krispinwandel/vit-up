import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from .base import SupervisionStrategy
from vit_up.utils.grid_coords import build_regular_grid_centers_xy


def build_quadrature_grid(quadrature_n: int, n_cells: int):
    """
    Returns:
        base_grid: Tensor of shape (quadrature_n**2, 2) containing the (x, y) offsets of the quadrature points within a unit cell.
        weights_2d: Tensor of shape (quadrature_n, quadrature_n) containing the 2D quadrature weights corresponding to each point in base_grid.
        full_grid: (n_cells * quadrature_n, n_cells * quadrature_n, 2) tensor of (x, y) coordinates of all quadrature points in the full query grid, normalized to [0, 1].
    """
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(quadrature_n)
    gx, gy = torch.meshgrid(
        torch.as_tensor(nodes_1d, dtype=torch.float32),
        torch.as_tensor(nodes_1d, dtype=torch.float32),
        indexing="xy",
    )
    offsets_xy = 0.5 * torch.stack((gx, gy), dim=-1)
    w2d = (
        torch.as_tensor(weights_1d, dtype=torch.float32)[:, None]
        * torch.as_tensor(weights_1d, dtype=torch.float32)[None, :]
    )
    w2d = w2d / w2d.sum()

    cell_size = 1.0 / n_cells
    base_grid_xy = offsets_xy * cell_size
    quadrature_grid_xy = (
        build_regular_grid_centers_xy(n_cells)[:, :, None, None, :]
        + base_grid_xy[None, None, :, :, :]
    )
    quadrature_grid_xy = quadrature_grid_xy.permute(0, 2, 1, 3, 4).reshape(
        n_cells * quadrature_n, n_cells * quadrature_n, 2
    )
    return base_grid_xy, w2d, quadrature_grid_xy


class CellIntegralQuadratureSupervision(SupervisionStrategy):
    def __init__(
        self,
        quadrature_n: int,
        embd_size: int,
        patch_size: int = 16,
        block_chunk_size: int = 64,
    ):
        self.quadrature_n = quadrature_n
        self.embd_size = embd_size
        self.patch_size = patch_size
        base_grid_xy, w2d, quadrature_grid_xy = build_quadrature_grid(
            quadrature_n, embd_size
        )
        self.w2d = w2d
        self.quadrature_grid_xy = quadrature_grid_xy
        super().__init__(block_chunk_size=block_chunk_size, patch_size=patch_size)

    def gt_img_sizes(self) -> Dict[int, int]:
        return {0: self.embd_size * self.patch_size}

    def build_query_xy_grid(self):
        return self.quadrature_grid_xy

    def block_size_query(self):
        return self.quadrature_n

    def block_size_hidden(self):
        return {0: 1}

    def query_features_to_hidden(self, q_ft_chunk: torch.Tensor):
        """
        Args:
            q_ft_chunk: (b, n_blocks, block_size_q, block_size_q, c)

        """
        q_ft_integrated = torch.sum(
            q_ft_chunk * self.w2d.to(q_ft_chunk.device)[None, None, :, :, None],
            dim=[2, 3],
            keepdim=True,
        )  # (b, n_blocks, 1, 1, c)
        return {
            0: q_ft_integrated,
        }
