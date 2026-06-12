import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import SupervisionStrategy
from vit_up.utils.grid_coords import build_regular_grid_centers_xy


class MultiScaleSupervision(SupervisionStrategy):
    def __init__(
        self,
        min_embd_size: int,
        n_iters: int,
        patch_size: int = 16,
        scale: int = 2,
        block_chunk_size: int = 64,
        mode: str = "bilinear",
        use_denoised_gt: bool = False,
        denoise_offsets_dists: Optional[List[int]] = None,
        use_full_image: bool = False,
    ):
        self.min_embd_size = min_embd_size
        self.scale = scale
        self.n_iters = n_iters
        self.mode = mode
        super().__init__(
            block_chunk_size=block_chunk_size,
            patch_size=patch_size,
            use_denoised_gt=use_denoised_gt,
            denoise_offsets_dists=denoise_offsets_dists,
            use_full_image=use_full_image,
        )

    def gt_img_sizes(self) -> Dict[int, int]:
        return {
            i: self.patch_size * self.min_embd_size * self.scale**i
            for i in range(self.n_iters)
        }

    def build_query_xy_grid(self):
        return build_regular_grid_centers_xy(
            self.min_embd_size * self.block_size_query()
        )

    def block_size_query(self):
        return self.scale ** (self.n_iters - 1)

    def block_size_hidden(self):
        return {i: self.scale**i for i in range(self.n_iters)}

    def query_features_to_hidden(self, q_ft_chunk):
        """
        Args:
            q_ft_chunk: (b, n_blocks, block_size_q, block_size_q, c)
        Returns:
            Dict[int, torch.Tensor] where tensors have shape (b, n_blocks, block_size_h, block_size_h, c)
        """
        b, n_blocks, block_size_q, _, c = q_ft_chunk.shape
        q_ft_chunk_flat = q_ft_chunk.flatten(
            0, 1
        )  # (b * n_blocks, block_size_q, block_size_q, c)
        q_ft_chunk_flat = q_ft_chunk_flat.permute(
            0, 3, 1, 2
        )  # (b * n_blocks, c, block_size_q, block_size_q)
        block_size_hidden = self.block_size_hidden()

        def _downsample_mean(x: torch.Tensor, out_size: int) -> torch.Tensor:
            if out_size == block_size_q:
                return x
            if block_size_q % out_size == 0:
                kernel = block_size_q // out_size
                return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
            return F.adaptive_avg_pool2d(x, output_size=(out_size, out_size))

        def _downsample_bilinear(x: torch.Tensor, out_size: int) -> torch.Tensor:
            if out_size == block_size_q:
                return x
            return F.interpolate(
                x,
                size=(out_size, out_size),
                mode="bilinear",
                align_corners=False,
            )

        _downsample_fn = (
            _downsample_mean if self.mode == "mean" else _downsample_bilinear
        )

        return {
            i: _downsample_fn(q_ft_chunk_flat, block_size_hidden[i])
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(
                b,
                n_blocks,
                block_size_hidden[i],
                block_size_hidden[i],
                c,
            )
            for i in range(self.n_iters)
        }
