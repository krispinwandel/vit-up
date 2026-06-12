from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch


class SupervisionStrategy(ABC):

    def __init__(
        self,
        block_chunk_size: int,
        patch_size: int,
        use_random_roll_for_gt: bool = False,
        use_denoised_gt: bool = False,
        denoise_offsets_dists: Optional[List[int]] = None,
        use_full_image: bool = False,
    ) -> None:
        super().__init__()
        self.block_chunk_size = block_chunk_size
        self.patch_size = patch_size
        self.use_random_roll_for_gt = use_random_roll_for_gt
        self.use_full_image = use_full_image
        self.use_denoised_gt = use_denoised_gt
        self.denoise_offsets_dists = denoise_offsets_dists
        self._blocks = self._init_blocks()

    @abstractmethod
    def gt_img_sizes(self) -> Dict[int, int]:
        """
        Returns a dict mapping scale -> gt image size (assumed square).
        """
        raise NotImplementedError

    @abstractmethod
    def build_query_xy_grid(self) -> torch.Tensor:
        """
        Returns a tensor of shape (n_q, n_q, 2) containing the (x,y) coordinates of each query token in normalized [0,1] coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def block_size_query(self) -> int:
        """
        Returns the block size of query tokens. For example, if block_size_query=2, then each 2x2 block of query tokens corresponds to one gt cell.
        """
        raise NotImplementedError

    @abstractmethod
    def block_size_hidden(self) -> Dict[int, int]:
        """
        Returns a dict mapping scale -> block size of hidden tokens. For example, if block_size_hidden[scale]=4, then each 4x4 block of hidden tokens corresponds to one gt cell at that scale.
        """
        raise NotImplementedError

    @abstractmethod
    def query_features_to_hidden(
        self, q_ft_chunk: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Args:
            q_ft_chunk: (b, n_blocks, block_size_q, block_size_q, c)
        Returns:
            Dict[int, torch.Tensor] where tensors have shape (b, n_blocks, block_size_h, block_size_h, c)
        """
        raise NotImplementedError

    def _init_blocks(self):
        """

        Returns dict:
            - "q_xy_blocks": List[torch.Tensor] of length n_chunks,
                each tensor has shape (n_blocks_chunk, block_size_q, block_size_q, 2)
            - "hidden_ij_blocks": Dict[int, List[torch.Tensor]] mapping scale
                -> list of length n_chunks, each tensor has shape
                (n_blocks_chunk, block_size_h, block_size_h, 2)
                containing the (i,j) indices of gt cells corresponding
                to each query
        """
        query_xy_grid = self.build_query_xy_grid()
        block_size_query = self.block_size_query()
        n_q = query_xy_grid.shape[0]
        q_xy_blocks = []
        for i in range(0, n_q, block_size_query):
            for j in range(0, n_q, block_size_query):
                chunk = query_xy_grid[
                    i : i + block_size_query, j : j + block_size_query, :
                ]
                q_xy_blocks.append(chunk)

        block_size_hidden_dict = self.block_size_hidden()
        gt_img_sizes = self.gt_img_sizes()
        hidden_ij_blocks = {}
        for scale, block_size_hidden in block_size_hidden_dict.items():
            gt_img_size = gt_img_sizes[scale]
            embd_size = gt_img_size // self.patch_size
            hidden_ij_blocks[scale] = []
            for i in range(0, embd_size, block_size_hidden):
                for j in range(0, embd_size, block_size_hidden):
                    hidden_ij_chunk = torch.stack(
                        torch.meshgrid(
                            torch.arange(i, i + block_size_hidden),
                            torch.arange(j, j + block_size_hidden),
                            indexing="ij",
                        ),
                        dim=-1,
                    )
                    hidden_ij_blocks[scale].append(hidden_ij_chunk)

        assert all(
            len(q_xy_blocks) == len(hidden_ij_blocks[scale])
            for scale in hidden_ij_blocks
        ), "Number of blocks must match between query and hidden for all scales."

        # chunk blocks and stack into tensors
        q_xy_blocks = torch.stack(q_xy_blocks, dim=0)
        hidden_ij_blocks = {
            scale: torch.stack(hidden_ij_block, dim=0)
            for scale, hidden_ij_block in hidden_ij_blocks.items()
        }
        return {
            "q_xy_blocks": q_xy_blocks,
            "hidden_ij_blocks": hidden_ij_blocks,
        }

    def get_transformed_q_xy_blocks(self, bbox_x1y2x2y2: torch.Tensor):
        """
        Args:
            bbox_x1y2x2y2: (b, 4)
        """
        q_xy_blocks = self._blocks["q_xy_blocks"].to(
            device=bbox_x1y2x2y2.device, dtype=bbox_x1y2x2y2.dtype
        )
        batch_q_xy_blocks = q_xy_blocks.unsqueeze(0).expand(
            bbox_x1y2x2y2.shape[0], -1, -1, -1, -1
        )
        bbox_w = bbox_x1y2x2y2[:, 2] - bbox_x1y2x2y2[:, 0]
        bbox_h = bbox_x1y2x2y2[:, 3] - bbox_x1y2x2y2[:, 1]
        batch_q_xy_blocks_transformed = batch_q_xy_blocks.clone()
        batch_q_xy_blocks_transformed[..., 0] = (
            bbox_x1y2x2y2[:, 0:1, None, None]
            + batch_q_xy_blocks[..., 0] * bbox_w[:, None, None, None]
        )
        batch_q_xy_blocks_transformed[..., 1] = (
            bbox_x1y2x2y2[:, 1:2, None, None]
            + batch_q_xy_blocks[..., 1] * bbox_h[:, None, None, None]
        )
        return batch_q_xy_blocks_transformed

    def get_hidden_ij_blocks(self):
        return self._blocks["hidden_ij_blocks"]
