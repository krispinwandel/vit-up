"""Query encoder: sample DINO patch embeddings at query coordinates."""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEncoderBase(nn.Module):
    """Base class for query encoders."""

    def __init__(self, img_in_size: int, window_size: int, layer_index: int):
        super().__init__()
        self.img_in_size = img_in_size
        self.window_size = window_size
        self.layer_index = layer_index

    def compute_cache_data(
        self,
        pixel_values: torch.Tensor,
        backbone: nn.Module,
    ) -> Any:
        """
        Optional method to precompute and return any cache data that can be reused across multiple forward passes.
        This can be used to avoid redundant computation when the same image features are used for multiple query sets.
        """
        pixel_values = F.interpolate(
            pixel_values,
            size=(self.img_in_size, self.img_in_size),
            mode="bilinear",
            align_corners=False,
        )
        last_hidden_state_hwc = backbone(
            pixel_values,
            window_size=self.window_size,
            max_layer_index=self.layer_index,
        )[
            -1
        ]  # (B, H_tokens, W_tokens, D)
        return {
            "last_hidden_state_hwc": last_hidden_state_hwc,
        }

    def maybe_compute_cache_data(
        self,
        pixel_values: torch.Tensor,
        backbone: Optional[nn.Module] = None,
        cache_data: Any = None,
    ) -> Any:
        if cache_data is None:
            if backbone is None:
                raise ValueError(
                    "backbone must be provided if query_encoder_cache_data is not given."
                )
            cache_data = self.compute_cache_data(
                pixel_values=pixel_values,
                backbone=backbone,
            )
        return cache_data

    def forward(
        self,
        pixel_values: torch.Tensor,
        q_xy_normalized: torch.Tensor,
        backbone: Optional[nn.Module] = None,
        cache_data: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            pixel_values: (B, C_in, H, W) input images
            q_xy_normalized: (B, N_q, 2) normalized query coordinates in [0, 1], where N_q is the number of query points.
        Returns:
            query_features: (B, N_q, D) features for each query point
        """
        raise NotImplementedError


class QueryEncoder(QueryEncoderBase):
    """
    Query encoder that samples DINO patch embeddings directly.

    This uses layer index 0 from the backbone cache, which corresponds to the
    patch embedding output before transformer blocks.
    """

    def __init__(
        self,
        img_in_size: int,
        window_size: int = 0,
        layer_index: int = 0,
        out_proj_module: Optional[nn.Module] = None,
    ):
        if int(layer_index) != 0:
            raise ValueError(
                "QueryEncoder samples patch embeddings directly and requires "
                f"layer_index=0. Got {layer_index}."
            )
        super().__init__(
            img_in_size=img_in_size,
            window_size=window_size,
            layer_index=0,
        )
        self.out_proj = (
            out_proj_module if out_proj_module is not None else nn.Identity()
        )

    @staticmethod
    def _build_grid_sample_coords(q_xy_normalized: torch.Tensor) -> torch.Tensor:
        if q_xy_normalized.ndim != 3 or q_xy_normalized.shape[-1] != 2:
            raise ValueError(
                "q_xy_normalized must have shape (B, N_q, 2). "
                f"Got {tuple(q_xy_normalized.shape)}."
            )
        return (q_xy_normalized * 2.0 - 1.0).reshape(
            q_xy_normalized.shape[0],
            q_xy_normalized.shape[1],
            1,
            2,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        q_xy_normalized: torch.Tensor,
        backbone: Optional[nn.Module] = None,
        query_encoder_cache_data: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        query_encoder_cache_data = self.maybe_compute_cache_data(
            pixel_values=pixel_values,
            backbone=backbone,
            cache_data=query_encoder_cache_data,
        )
        patch_features_hwc = query_encoder_cache_data["last_hidden_state_hwc"]
        patch_features_bchw = patch_features_hwc.permute(0, 3, 1, 2).contiguous()
        sample_grid = self._build_grid_sample_coords(q_xy_normalized).to(
            device=patch_features_bchw.device,
            dtype=patch_features_bchw.dtype,
        )

        q_features = F.grid_sample(
            patch_features_bchw,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        q_features = q_features.squeeze(-1).transpose(1, 2).contiguous()
        q_features = self.out_proj(q_features)

        return q_features, query_encoder_cache_data
