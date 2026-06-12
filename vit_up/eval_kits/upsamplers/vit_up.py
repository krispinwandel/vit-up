import torch
from typing import List, Optional, Any

from vit_up.inference.vit_up_wrapper import ViTUpWrapper
from vit_up.utils.grid_coords import build_regular_grid_centers_xy

from .base import UpsamplerBase


class ViTUpUpsampler(UpsamplerBase):

    def __init__(
        self,
        model_name: str = "vit_up_dinov3_splus",
        name: str = "vit_up",
        out_size: Optional[int] = None,
        amp_dtype=torch.bfloat16,
        query_chunk_size=3136,
        hidden_layer_img_size: int = 448,
        device: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        if ckpt_path is not None:
            raise ValueError(
                "ViTUpUpsampler no longer loads Lightning checkpoints directly. "
                "Pass model_name='vit_up_dinov3_base' or "
                "model_name='vit_up_dinov3_splus' instead."
            )

        use_bfloat16 = amp_dtype == torch.bfloat16
        self.vit_up = ViTUpWrapper(
            model_name=model_name,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            use_bfloat16=use_bfloat16,
            hidden_layer_img_size=hidden_layer_img_size,
            query_chunk_size=query_chunk_size,
        ).eval()
        self.amp_dtype = amp_dtype
        self.out_size = out_size
        self._q_xy_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self.query_chunk_size = query_chunk_size
        self.hidden_layer_img_size = hidden_layer_img_size

    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int | tuple[int, int],
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
        query_chunk_size=None,
    ):
        query_chunk_size = query_chunk_size or self.query_chunk_size
        pixel_values_bchw = self._maybe_resize_pixel_values(
            pixel_values_bchw, input_size
        )
        # NOTE currently the model only supports square output sizes, so we take the first element if a tuple is provided
        output_size = self._normalize_img_size(output_size)[0]
        output_size = (
            min(self.out_size, output_size)
            if self.out_size is not None
            else output_size
        )
        q_device = pixel_values_bchw.device
        q_dtype = (
            self.amp_dtype
            if pixel_values_bchw.is_cuda and self.amp_dtype is not None
            else pixel_values_bchw.dtype
        )
        cache_key = (output_size, q_device, q_dtype)
        q_xy_normalized = self._q_xy_cache.get(cache_key)
        if (
            q_xy_normalized is None
            or q_xy_normalized.shape[0] != pixel_values_bchw.shape[0]
        ):
            q_xy_normalized = build_regular_grid_centers_xy(output_size).to(
                device=q_device
            )
            q_xy_normalized = q_xy_normalized.reshape(1, output_size * output_size, 2)
            q_xy_normalized = q_xy_normalized.expand(pixel_values_bchw.shape[0], -1, -1)
            q_xy_normalized = q_xy_normalized.to(dtype=q_dtype)
            self._q_xy_cache[cache_key] = q_xy_normalized
        q_fts = self.vit_up(
            images=pixel_values_bchw,
            query_coords=q_xy_normalized,
            hidden_layer_img_size=self.hidden_layer_img_size,
            query_chunk_size=query_chunk_size,
        )
        b, n_q, c = q_fts.shape
        q_fts = q_fts.reshape(b, output_size, output_size, c)
        return q_fts.float()
