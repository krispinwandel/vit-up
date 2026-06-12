from typing import List, Optional, Any
import torch.nn as nn
import torch
import torchvision.transforms as T
from abc import abstractmethod


class UpsamplerBase(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def _normalize_img_size(self, img_size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(img_size, int):
            return img_size, img_size
        if len(img_size) != 2:
            raise ValueError("output_size must be an int or a tuple of length 2.")
        return int(img_size[0]), int(img_size[1])

    def _maybe_resize_pixel_values(
        self,
        pixel_values_bchw: torch.Tensor,
        input_size: Optional[tuple[int, int] | int] = None,
    ) -> torch.Tensor:
        if input_size is None:
            return pixel_values_bchw
        return T.Resize(
            size=self._normalize_img_size(input_size),
        )(pixel_values_bchw)

    @abstractmethod
    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int,
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Returns:
            last_hidden_states_bchw of shape (b,h,w,c)
        """
        raise NotImplementedError(
            "UpsamplerBase is an abstract class. Implement forward method in subclass."
        )

    @abstractmethod
    def pre_compute_cache(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int,
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
    ):
        # Optional method to pre-compute any cache data needed for forward pass.
        # By default, does nothing and returns None.
        return None
