from __future__ import annotations

from typing import Any, List, Optional

import torch

from vit_up.layers.backbones.dino_vit_base import DinoViTBackboneBase
from vit_up.layers.backbones.dinov2_vit import DINOv2ViT
from vit_up.layers.backbones.dinov3_vit import DINOv3ViT

from .base import UpsamplerBase


def _contruct_backbone(backbone_model_name: str) -> DinoViTBackboneBase:
    backbone_name_lc = backbone_model_name.lower()
    if "v2" in backbone_name_lc:
        return DINOv2ViT.init_from_hf(backbone_model_name)
    if "v3" in backbone_name_lc:
        return DINOv3ViT.init_from_hf(backbone_model_name)
    raise ValueError(
        "Unable to infer backbone family from backbone_model_name. "
        "Expected the name to contain 'dinov2' or 'dinov3'. "
        f"Got: {backbone_model_name}"
    )


class AnyUpUpsampler(UpsamplerBase):
    def __init__(
        self,
        backbone: str,
        use_natten: bool = True,
        local_repo: str = "~/.cache/torch/hub/wimmerth_anyup_main",
        force_local: bool = False,
        name: str = "anyup",
        **kwargs,
    ):
        super().__init__(name=name)
        self.use_natten = use_natten
        self.local_repo = local_repo
        self.force_local = force_local
        self._device: Optional[torch.device] = None

        if not force_local:
            self.upsampler = torch.hub.load(
                "wimmerth/anyup",
                "anyup_multi_backbone",
                use_natten=self.use_natten,
            ).eval()
        else:
            self.upsampler = torch.hub.load(
                self.local_repo,
                "anyup_multi_backbone",
                use_natten=self.use_natten,
                source="local",
            ).eval()

        self.backbone = _contruct_backbone(backbone)
        self.backbone.eval()
        self.backbone.compile()
        self.patch_size = int(self.backbone.get_patch_size())

    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int,
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
    ) -> torch.Tensor:

        out_h, out_w = self._normalize_img_size(output_size)
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output_size must be positive.")

        device = pixel_values_bchw.device
        if self._device != device:
            self.upsampler.to(device)
            self.backbone.to(device)
            self._device = device

        with torch.no_grad():
            layer_hidden_states_bhwc = (
                DinoViTBackboneBase._compute_backbone_hidden_states(
                    backbone=self.backbone,
                    pixel_values=self._maybe_resize_pixel_values(
                        pixel_values_bchw, input_size
                    ),
                    img_size=None,
                    window_size=0,
                )
            )
            last_hidden_states = layer_hidden_states_bhwc[-1]
            last_hidden_states = last_hidden_states.permute(
                0, 3, 1, 2
            ).contiguous()  # bhwc -> bchw
            hr_features_anyup = self.upsampler(
                self._maybe_resize_pixel_values(pixel_values_bchw, output_size),
                last_hidden_states,
            )
            hr_features_anyup = hr_features_anyup.permute(
                0, 2, 3, 1
            ).contiguous()  # bchw -> bhwc

        return hr_features_anyup
