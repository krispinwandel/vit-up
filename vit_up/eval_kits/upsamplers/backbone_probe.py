from __future__ import annotations

from typing import Any, List, Optional, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from vit_up.layers.backbones.dino_vit_base import DinoViTBackboneBase
from vit_up.layers.backbones.dinov2_vit import DINOv2ViT
from vit_up.layers.backbones.dinov3_vit import DINOv3ViT

from .base import UpsamplerBase


def _normalize_optional_img_size(
    img_size: int | tuple[int, int] | None,
    name: str,
) -> tuple[int, int] | None:
    if img_size is None:
        return None
    if isinstance(img_size, int):
        if img_size <= 0:
            raise ValueError(f"`{name}` must be > 0 when provided as int.")
        return img_size, img_size
    if len(img_size) != 2:
        raise ValueError(f"`{name}` tuple must be (height, width).")
    h, w = int(img_size[0]), int(img_size[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"`{name}` values must be > 0.")
    return h, w


def _infer_hw(pixel_values_bchw: torch.Tensor, patch_size: int) -> tuple[int, int]:
    if pixel_values_bchw.ndim != 4:
        raise ValueError(
            "Expected pixel_values_bchw as BCHW, "
            f"got shape={tuple(pixel_values_bchw.shape)}"
        )
    return (
        pixel_values_bchw.shape[-2] // patch_size,
        pixel_values_bchw.shape[-1] // patch_size,
    )


def _resolve_backbone_class(backbone_model_name: str):
    model_name = str(backbone_model_name).strip().lower()
    if "dinov3" in model_name:
        return DINOv3ViT
    if "dinov2" in model_name:
        return DINOv2ViT
    raise ValueError(
        "Unable to infer backbone family from backbone_model_name. "
        "Expected model name to include 'dinov2' or 'dinov3'. "
        f"Got: {backbone_model_name}"
    )


class BackboneProbe(UpsamplerBase):
    """Return low-resolution backbone spatial features without learned upsampling."""

    def __init__(
        self,
        backbone_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        name: str = "backbone_probe",
        n_prefix_tokens: int | None = None,
        img_in_size: int | tuple[int, int] | None = None,
        out_size: int | tuple[int, int] | None = None,
        out_inter: str = "bilinear",
        **kwargs,
    ):
        super().__init__(name=name)
        del kwargs
        if not backbone_model_name:
            raise ValueError("`backbone_model_name` must be a non-empty string.")

        loaded_hf_model = cast(Any, AutoModel.from_pretrained(backbone_model_name))
        hf_model: Any = loaded_hf_model
        backbone = cast(
            Any,
            hf_model.base_model if hasattr(hf_model, "base_model") else hf_model,
        )
        if hasattr(backbone, "model"):
            backbone = backbone.model

        self.backbone = cast(nn.Module, backbone).eval()
        self.backbone.requires_grad_(False)
        self.backbone.compile()

        config = cast(Any, loaded_hf_model).config
        self.patch_size = int(getattr(config, "patch_size", 16))
        if self.patch_size <= 0:
            raise ValueError(
                f"Invalid patch_size={self.patch_size} from backbone config."
            )

        if n_prefix_tokens is None:
            n_register = int(getattr(config, "num_register_tokens", 0))
            n_prefix_tokens = 1 + n_register
        self.n_prefix_tokens = int(n_prefix_tokens)
        if self.n_prefix_tokens < 0:
            raise ValueError("`n_prefix_tokens` must be >= 0.")

        self.img_in_size = _normalize_optional_img_size(img_in_size, "img_in_size")
        self.out_size = _normalize_optional_img_size(out_size, "out_size")
        self.out_inter = out_inter

    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int | tuple[int, int],
        input_size: Optional[int | tuple[int, int]] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
    ) -> torch.Tensor:
        del output_size, layer_hidden_states_bhwc, cache_data
        if (
            self.img_in_size is not None
            and tuple(pixel_values_bchw.shape[-2:]) != self.img_in_size
        ):
            backbone_input = F.interpolate(
                pixel_values_bchw,
                size=self.img_in_size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            backbone_input = self._maybe_resize_pixel_values(
                pixel_values_bchw, input_size
            )
        h_embd, w_embd = _infer_hw(backbone_input, self.patch_size)
        out = self.backbone(backbone_input)

        if isinstance(out, dict):
            hidden_state = out["last_hidden_state"]
        else:
            hidden_state = out.last_hidden_state
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]

        spatial_tokens = hidden_state[:, self.n_prefix_tokens :, :]
        spatial_tokens_bchw = (
            spatial_tokens.reshape(spatial_tokens.shape[0], h_embd, w_embd, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        if self.out_size is not None:
            spatial_tokens_bchw = F.interpolate(
                spatial_tokens_bchw,
                size=self.out_size,
                mode=self.out_inter,
                align_corners=False,
            )
        return spatial_tokens_bchw.permute(0, 2, 3, 1).contiguous()


class BackboneDenoisedProbe(UpsamplerBase):
    """Return denoised low-resolution DINO spatial features."""

    def __init__(
        self,
        backbone_model_name: str,
        name: str = "backbone_denoised_probe",
        img_in_size: int | tuple[int, int] | None = None,
        # denoise_offsets_dists: Optional[list[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        del kwargs

        # if denoise_offsets_dists is None:
        #     denoise_offsets_dists = [1, 2, 3, 11]
        # self.denoise_offsets_dists = denoise_offsets_dists

        backbone_class = _resolve_backbone_class(backbone_model_name)
        self.backbone = backbone_class.init_from_hf(
            backbone_model_name=backbone_model_name,
            freeze_weights=True,
        )
        self.backbone.eval()
        self.patch_size = self.backbone.get_patch_size()
        self.n_prefix_tokens = self.backbone.n_prefix_tokens
        self.img_in_size = _normalize_optional_img_size(img_in_size, "img_in_size")

    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int | tuple[int, int],
        input_size: Optional[int | tuple[int, int]] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
    ) -> torch.Tensor:
        del output_size, layer_hidden_states_bhwc, cache_data
        backbone_input = self._maybe_resize_pixel_values(pixel_values_bchw, input_size)
        if (
            self.img_in_size is not None
            and tuple(backbone_input.shape[-2:]) != self.img_in_size
        ):
            backbone_input = F.interpolate(
                backbone_input,
                size=self.img_in_size,
                mode="bilinear",
                align_corners=False,
            )

        out = DinoViTBackboneBase._compute_gt_features(
            backbone=self.backbone,
            pixel_values=backbone_input,
            img_size=None,
            layer_indices=[-1],
            window_size=0,
            flatten_hw_to_seq=False,
        )
        return out[-1].contiguous()
