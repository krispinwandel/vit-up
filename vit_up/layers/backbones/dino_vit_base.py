from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, List, Optional, cast, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def make_circular_offsets(offsets_dists: list[int]):
    offsets = [[0, 0]]
    for i, offset_dist in enumerate(offsets_dists):
        sign_x = 1 if i % 2 == 0 else -1
        sign_y = 1 if i // 2 % 2 == 0 else -1
        offsets.append([sign_x * offset_dist, sign_y * offset_dist])
    return torch.tensor(offsets)


class DinoViTBackboneBase(nn.Module, ABC):
    """Shared utility methods for DINO ViT backbones."""

    @abstractmethod
    def get_patch_size(self) -> int:
        """Return the model patch size as a positive integer."""
        raise NotImplementedError

    def get_num_layers(self) -> int:
        """Return the number of layers, including embedding layer, in the backbone as a positive integer."""
        return len(self.layer) + 1

    @staticmethod
    def _load_module(dst_module: Any, src_module: Any, strict: bool = True) -> None:
        if not isinstance(dst_module, nn.Module) or not isinstance(
            src_module, nn.Module
        ):
            raise TypeError(
                "_load_module expects torch.nn.Module instances for destination "
                "and source modules."
            )
        src_state_dict = src_module.state_dict()
        try:
            dst_module.load_state_dict(src_state_dict, strict=strict, assign=True)
        except TypeError:
            dst_module.load_state_dict(src_state_dict, strict=strict)

    @staticmethod
    def _unwrap_hf_backbone(hf_model: Any) -> Any:
        required_attrs = ("embeddings", "layer")

        def has_backbone_interface(candidate: Any) -> bool:
            return all(hasattr(candidate, attr) for attr in required_attrs)

        candidates = [hf_model]

        base_model = getattr(hf_model, "base_model", None)
        if base_model is not None and base_model is not hf_model:
            candidates.append(base_model)

        for candidate in list(candidates):
            nested_model = getattr(candidate, "model", None)
            if nested_model is not None and nested_model is not candidate:
                candidates.append(nested_model)

        for candidate in candidates:
            if has_backbone_interface(candidate):
                return candidate

        return cast(Any, base_model if base_model is not None else hf_model)

    @staticmethod
    def _compute_backbone_hidden_states(
        backbone: "DinoViTBackboneBase",
        pixel_values: torch.Tensor,
        img_size: Optional[int] = None,
        window_size: int = 0,
    ) -> List[torch.Tensor]:
        if img_size is not None and int(img_size) <= 0:
            raise ValueError(f"img_size must be > 0 when provided. Got {img_size}.")

        backbone_input = pixel_values
        if img_size is not None and tuple(pixel_values.shape[-2:]) != (
            img_size,
            img_size,
        ):
            backbone_input = F.interpolate(
                pixel_values,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )

        use_autocast = backbone_input.device.type in (
            "cuda",
            "xpu",
        ) and backbone_input.dtype in (torch.float16, torch.bfloat16)
        autocast_ctx = (
            torch.autocast(
                dtype=backbone_input.dtype,
                device_type=backbone_input.device.type,
            )
            if use_autocast
            else nullcontext()
        )
        with autocast_ctx:
            out = backbone(
                pixel_values=backbone_input,
                window_size=window_size,
            )
        return cast(List[torch.Tensor], out)

    @staticmethod
    def _compute_rolled_hidden_states_brhwc(
        backbone: "DinoViTBackboneBase",
        pixel_values: torch.Tensor,
        roll_offsets: torch.Tensor,
        img_size: Optional[int] = None,
        window_size: int = 0,
        layer_indices: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        if layer_indices is None:
            layer_indices = list(range(backbone.get_num_layers()))

        # NOTE important to resize here for rolled input computation
        if img_size is not None:
            pixel_values = T.Resize(img_size)(pixel_values)

        patch_size = backbone.get_patch_size()
        dydx_tokens = []
        hidden_states_bhwc_roll_samples = []
        n_r = len(roll_offsets)
        for si in range(n_r):
            # Offset in token units.
            roll_token_offset = roll_offsets[si]

            dy_tokens = int(roll_token_offset[0].item())
            dx_tokens = int(roll_token_offset[1].item())

            dydx_tokens.append((dy_tokens, dx_tokens))

            # Equivalent offset in pixel units for the image.
            dy_pixels = dy_tokens * patch_size
            dx_pixels = dx_tokens * patch_size

            rolled_input = torch.roll(
                pixel_values,
                shifts=(dy_pixels, dx_pixels),
                dims=(2, 3),
            )

            hidden_states_hwc_rolled_all_layers = (
                DinoViTBackboneBase._compute_backbone_hidden_states(
                    backbone=backbone,
                    pixel_values=rolled_input,
                    img_size=img_size,
                    window_size=window_size,
                )
            )
            hidden_states_bhwc_roll_samples.append(
                [hidden_states_hwc_rolled_all_layers[i] for i in layer_indices]
            )

        hidden_states_brhwc = [
            torch.stack(
                [hidden_states_bhwc_roll_samples[si][li] for si in range(n_r)],
                dim=1,
            )
            for li in range(len(layer_indices))
        ]

        return hidden_states_brhwc, dydx_tokens

    @staticmethod
    def _select_hidden_layers(
        hidden_states: List[torch.Tensor],
        layer_indices: List[int],
    ) -> List[torch.Tensor]:
        max_idx = len(hidden_states) - 1
        selected: List[torch.Tensor] = []
        for layer_idx in layer_indices:
            idx = int(layer_idx)
            if idx < 0 or idx > max_idx:
                raise ValueError(
                    "layer_indices contains out-of-range values. "
                    f"Expected index in [0, {max_idx}], got {idx}."
                )
            selected.append(hidden_states[idx])
        return selected

    @staticmethod
    def _compute_lr_hidden_states(
        backbone: "DinoViTBackboneBase",
        pixel_values: torch.Tensor,
        layer_indices: List[int],
        img_size: Optional[int] = None,
        window_size: int = 0,
    ) -> List[torch.Tensor]:
        hidden_states = DinoViTBackboneBase._compute_backbone_hidden_states(
            backbone=backbone,
            pixel_values=pixel_values,
            img_size=img_size,
            window_size=window_size,
        )
        return DinoViTBackboneBase._select_hidden_layers(hidden_states, layer_indices)

    @staticmethod
    def _resolve_patch_size(backbone: "DinoViTBackboneBase") -> int:
        return int(backbone.get_patch_size())

    @staticmethod
    def _compute_gt_features(
        backbone: "DinoViTBackboneBase",
        pixel_values: torch.Tensor,
        layer_indices: List[int],
        img_size: Optional[int] = None,
        window_size: int = 0,
        flatten_hw_to_seq: bool = True,
    ) -> List[torch.Tensor]:
        """Compute ground truth features from backbone.

        Args:
            backbone: DinoViTBackboneBase instance
            pixel_values: Input images tensor
            layer_indices: Indices of layers to extract features from
            img_size: Optional target image size
            window_size: Window size for computation (default 0)
            flatten_hw_to_seq: Whether to flatten spatial dimensions to sequence

        Returns:
            List of feature tensors from selected layers
        """
        disable_adapter = getattr(backbone, "disable_adapter", None)
        disable_lora_ctx = cast(
            Any,
            disable_adapter() if callable(disable_adapter) else nullcontext(),
        )
        with torch.no_grad(), disable_lora_ctx:
            hidden_states = DinoViTBackboneBase._compute_backbone_hidden_states(
                backbone=backbone,
                pixel_values=pixel_values,
                img_size=img_size,
                window_size=window_size,
            )
            selected_layers = DinoViTBackboneBase._select_hidden_layers(
                hidden_states,
                layer_indices,
            )

        if flatten_hw_to_seq:
            selected_layers = [
                layer_hwc.reshape(layer_hwc.shape[0], -1, layer_hwc.shape[-1])
                for layer_hwc in selected_layers
            ]
        return selected_layers
