import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dinov2_vit import DINOv2ViT


class PEInvBiasBlock(nn.Module):
    """Small residual MLP used to refine the learned position-bias field."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, pe_prev: torch.Tensor) -> torch.Tensor:
        update = self.mlp(pe_prev)
        return update


class PEInvBias(nn.Module):
    """PEInvBias module that inverts the learned position-bias field."""

    def __init__(self, mlp: nn.Module, layer_indices: List[int]):
        super().__init__()
        self.layer_indices = layer_indices
        self.register_buffer("pe", torch.zeros(1, 384, 37, 37))
        self.blocks = nn.ModuleList(
            [PEInvBiasBlock(mlp) for _ in range(len(layer_indices))]
        )
        self.size_to_inv_bias = {}  # Cache for interpolated position embeddings

    def _maybe_compute_inv_bias_at_size(
        self, out_h: int, out_w: int, force_recompute: bool = False
    ) -> List[torch.Tensor]:
        if (out_h, out_w) in self.size_to_inv_bias and not force_recompute:
            return self.size_to_inv_bias[(out_h, out_w)]
        inv_bias_layers = []
        pe = F.interpolate(
            self.pe, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        pe = pe.permute(0, 2, 3, 1)  # (1, out_h, out_w, c)
        for block in self.blocks:
            # pe = block(pe)
            # inv_bias_layers.append(pe)
            inv_bias_layers.append(block(pe))
        self.size_to_inv_bias[(out_h, out_w)] = inv_bias_layers
        return inv_bias_layers

    def forward(self, hidden_states_hwc: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            hidden_states_hwc: List[torch.Tensor] where each tensor has shape (b, h, w, c)
        """
        if len(hidden_states_hwc) != len(self.layer_indices):
            hidden_states_hwc = [hidden_states_hwc[i] for i in self.layer_indices]

        out_h, out_w = hidden_states_hwc[0].shape[
            1:3
        ]  # Get spatial dimensions from the first hidden state
        inv_bias_layers = self._maybe_compute_inv_bias_at_size(
            out_h, out_w, force_recompute=True
        )
        hidden_states_hwc_no_bias = []
        for i in range(len(self.layer_indices)):
            hidden_states_hwc_no_bias.append(hidden_states_hwc[i] + inv_bias_layers[i])
        return hidden_states_hwc_no_bias

    def load_weights_from_backbone(self, backbone: DINOv2ViT):
        embeddings = getattr(backbone, "embeddings", None)
        if embeddings is None or not hasattr(embeddings, "position_embeddings"):
            raise AttributeError(
                "Backbone does not expose embeddings.position_embeddings."
            )

        position_embeddings = embeddings.position_embeddings
        if position_embeddings.ndim != 3 or position_embeddings.shape[0] != 1:
            raise ValueError(
                "Expected backbone position embeddings with shape (1, N, D). "
                f"Got {tuple(position_embeddings.shape)}."
            )

        n_prefix_tokens = int(getattr(backbone, "n_prefix_tokens", 1))
        patch_position_embeddings = position_embeddings[:, n_prefix_tokens:, :]
        n_patches = int(patch_position_embeddings.shape[1])
        base_side = int(math.isqrt(n_patches))
        if base_side * base_side != n_patches:
            raise ValueError(
                "Backbone patch position embeddings are not square. "
                f"Got {n_patches} patch tokens."
            )

        patch_position_embeddings = patch_position_embeddings.reshape(
            1,
            base_side,
            base_side,
            patch_position_embeddings.shape[-1],
        ).permute(0, 3, 1, 2)

        print("patch_position_embeddings.shape", patch_position_embeddings.shape)

        if patch_position_embeddings.shape[1] != self.pe.shape[1]:
            raise ValueError(
                "PEInvBias channel dimension does not match the backbone position embeddings. "
                f"Got {self.pe.shape[1]} and {patch_position_embeddings.shape[1]}."
            )

        self.pe = patch_position_embeddings.detach().clone()
        self.size_to_inv_bias.clear()
