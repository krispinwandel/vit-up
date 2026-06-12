"""
ViT-Up: Vision Transformer Upsampling
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn

from ..layers import continuous_rope
from ..layers.query_encoder import QueryEncoderBase
from ..layers.backbones.dino_vit_base import DinoViTBackboneBase
from vit_up.utils import grid_coords, spatial_tokens


class ViTUpBlock(nn.Module):
    """One ViTUp transition from one backbone layer feature space to the next."""

    def __init__(
        self,
        transition_mlp: Optional[nn.Module],
        cross_attention: Optional[nn.Module],
        featx: Optional[nn.Module],
        mlp: nn.Module,
        dim: int,
        dim_kv: Optional[int] = None,
        dim_h: Optional[int] = None,
    ):
        super().__init__()
        self.transition_mlp = transition_mlp
        self.cross_attention = cross_attention
        self.featx = featx

        self.norm_q_global = nn.LayerNorm(dim)
        if dim_h is not None and dim_kv is None:
            dim_kv = dim_h
        if dim_kv is None:
            dim_kv = dim
        self.norm_h_local = nn.LayerNorm(dim_kv)
        self.norm_h_global = nn.LayerNorm(dim_kv)
        self.norm_post = nn.LayerNorm(dim)
        self.mlp = mlp

    @property
    def global_cross_attention(self) -> Optional[nn.Module]:
        return self.cross_attention

    def forward(
        self,
        q_fts: torch.Tensor,
        hidden_state_hwc: torch.Tensor,
        closest_token: torch.Tensor,
        rel_pos_enc: torch.Tensor,
        q_position_embeddings: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        if self.transition_mlp is not None:
            q_fts = self.transition_mlp(q_fts)

        if self.cross_attention is not None:
            q_out_global = self.cross_attention(
                query_feature=self.norm_q_global(q_fts),
                hidden_states=self.norm_h_global(hidden_state_hwc.flatten(1, 2)),
                position_embeddings=q_position_embeddings,
            )
            q_fts = q_out_global + q_fts

        if self.featx is not None:
            q_fts = q_fts + self.norm_h_local(
                self.featx(closest_token, rel_pos_enc)
            )

        q_fts = q_fts + self.mlp(self.norm_post(q_fts))
        return q_fts


class ViTUp(nn.Module):
    """ViTUp: local-global feature upsampling over selected backbone layers."""

    def __init__(
        self,
        layer_indices: list[int],
        query_embedding: QueryEncoderBase,
        rel_pos_enc: nn.Module,
        vit_up_blocks: nn.ModuleList,
        decoder_mlp: Optional[Union[nn.ModuleList, nn.Module]] = None,
        q_rope_embeddings: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layer_indices = [int(x) for x in layer_indices]

        self.query_embedding: QueryEncoderBase = query_embedding

        self.rel_pos_enc = rel_pos_enc
        self.vit_up_blocks = vit_up_blocks

        self.q_rope_embeddings = q_rope_embeddings
        self.decoder_mlp = decoder_mlp

    @staticmethod
    def _module_at(
        modules: Union[nn.ModuleList, nn.Module],
        index: int,
    ) -> nn.Module:
        if isinstance(modules, nn.ModuleList):
            return modules[index]
        return modules

    def compute_cache_data(
        self,
        pixel_values: torch.Tensor,
        backbone: DinoViTBackboneBase,
        hidden_layer_img_size: Optional[int] = None,
    ) -> dict:
        layer_hidden_states_hwc = DinoViTBackboneBase._compute_lr_hidden_states(
            backbone=backbone,
            pixel_values=pixel_values,
            layer_indices=self.layer_indices,
            img_size=hidden_layer_img_size,
            window_size=0,
        )
        query_embedding_cache_data = self.query_embedding.compute_cache_data(
            pixel_values=pixel_values, backbone=backbone
        )
        return {
            "layer_hidden_states_hwc": layer_hidden_states_hwc,
            "query_embedding_cache_data": query_embedding_cache_data,
        }

    def maybe_compute_cache_data(
        self,
        pixel_values: torch.Tensor,
        backbone: Optional[DinoViTBackboneBase] = None,
        hidden_layer_img_size: Optional[int] = None,
        cache_data: Optional[dict] = None,
    ):
        if cache_data is not None:
            return cache_data
        if backbone is None:
            raise ValueError("backbone must be provided if cache_data is not provided.")
        return self.compute_cache_data(
            pixel_values=pixel_values,
            backbone=backbone,
            hidden_layer_img_size=hidden_layer_img_size,
        )

    def compute_query_embedding(
        self,
        pixel_values,
        q_xy_normalized,
        backbone=None,
        query_embedding_cache_data=None,
    ):
        query_embedding_cache_data = self.query_embedding.maybe_compute_cache_data(
            pixel_values=pixel_values,
            backbone=backbone,
            cache_data=query_embedding_cache_data,
        )
        q_fts, query_embedding_cache_data = self.query_embedding(
            pixel_values=pixel_values,
            q_xy_normalized=q_xy_normalized,
            backbone=backbone,
            query_encoder_cache_data=query_embedding_cache_data,
        )
        return q_fts, query_embedding_cache_data

    def forward(
        self,
        pixel_values: torch.Tensor,
        q_xy_normalized: torch.Tensor,
        backbone: Optional[DinoViTBackboneBase] = None,
        hidden_layer_img_size: Optional[int] = None,
        cache_data: Any = None,
    ):
        cache_data = self.maybe_compute_cache_data(
            pixel_values=pixel_values,
            backbone=backbone,
            hidden_layer_img_size=hidden_layer_img_size,
            cache_data=cache_data,
        )
        layer_hidden_states_hwc = cache_data["layer_hidden_states_hwc"]

        q_fts, _ = self.compute_query_embedding(
            pixel_values=pixel_values,
            q_xy_normalized=q_xy_normalized,
            backbone=backbone,
            query_embedding_cache_data=cache_data["query_embedding_cache_data"],
        )

        if self.decoder_mlp is not None:
            q_fts_decoded = self._module_at(self.decoder_mlp, 0)(q_fts)
        else:
            q_fts_decoded = q_fts
        layer_q_ft = [q_fts_decoded]

        h_tokens, w_tokens = layer_hidden_states_hwc[0].shape[1:3]
        _, closest_idx, _, _ = spatial_tokens.fetch_closest_token(
            q_xy=q_xy_normalized,
            hidden_state_hwc=layer_hidden_states_hwc[0],
        )

        rel_pos = grid_coords.compute_rel_position(
            q_xy=q_xy_normalized,
            closest_idx=closest_idx,
            h_tokens=h_tokens,
            w_tokens=w_tokens,
        )
        rel_pos_enc = self.rel_pos_enc(rel_pos)  # (b, n_q, fourier_dim)

        q_position_embeddings = None
        if self.q_rope_embeddings is not None:
            q_position_embeddings = continuous_rope.build_q_position_embeddings(
                h_tokens=h_tokens,
                w_tokens=w_tokens,
                q_xy_normalized=q_xy_normalized,
                query_rope_embeddings=self.q_rope_embeddings,
            )

        for i, hidden_state_hwc in enumerate(layer_hidden_states_hwc[1:]):
            vit_up_block = self._module_at(self.vit_up_blocks, i)
            closest_token, closest_idx, _, _ = spatial_tokens.fetch_closest_token(
                q_xy=q_xy_normalized,
                hidden_state_hwc=hidden_state_hwc,
            )
            q_fts = vit_up_block(
                q_fts=q_fts,
                hidden_state_hwc=hidden_state_hwc,
                closest_token=closest_token,
                rel_pos_enc=rel_pos_enc,
                q_position_embeddings=q_position_embeddings,
            )
            if self.decoder_mlp is not None:
                q_fts_decoded = self._module_at(self.decoder_mlp, i + 1)(q_fts)
            else:
                q_fts_decoded = q_fts
            layer_q_ft.append(q_fts_decoded)

        return layer_q_ft
