import math
from typing import Any, Dict, List, Optional, Tuple, cast
import torch
import torch.nn as nn

from vit_up.training.lightning_module import ViTUpPL
from vit_up.model.vit_up import ViTUp
from vit_up.layers.backbones.dino_vit_base import DinoViTBackboneBase
from vit_up.layers import continuous_rope
from vit_up.utils import grid_coords, spatial_tokens


def compute_backbone_layer_features_bhwc_by_scale(
    layer_indices: List[int],
    pixel_values: torch.Tensor,
    base_img_in_sizes: List[int],
    backbone: DinoViTBackboneBase,
    use_lora=False,
):
    """
    Args:
        pixel_values (torch.Tensor): The input pixel values for the images (b, c, h, w).
        base_img_in_sizes (List[int]): A list of input sizes for the base images.
        backbone (DinoViTBackboneBase): An instance of the DinoViTBackboneBase class for feature extraction.
    Returns:
        Dict[int, torch.Tensor]: A dictionary mapping each input size to its corresponding features in bhwc format.
    """

    features_bhwc_by_img_in_size = {}
    for img_size in base_img_in_sizes:
        if not use_lora:
            features_bhwc_by_img_in_size[img_size] = (
                DinoViTBackboneBase._compute_gt_features(
                    backbone=backbone,
                    pixel_values=pixel_values,
                    img_size=img_size,
                    layer_indices=layer_indices,
                    window_size=0,
                    flatten_hw_to_seq=False,
                )
            )
        else:
            features_bhwc_by_img_in_size[img_size] = (
                DinoViTBackboneBase._compute_lr_hidden_states(
                    backbone=backbone,
                    pixel_values=pixel_values,
                    img_size=img_size,
                    layer_indices=layer_indices,
                    window_size=0,
                )
            )
    return features_bhwc_by_img_in_size


def compute_query_coords_by_out_res(
    out_sizes: List[int],
):
    q_xy_normalized_by_out_size = {}
    for out_res in out_sizes:
        coords = torch.linspace(0.5, out_res - 0.5, out_res) / out_res
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        q_xy_normalized = torch.stack((grid_x, grid_y), dim=-1)  # (out_res, out_res, 2)
        q_xy_normalized_by_out_size[out_res] = q_xy_normalized
    return q_xy_normalized_by_out_size


def compute_layer_query_features(
    q_xy_normalized_by_out_size: Dict[int, torch.Tensor],
    pixel_values: torch.Tensor,
    vit_up_pl: ViTUpPL,
    hidden_layer_img_size=None,
    query_chunk_size=None,
) -> Dict[int, List[torch.Tensor]]:
    q_fts_by_out_size = {}
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    b = pixel_values.shape[0]
    for out_res in q_xy_normalized_by_out_size.keys():
        q_xy_normalized = q_xy_normalized_by_out_size[out_res].to(
            pixel_values.device
        )  # (out_res, out_res, 2)
        q_xy_normalized_flat = q_xy_normalized.reshape(1, -1, 2).expand(b, -1, -1)

        q_fts, _ = vit_up_pl(
            pixel_values=pixel_values,
            q_xy_normalized=q_xy_normalized_flat,  # (B, out_res*out_res, 2)
            hidden_layer_img_size=hidden_layer_img_size,
            query_chunk_size=query_chunk_size,
            return_all_layers=True,
        )
        q_fts = [
            q_fts_layer.reshape(b, out_res, out_res, -1) for q_fts_layer in q_fts
        ]  # (b, out_res, out_res, c)
        q_fts = [qft.squeeze() for qft in q_fts]  # remove batch dim if b=1
        q_fts_by_out_size[out_res] = q_fts
    return q_fts_by_out_size


def _extract_global_qkv(
    q_feature: torch.Tensor,
    hidden_state_hwc: torch.Tensor,
    global_context_module: nn.Module,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Project query/hidden tensors with global attention q/k/v modules when present."""
    hidden_flat = hidden_state_hwc.flatten(1, 2)

    q_proj = getattr(global_context_module, "q_proj", None)
    k_proj = getattr(global_context_module, "k_proj", None)
    v_proj = getattr(global_context_module, "v_proj", None)

    q_latent = q_proj(q_feature) if isinstance(q_proj, nn.Module) else None
    k_tokens = k_proj(hidden_flat) if isinstance(k_proj, nn.Module) else None
    v_tokens = v_proj(hidden_flat) if isinstance(v_proj, nn.Module) else None

    if q_latent is not None and q_latent.ndim != 3:
        q_latent = None
    if k_tokens is not None and k_tokens.ndim != 3:
        k_tokens = None
    if v_tokens is not None and v_tokens.ndim != 3:
        v_tokens = None

    return q_latent, k_tokens, v_tokens


def _module_at_runtime(modules: Any, index: int) -> nn.Module:
    if isinstance(modules, nn.ModuleList):
        return modules[index]
    if isinstance(modules, nn.Module):
        return modules
    raise TypeError(f"Expected an nn.Module or nn.ModuleList, got {type(modules)}.")


def _compute_vit_up_layers_and_meta(
    vit_up_model: ViTUp,
    pixel_values: torch.Tensor,
    q_xy_normalized_flat: torch.Tensor,
    hidden_layer_img_size: Optional[int],
    backbone: nn.Module,
) -> Tuple[
    List[torch.Tensor],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
]:
    """Reimplement ViTUp forward to expose per-layer q_latent/k/v used by cross-attn."""
    vit_up_runtime = cast(Any, vit_up_model)
    cache_data = vit_up_runtime.maybe_compute_cache_data(
        pixel_values=pixel_values,
        backbone=cast(Any, backbone),
        hidden_layer_img_size=hidden_layer_img_size,
        cache_data=None,
    )
    layer_hidden_states_hwc = cache_data["layer_hidden_states_hwc"]

    q_fts, _ = vit_up_runtime.compute_query_embedding(
        pixel_values=pixel_values,
        q_xy_normalized=q_xy_normalized_flat,
        backbone=cast(Any, backbone),
        query_embedding_cache_data=cache_data["query_embedding_cache_data"],
    )

    n_layers = len(layer_hidden_states_hwc)
    q_latent_layers: List[Optional[torch.Tensor]] = [None] * n_layers
    k_layers: List[Optional[torch.Tensor]] = [None] * n_layers
    v_layers: List[Optional[torch.Tensor]] = [None] * n_layers

    h_tokens, w_tokens = layer_hidden_states_hwc[0].shape[1:3]
    _, closest_idx, _, _ = spatial_tokens.fetch_closest_token(
        q_xy=q_xy_normalized_flat,
        hidden_state_hwc=layer_hidden_states_hwc[0],
    )
    rel_pos = grid_coords.compute_rel_position(
        q_xy=q_xy_normalized_flat,
        closest_idx=closest_idx,
        h_tokens=h_tokens,
        w_tokens=w_tokens,
    )
    rel_pos_enc = vit_up_runtime.rel_pos_enc(rel_pos)

    query_rope_embeddings = getattr(vit_up_runtime, "q_rope_embeddings", None)
    q_position_embeddings = None
    if query_rope_embeddings is not None:
        q_position_embeddings = continuous_rope.build_q_position_embeddings(
            h_tokens=h_tokens,
            w_tokens=w_tokens,
            q_xy_normalized=q_xy_normalized_flat,
            query_rope_embeddings=cast(nn.Module, vit_up_runtime.q_rope_embeddings),
        )

    decoder_mlp = getattr(vit_up_runtime, "decoder_mlp", None)
    if isinstance(decoder_mlp, (nn.ModuleList, nn.Module)):
        q_layers = [_module_at_runtime(decoder_mlp, 0)(q_fts)]
    else:
        q_layers = [q_fts]

    for i, hidden_state_hwc in enumerate(layer_hidden_states_hwc[1:]):
        vit_up_block = _module_at_runtime(vit_up_runtime.vit_up_blocks, i)
        transition_mlp = getattr(vit_up_block, "transition_mlp", None)
        q_for_global = q_fts
        if isinstance(transition_mlp, nn.Module):
            q_for_global = transition_mlp(q_for_global)

        global_module = getattr(vit_up_block, "cross_attention", None)
        if isinstance(global_module, nn.Module):
            q_latent, k_tokens, v_tokens = _extract_global_qkv(
                q_feature=q_for_global,
                hidden_state_hwc=hidden_state_hwc,
                global_context_module=global_module,
            )
            q_latent_layers[i + 1] = q_latent
            k_layers[i + 1] = k_tokens
            v_layers[i + 1] = v_tokens

        closest_token, closest_idx, _, _ = spatial_tokens.fetch_closest_token(
            q_xy=q_xy_normalized_flat,
            hidden_state_hwc=hidden_state_hwc,
        )
        q_fts = vit_up_block(
            q_fts=q_fts,
            hidden_state_hwc=hidden_state_hwc,
            closest_token=closest_token,
            rel_pos_enc=rel_pos_enc,
            q_position_embeddings=q_position_embeddings,
        )

        if isinstance(decoder_mlp, (nn.ModuleList, nn.Module)):
            q_layers.append(_module_at_runtime(decoder_mlp, i + 1)(q_fts))
        else:
            q_layers.append(q_fts)

    return q_layers, q_latent_layers, k_layers, v_layers


def compute_layer_query_features_and_meta(
    q_xy_normalized_by_out_size: Dict[int, torch.Tensor],
    pixel_values: torch.Tensor,
    vit_up_pl: ViTUpPL,
    hidden_layer_img_size: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Compute per-layer query features plus per-layer q_latent/k/v cross-attention tensors."""
    outputs: Dict[int, Dict[str, Any]] = {}
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    b = int(pixel_values.shape[0])

    vit_up_model = vit_up_pl.vit_up
    backbone = vit_up_pl.backbone

    for out_res, q_xy_normalized in q_xy_normalized_by_out_size.items():
        q_xy_normalized = q_xy_normalized.to(pixel_values.device)
        q_xy_normalized_flat = q_xy_normalized.reshape(1, -1, 2).expand(b, -1, -1)

        q_layers_bnq, q_latent_layers, k_layers, v_layers = (
            _compute_vit_up_layers_and_meta(
                vit_up_model=vit_up_model,
                pixel_values=pixel_values,
                q_xy_normalized_flat=q_xy_normalized_flat,
                hidden_layer_img_size=hidden_layer_img_size,
                backbone=backbone,
            )
        )

        q_layers_bhwc = [
            q_layer.reshape(b, out_res, out_res, -1) for q_layer in q_layers_bnq
        ]

        if b == 1:
            q_layers_bhwc = [q_layer.squeeze(0) for q_layer in q_layers_bhwc]
            q_latent_layers = [
                q_latent.squeeze(0) if q_latent is not None else None
                for q_latent in q_latent_layers
            ]
            k_layers = [
                k_tokens.squeeze(0) if k_tokens is not None else None
                for k_tokens in k_layers
            ]
            v_layers = [
                v_tokens.squeeze(0) if v_tokens is not None else None
                for v_tokens in v_layers
            ]

        outputs[out_res] = {
            "q_layer_features": q_layers_bhwc,
            "q_latent": q_latent_layers,
            "k": k_layers,
            "v": v_layers,
        }

    return outputs
