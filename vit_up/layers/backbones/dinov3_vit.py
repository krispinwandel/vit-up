import math
from typing import Any, Dict, List, Optional, Callable, cast, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTMLP,
    DINOv3ViTGatedMLP,
    DINOv3ViTLayerScale,
    DINOv3ViTEmbeddings,
    DINOv3ViTRopePositionEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from peft import LoraConfig, get_peft_model
from .dino_vit_base import DinoViTBackboneBase

try:
    from natten import na2d as natten_na2d
except ImportError:
    natten_na2d = None


def window_attn_natten(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    scaling: Optional[float] = None,
    n_prefix: int = 5,
):
    """
    Args:
        q/k/v: (B, H, S, Dh) where S = n_prefix + n_spatial and n_spatial = s_n * s_n
    Out:
        attn_output: (B, S, H, Dh)
    """
    B, num_heads, S, head_dim = q.shape
    attn_dtype = q.dtype
    k = k.to(attn_dtype)
    v = v.to(attn_dtype)
    n_spatial = S - n_prefix
    s_n = int(math.isqrt(n_spatial))
    if natten_na2d is None:
        raise ImportError(
            "NATTEN is not installed. Install it to use "
            "_forward_layer_with_window_size_natten."
        )
    # Prefix queries attend globally via stable fused SDPA.
    scale = head_dim ** (-0.5) if scaling is None else scaling
    q_prefix = q[:, :, :n_prefix, :]
    out_prefix = F.scaled_dot_product_attention(
        q_prefix,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )

    # Spatial queries use fused 2D neighborhood attention + additional prefix context.
    kernel_size = 2 * (window_size // 2) + 1
    if kernel_size > s_n:
        kernel_size = s_n

    q_spatial = q[:, :, n_prefix:, :].view(B, num_heads, s_n, s_n, head_dim)
    k_spatial = k[:, :, n_prefix:, :].view(B, num_heads, s_n, s_n, head_dim)
    v_spatial = v[:, :, n_prefix:, :].view(B, num_heads, s_n, s_n, head_dim)

    # NATTEN expects heads-last:
    # q/k/v: [B, X, Y, H, Dh], additional_{k,v}: [B, Lctx, H, Dh]
    q_na = q_spatial.permute(0, 2, 3, 1, 4).contiguous()
    k_na = k_spatial.permute(0, 2, 3, 1, 4).contiguous()
    v_na = v_spatial.permute(0, 2, 3, 1, 4).contiguous()
    add_k = k[:, :, :n_prefix, :].permute(0, 2, 1, 3).contiguous()
    add_v = v[:, :, :n_prefix, :].permute(0, 2, 1, 3).contiguous()

    out_spatial_na = natten_na2d(
        q_na,
        k_na,
        v_na,
        kernel_size=kernel_size,
        scale=scale,
        additional_keys=add_k,
        additional_values=add_v,
    )
    out_spatial = (
        out_spatial_na.permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(B, num_heads, n_spatial, head_dim)
    )

    attn_output = torch.cat([out_prefix, out_spatial], dim=2)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


class DINOv3ViTWindowAttention(nn.Module):
    """
    Multi-headed attention compatible with ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False

        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        window_size: int,
        return_key_value_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        num_tokens = query_states.shape[-2]
        num_patches = sin.shape[-2]
        num_prefix_tokens = num_tokens - num_patches  # cls token + register tokens

        if window_size > 0:
            attn_output = window_attn_natten(
                query_states,
                key_states,
                value_states,
                window_size,
                n_prefix=num_prefix_tokens,
                scaling=self.scaling,
            )
        else:
            if self.config._attn_implementation is None:
                raise ValueError(
                    "Attention implementation must be specified in the config to use non-windowed attention."
                )
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                dropout=0.0 if not self.training else self.dropout,
                scaling=self.scaling,
            )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if return_key_value_states:
            return attn_output, key_states, value_states
        return attn_output


class DINOv3ViTLayer(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTWindowAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        # NOTE we do not use drop path
        # self.drop_path = (
        #     DINOv3ViTDropPath(config.drop_path_rate)
        #     if config.drop_path_rate > 0.0
        #     else nn.Identity()
        # )
        # disable drop path
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(config)
        else:
            self.mlp = DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        window_size: int = 0,
        return_key_value_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        key_states: Optional[torch.Tensor]
        value_states: Optional[torch.Tensor]
        if return_key_value_states:
            hidden_states, key_states, value_states = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                self.attention(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    window_size=window_size,
                    return_key_value_states=True,
                ),
            )
        else:
            hidden_states = cast(
                torch.Tensor,
                self.attention(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    window_size=window_size,
                    return_key_value_states=False,
                ),
            )
            key_states = None
            value_states = None
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        if return_key_value_states:
            if key_states is None or value_states is None:
                raise RuntimeError(
                    "Expected key/value states when return_key_value_states=True."
                )
            return hidden_states, key_states, value_states
        return hidden_states


class DINOv3ViT(DinoViTBackboneBase):

    def __init__(
        self,
        dino_v3_config: DINOv3ViTConfig,
    ):
        super().__init__()
        self.dino_v3_config = dino_v3_config
        self.embeddings = DINOv3ViTEmbeddings(dino_v3_config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(dino_v3_config)
        self.layer = nn.ModuleList(
            [
                DINOv3ViTLayer(dino_v3_config)
                for _ in range(dino_v3_config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(
            dino_v3_config.hidden_size, eps=dino_v3_config.layer_norm_eps
        )
        self.n_prefix_tokens = 1 + int(
            getattr(dino_v3_config, "num_register_tokens", 4)
        )

    def get_patch_size(self) -> int:
        patch_size = int(getattr(self.dino_v3_config, "patch_size", 0))
        if patch_size <= 0:
            raise ValueError(
                "DINOv3ViT config has invalid patch_size. "
                f"Got: {getattr(self.dino_v3_config, 'patch_size', None)}"
            )
        return patch_size

    @classmethod
    def _init_backbone_from_hf(cls, backbone_model_name: str) -> "DINOv3ViT":
        from pathlib import Path

        local_files_only = Path(backbone_model_name).is_dir()
        loaded_hf_model = cast(
            Any,
            AutoModel.from_pretrained(
                backbone_model_name, local_files_only=local_files_only
            ),
        )
        hf_model: Any = loaded_hf_model
        model_config = cast(Any, loaded_hf_model).config

        if not isinstance(model_config, DINOv3ViTConfig):
            raise TypeError(
                "DINOv3ViT expects a DINOv3ViTConfig-compatible checkpoint. "
                f"Got: {type(model_config)}"
            )

        model = cls(model_config)
        try:
            hf_backbone = cls._unwrap_hf_backbone(hf_model)
            cls._load_from_hf_module_tree(model, hf_backbone)
        except AttributeError:
            cls._load_from_hf_state_dict(model, hf_model)
        return model

    @classmethod
    def _load_from_hf_module_tree(cls, model: "DINOv3ViT", hf_backbone: Any) -> None:
        cls._load_module(model.embeddings, hf_backbone.embeddings, strict=True)
        cls._load_module(
            model.rope_embeddings, hf_backbone.rope_embeddings, strict=True
        )
        hf_layers = cast(List[nn.Module], hf_backbone.layer)
        if len(hf_layers) != len(model.layer):
            raise ValueError(
                "Layer count mismatch between HF model and DINOv3ViT: "
                f"{len(hf_layers)} != {len(model.layer)}"
            )

        for dst_layer, src_layer in zip(model.layer, hf_layers):
            cls._load_module(dst_layer.norm1, src_layer.norm1)
            cls._load_module(dst_layer.attention, src_layer.attention)
            cls._load_module(dst_layer.layer_scale1, src_layer.layer_scale1)
            cls._load_module(dst_layer.norm2, src_layer.norm2)
            cls._load_module(dst_layer.mlp, src_layer.mlp)
            cls._load_module(dst_layer.layer_scale2, src_layer.layer_scale2)
        cls._load_module(model.norm, hf_backbone.norm)

    @classmethod
    def _load_from_hf_state_dict(cls, model: "DINOv3ViT", hf_model: Any) -> None:
        hf_state_dict = cast(Dict[str, torch.Tensor], hf_model.state_dict())
        model_state_keys = set(model.state_dict().keys())
        candidate_prefixes = (
            "",
            "model.",
            "base_model.",
            "base_model.model.",
            "dinov3_vit.",
            "base_model.dinov3_vit.",
            "base_model.model.dinov3_vit.",
        )

        best_state_dict: Dict[str, torch.Tensor] = {}
        best_prefix = ""
        best_matches = -1
        for prefix in candidate_prefixes:
            stripped_state_dict = {
                key.removeprefix(prefix): value
                for key, value in hf_state_dict.items()
                if key.startswith(prefix)
            }
            matches = sum(1 for key in stripped_state_dict if key in model_state_keys)
            if matches > best_matches:
                best_state_dict = stripped_state_dict
                best_prefix = prefix
                best_matches = matches

        if best_matches <= 0:
            sample_keys = ", ".join(list(hf_state_dict.keys())[:10])
            raise AttributeError(
                "Could not map Hugging Face DINOv3 state_dict to the local "
                f"DINOv3ViT module. Sample HF keys: {sample_keys}"
            )

        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                best_state_dict,
                strict=False,
                assign=True,
            )
        except TypeError:
            missing_keys, unexpected_keys = model.load_state_dict(
                best_state_dict,
                strict=False,
            )

        if missing_keys:
            raise AttributeError(
                "Could not load all required DINOv3 weights from the Hugging Face "
                f"state_dict using prefix {best_prefix!r}. Missing keys: "
                f"{list(missing_keys)[:20]}"
            )

    @classmethod
    def init_from_hf(
        cls,
        backbone_model_name: str,
        backbone_lora_config: Optional[LoraConfig] = None,
        freeze_weights: bool = True,
    ):
        def model_name_to_cache_path(model_name: str) -> str:
            return AutoModel.from_pretrained(
                model_name, cache_dir=None, local_files_only=True
            ).pretrained_model_archive_map.get(model_name, "")

        model: DINOv3ViT = cls._init_backbone_from_hf(backbone_model_name)
        if freeze_weights:
            model.requires_grad_(False)

        if backbone_lora_config is not None:
            if getattr(backbone_lora_config, "bias", None) in (None, "None"):
                backbone_lora_config.bias = "none"
            if getattr(backbone_lora_config, "init_lora_weights", None) != "loftq":
                backbone_lora_config.loftq_config = {}
            model = cast(
                DINOv3ViT, get_peft_model(cast(Any, model), backbone_lora_config)
            )
        return model

    def forward(
        self,
        pixel_values: torch.Tensor,
        window_size: int = 0,
        max_layer_index: int = -1,
    ) -> List[torch.Tensor]:
        w_embd = pixel_values.shape[2] // self.dino_v3_config.patch_size
        h_embd = pixel_values.shape[3] // self.dino_v3_config.patch_size

        def to_hwc(x):
            return x[:, self.n_prefix_tokens :, :].reshape(
                -1, h_embd, w_embd, x.shape[-1]
            )

        pos_embds = self.rope_embeddings(pixel_values)
        hidden_state_raw = self.embeddings(pixel_values)

        # Indexing convention: 0 is embeddings output, i>0 is output after layer i.
        hidden_states_hwc: List[torch.Tensor] = [to_hwc(hidden_state_raw)]
        if max_layer_index == -1:
            max_layer_index = len(self.layer)
        for layer_idx in range(max_layer_index):
            hidden_state_raw = cast(
                torch.Tensor,
                self.layer[layer_idx](
                    hidden_state_raw,
                    pos_embds,
                    window_size=window_size,
                ),
            )
            if layer_idx + 1 < len(self.layer):
                hidden_state_normed = cast(
                    DINOv3ViTLayer,
                    self.layer[layer_idx + 1],
                ).norm1(hidden_state_raw)
            else:
                hidden_state_normed = cast(torch.Tensor, self.norm(hidden_state_raw))
            hidden_states_hwc.append(to_hwc(hidden_state_normed))

        return hidden_states_hwc
