import math
import types
from typing import Any, List, Optional, Type, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Embeddings,
    Dinov2Layer,
)

from .dino_vit_base import DinoViTBackboneBase

try:
    from transformers.models.dinov2_with_registers.configuration_dinov2_with_registers import (
        Dinov2WithRegistersConfig,
    )
    from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
        Dinov2WithRegistersEmbeddings,
        Dinov2WithRegistersLayer,
    )
except Exception:
    Dinov2WithRegistersConfig = None
    Dinov2WithRegistersEmbeddings = None
    Dinov2WithRegistersLayer = None


class DINOv2ViT(DinoViTBackboneBase):
    def __init__(self, dino_v2_config: Any):
        super().__init__()
        self.dino_v2_config = dino_v2_config

        config_model_type = str(getattr(dino_v2_config, "model_type", "")).lower()
        use_with_registers_impl = "dinov2_with_registers" in config_model_type or (
            Dinov2WithRegistersConfig is not None
            and isinstance(dino_v2_config, Dinov2WithRegistersConfig)
        )

        if use_with_registers_impl:
            if (
                Dinov2WithRegistersEmbeddings is None
                or Dinov2WithRegistersLayer is None
            ):
                raise ImportError(
                    "Dinov2-with-registers backbone requested, but transformers "
                    "does not expose Dinov2WithRegisters modeling classes."
                )
            embeddings_cls: Type[nn.Module] = cast(
                Type[nn.Module],
                Dinov2WithRegistersEmbeddings,
            )
            layer_cls: Type[nn.Module] = cast(Type[nn.Module], Dinov2WithRegistersLayer)
        else:
            embeddings_cls = cast(Type[nn.Module], Dinov2Embeddings)
            layer_cls = cast(Type[nn.Module], Dinov2Layer)

        self.embeddings = embeddings_cls(dino_v2_config)
        self.layer = nn.ModuleList(
            [layer_cls(dino_v2_config) for _ in range(dino_v2_config.num_hidden_layers)]
        )
        self.layernorm = nn.LayerNorm(
            dino_v2_config.hidden_size,
            eps=dino_v2_config.layer_norm_eps,
        )
        self.n_prefix_tokens = 1 + int(
            getattr(dino_v2_config, "num_register_tokens", 0)
        )
        self._use_antialiased_pos_resize()

    @staticmethod
    def _antialiased_interpolate_pos_encoding(
        embeddings_module: nn.Module,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        position_embeddings = cast(torch.Tensor, embeddings_module.position_embeddings)
        patch_size = int(getattr(embeddings_module.config, "patch_size"))
        dim = embeddings.shape[-1]
        target_grid = (height // patch_size, width // patch_size)

        num_patch_positions = int(
            getattr(
                getattr(embeddings_module, "patch_embeddings", None),
                "num_patches",
                position_embeddings.shape[1] - 1,
            )
        )
        num_prefix_positions = position_embeddings.shape[1] - num_patch_positions
        source_grid_size = int(math.sqrt(num_patch_positions))
        if source_grid_size * source_grid_size != num_patch_positions:
            raise ValueError(
                "DINOv2 position embeddings are expected to have a square spatial grid. "
                f"Got {num_patch_positions} spatial positions."
            )

        if target_grid == (source_grid_size, source_grid_size):
            return position_embeddings

        prefix_pos_embed = position_embeddings[:, :num_prefix_positions]
        patch_pos_embed = position_embeddings[:, num_prefix_positions:]
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = patch_pos_embed.reshape(
            1,
            source_grid_size,
            source_grid_size,
            dim,
        ).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.float(),
            size=target_grid,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((prefix_pos_embed, patch_pos_embed), dim=1)

    def _use_antialiased_pos_resize(self) -> None:
        interpolate_pos_encoding = getattr(
            self.embeddings,
            "interpolate_pos_encoding",
            None,
        )
        position_embeddings = getattr(self.embeddings, "position_embeddings", None)
        if not callable(interpolate_pos_encoding) or position_embeddings is None:
            return

        self.embeddings.interpolate_pos_encoding = types.MethodType(
            self._antialiased_interpolate_pos_encoding,
            self.embeddings,
        )

    def get_patch_size(self) -> int:
        patch_size = int(getattr(self.dino_v2_config, "patch_size", 0))
        if patch_size <= 0:
            raise ValueError(
                "DINOv2ViT config has invalid patch_size. "
                f"Got: {getattr(self.dino_v2_config, 'patch_size', None)}"
            )
        return patch_size

    def get_num_layers(self) -> int:
        num_hidden_layers = int(getattr(self.dino_v2_config, "num_hidden_layers", 0))
        if num_hidden_layers <= 0:
            raise ValueError(
                "DINOv2ViT config has invalid num_hidden_layers. "
                f"Got: {getattr(self.dino_v2_config, 'num_hidden_layers', None)}"
            )
        return num_hidden_layers

    @classmethod
    def _init_backbone_from_hf(cls, backbone_model_name: str) -> "DINOv2ViT":
        loaded_hf_model = cast(Any, AutoModel.from_pretrained(backbone_model_name))
        hf_model: Any = loaded_hf_model
        model_config = cast(Any, loaded_hf_model).config

        config_model_type = str(getattr(model_config, "model_type", "")).lower()
        is_dinov2_compatible = isinstance(model_config, Dinov2Config) or (
            "dinov2" in config_model_type
        )
        if not is_dinov2_compatible:
            raise TypeError(
                "DINOv2ViT expects a Dinov2-compatible checkpoint "
                "(including Dinov2-with-registers). "
                f"Got: {type(model_config)}"
            )

        model = cls(model_config)
        hf_backbone = cls._unwrap_hf_backbone(hf_model)

        cls._load_module(model.embeddings, hf_backbone.embeddings, strict=True)

        hf_layers = cast(List[nn.Module], hf_backbone.encoder.layer)
        if len(hf_layers) != len(model.layer):
            raise ValueError(
                "Layer count mismatch between HF model and DINOv2ViT: "
                f"{len(hf_layers)} != {len(model.layer)}"
            )

        for dst_layer, src_layer in zip(model.layer, hf_layers):
            cls._load_module(dst_layer.norm1, src_layer.norm1)
            cls._load_module(dst_layer.attention, src_layer.attention)
            cls._load_module(dst_layer.layer_scale1, src_layer.layer_scale1)
            cls._load_module(dst_layer.norm2, src_layer.norm2)
            cls._load_module(dst_layer.mlp, src_layer.mlp)
            cls._load_module(dst_layer.layer_scale2, src_layer.layer_scale2)
        cls._load_module(model.layernorm, hf_backbone.layernorm)
        return model

    @classmethod
    def init_from_hf(
        cls,
        backbone_model_name: str,
        backbone_lora_config: Optional[LoraConfig] = None,
        freeze_weights: bool = True,
    ):
        model: DINOv2ViT = cls._init_backbone_from_hf(backbone_model_name)
        if freeze_weights:
            model.requires_grad_(False)

        if backbone_lora_config is not None:
            if getattr(backbone_lora_config, "bias", None) in (None, "None"):
                backbone_lora_config.bias = "none"
            if getattr(backbone_lora_config, "init_lora_weights", None) != "loftq":
                backbone_lora_config.loftq_config = {}
            model = cast(
                DINOv2ViT,
                get_peft_model(cast(Any, model), backbone_lora_config),
            )
        return model

    def forward(
        self,
        pixel_values: torch.Tensor,
        window_size: int = 0,
        max_layer_index: int = -1,
    ) -> List[torch.Tensor]:
        del window_size

        h_embd = pixel_values.shape[2] // self.dino_v2_config.patch_size
        w_embd = pixel_values.shape[3] // self.dino_v2_config.patch_size

        def to_hwc(x: torch.Tensor) -> torch.Tensor:
            return x[:, self.n_prefix_tokens :, :].reshape(
                -1,
                h_embd,
                w_embd,
                x.shape[-1],
            )

        embeddings_module = cast(nn.Module, self.embeddings)
        patch_embeddings = cast(
            torch.Tensor,
            getattr(embeddings_module, "patch_embeddings")(pixel_values),
        )
        if patch_embeddings.dim() == 4:
            patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        batch_size = patch_embeddings.shape[0]
        prefix_tokens: List[torch.Tensor] = []
        cls_token = getattr(embeddings_module, "cls_token", None)
        if cls_token is not None:
            prefix_tokens.append(
                cast(torch.Tensor, cls_token).expand(batch_size, -1, -1)
            )
        register_tokens = getattr(embeddings_module, "register_tokens", None)
        if register_tokens is not None:
            prefix_tokens.append(
                cast(torch.Tensor, register_tokens).expand(batch_size, -1, -1)
            )

        if prefix_tokens:
            hidden_state_pre_pos = torch.cat(prefix_tokens + [patch_embeddings], dim=1)
        else:
            hidden_state_pre_pos = patch_embeddings

        # Indexing convention: 0 is embeddings output, i>0 is output after layer i.
        hidden_states_hwc: List[torch.Tensor] = [to_hwc(hidden_state_pre_pos)]

        interpolate_pos_encoding = getattr(
            embeddings_module, "interpolate_pos_encoding", None
        )
        if callable(interpolate_pos_encoding):
            position_embeddings = cast(
                torch.Tensor,
                interpolate_pos_encoding(
                    hidden_state_pre_pos,
                    pixel_values.shape[2],
                    pixel_values.shape[3],
                ),
            )
        else:
            position_embeddings = cast(
                torch.Tensor,
                getattr(embeddings_module, "position_embeddings"),
            )

        if position_embeddings.shape[1] != hidden_state_pre_pos.shape[1]:
            register_tokens = getattr(embeddings_module, "register_tokens", None)
            if register_tokens is not None:
                register_count = int(register_tokens.shape[1])
                if (
                    position_embeddings.shape[1] + register_count
                    == hidden_state_pre_pos.shape[1]
                ):
                    # Some HF checkpoints omit register positions; keep them zeroed.
                    cls_pos = position_embeddings[:, :1, :]
                    patch_pos = position_embeddings[:, 1:, :]
                    register_pos = position_embeddings.new_zeros(
                        position_embeddings.shape[0],
                        register_count,
                        position_embeddings.shape[2],
                    )
                    position_embeddings = torch.cat(
                        (cls_pos, register_pos, patch_pos),
                        dim=1,
                    )
        if position_embeddings.shape[1] != hidden_state_pre_pos.shape[1]:
            raise ValueError(
                "Position embedding length does not match token sequence. "
                f"Got {position_embeddings.shape[1]} vs {hidden_state_pre_pos.shape[1]}."
            )

        hidden_state_raw = hidden_state_pre_pos + position_embeddings
        dropout = getattr(embeddings_module, "dropout", None)
        if dropout is not None:
            hidden_state_raw = cast(torch.Tensor, dropout(hidden_state_raw))
        if max_layer_index == -1:
            max_layer_index = len(self.layer)
        for layer_idx in range(max_layer_index):
            hidden_state_raw = cast(
                torch.Tensor,
                self.layer[layer_idx](hidden_state_raw),
            )
            if layer_idx + 1 < len(self.layer):
                hidden_state_normed = cast(
                    Dinov2Layer,
                    self.layer[layer_idx + 1],
                ).norm1(hidden_state_raw)
            else:
                hidden_state_normed = self.layernorm(hidden_state_raw)
            hidden_states_hwc.append(to_hwc(hidden_state_normed))

        return hidden_states_hwc
