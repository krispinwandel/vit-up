"""Frozen DINOv2/DINOv3 backbones exposing the hidden state of every layer.

Hidden states are (B, H, W, C) patch-token maps. Index 0 is the patch embedding,
index i > 0 is the output of block i normalized by the next block's first norm
(or the final norm), i.e. the features the next block actually attends over.
"""
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from transformers import AutoModel


class Backbone(nn.Module):
    patch_size: int
    n_prefix_tokens: int

    def to_hwc(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens[:, self.n_prefix_tokens:].unflatten(1, (height // self.patch_size, width // self.patch_size))

    def embed_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Patch embeddings before position encoding, as a (B, H, W, C) map."""
        raise NotImplementedError

    def hidden_states(self, pixel_values: torch.Tensor, image_size: int | None = None,
                      return_class_token: bool = False):
        """Resize to `image_size` and run the backbone, autocasting for low-precision inputs."""
        if image_size is not None and tuple(pixel_values.shape[-2:]) != (image_size, image_size):
            pixel_values = F.interpolate(pixel_values, size=(image_size, image_size), mode="bilinear",
                                         align_corners=False)
        low_precision = pixel_values.device.type == "cuda" and pixel_values.dtype in (torch.float16, torch.bfloat16)
        with torch.autocast("cuda", dtype=pixel_values.dtype) if low_precision else nullcontext():
            return self(pixel_values, return_class_token=return_class_token)


class DINOv3Backbone(Backbone):
    def __init__(self, hf_model):
        super().__init__()
        self.config = hf_model.config
        self.patch_size = self.config.patch_size
        self.n_prefix_tokens = 1 + self.config.num_register_tokens
        self.embeddings = hf_model.embeddings
        self.rope_embeddings = hf_model.rope_embeddings
        self.layer = hf_model.model.layer
        self.norm = hf_model.norm
        for layer in self.layer:
            layer.drop_path = nn.Identity()

    def forward(self, pixel_values: torch.Tensor, return_class_token: bool = False):
        height, width = pixel_values.shape[-2:]
        rope = self.rope_embeddings(pixel_values)
        tokens = self.embeddings(pixel_values)
        hidden_states = [self.to_hwc(tokens, height, width)]
        norms = [layer.norm1 for layer in self.layer[1:]] + [self.norm]
        for layer, norm in zip(self.layer, norms):
            tokens = layer(tokens, position_embeddings=rope)
            normed = norm(tokens)
            hidden_states.append(self.to_hwc(normed, height, width))
        return (hidden_states, normed[:, 0]) if return_class_token else hidden_states

    def embed_patches(self, pixel_values):
        projection = self.embeddings.patch_embeddings
        features = projection(pixel_values.to(projection.weight.dtype))
        dtype = torch.promote_types(features.dtype, self.embeddings.cls_token.dtype)
        dtype = torch.promote_types(dtype, self.embeddings.register_tokens.dtype)
        return features.to(dtype).permute(0, 2, 3, 1)


class DINOv2Backbone(Backbone):
    def __init__(self, hf_model):
        super().__init__()
        self.config = hf_model.config
        self.patch_size = self.config.patch_size
        self.n_prefix_tokens = 1
        self.embeddings = hf_model.embeddings
        self.layer = hf_model.encoder.layer
        self.layernorm = hf_model.layernorm

    def _tokens(self, pixel_values):
        patches = self.embeddings.patch_embeddings(pixel_values)
        return torch.cat([self.embeddings.cls_token.expand(len(patches), -1, -1), patches], dim=1)

    def _position_embeddings(self, height, width):
        """Resize the learned position grid with antialiased bicubic interpolation."""
        position = self.embeddings.position_embeddings
        source = math.isqrt(position.shape[1] - 1)
        target = (height // self.patch_size, width // self.patch_size)
        if target == (source, source):
            return position
        grid = position[:, 1:].reshape(1, source, source, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=target, mode="bicubic", align_corners=False, antialias=True)
        grid = grid.to(position.dtype).permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat([position[:, :1], grid], dim=1)

    def forward(self, pixel_values: torch.Tensor, return_class_token: bool = False):
        height, width = pixel_values.shape[-2:]
        tokens = self._tokens(pixel_values)
        hidden_states = [self.to_hwc(tokens, height, width)]
        tokens = self.embeddings.dropout(tokens + self._position_embeddings(height, width))
        norms = [layer.norm1 for layer in self.layer[1:]] + [self.layernorm]
        for layer, norm in zip(self.layer, norms):
            tokens = layer(tokens)
            normed = norm(tokens)
            hidden_states.append(self.to_hwc(normed, height, width))
        return (hidden_states, normed[:, 0]) if return_class_token else hidden_states

    def embed_patches(self, pixel_values):
        return self.to_hwc(self._tokens(pixel_values), *pixel_values.shape[-2:])


def load_backbone(name: str, lora_config: dict | None = None, revision: str | None = None) -> nn.Module:
    """Load a frozen pretrained backbone, optionally wrapped with (trainable) LoRA adapters."""
    hf_model = AutoModel.from_pretrained(name, revision=revision)
    model_type = hf_model.config.model_type
    if model_type == "dinov3_vit":
        backbone = DINOv3Backbone(hf_model)
    elif model_type == "dinov2":
        backbone = DINOv2Backbone(hf_model)
    else:
        raise ValueError(f"Unsupported backbone {name!r} of type {model_type!r}.")
    backbone.requires_grad_(False)
    if lora_config is not None:
        backbone = get_peft_model(backbone, LoraConfig(**lora_config))
    return backbone


def lora_state_dict(backbone: nn.Module) -> dict[str, torch.Tensor]:
    return get_peft_model_state_dict(backbone) if isinstance(backbone, PeftModel) else {}


def load_lora_state_dict(backbone: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if lora_state_dict(backbone).keys() != state.keys():
        raise ValueError("Checkpoint LoRA adapters do not match the backbone's LoRA configuration.")
    if state:
        set_peft_model_state_dict(backbone, state)


def disable_adapters(backbone: nn.Module):
    """Context in which a LoRA backbone runs with its original pretrained weights."""
    return backbone.disable_adapter() if isinstance(backbone, PeftModel) else nullcontext()
