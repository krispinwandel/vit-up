"""Encoder/decoder features for the video segmentation benchmarks (LILA protocol).

LILA's probes read two feature maps per image: low-resolution tokens of the
frozen backbone ("encoder") and dense features at the input resolution
("decoder"). Here the encoder is the frozen Hugging Face backbone, identical for
every method, and the decoder is any eval-kit upsampler (ViT-Up, LoftUp, NAF,
LILA, ...). Without a decoder only the encoder branch is used, which gives
the plain-backbone baseline rows of the LILA tables.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from vit_up.eval_kits.upsamplers.base import UpsamplerBase
from vit_up.eval_kits.upsamplers.vit_up import ViTUpUpsampler

from .datasets import IMAGENET_MEAN, IMAGENET_STD


class BackboneEncoder(nn.Module):
    """Final (layer-normalized) patch tokens of a frozen ViT at the input resolution, (B, C, h, w)."""

    def __init__(self, backbone_model_name: str = "facebook/dinov2-small", revision: str | None = None):
        super().__init__()
        self.model = AutoModel.from_pretrained(backbone_model_name, revision=revision).eval().requires_grad_(False)
        config = self.model.config
        self.patch_size = int(config.patch_size)
        self.n_prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0))
        self.feature_dim = int(config.hidden_size)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        p = self.patch_size
        h, w = max(1, round(height / p)), max(1, round(width / p))
        if (h * p, w * p) != (height, width):  # keep the input patch-aligned, as LILA's PatchResample does
            images = F.interpolate(images, (h * p, w * p), mode="bilinear", align_corners=False)
        tokens = self.model(pixel_values=images).last_hidden_state[:, self.n_prefix_tokens:]
        return tokens.transpose(1, 2).unflatten(-1, (h, w)).float()


class FeatureExtractor(nn.Module):
    """Returns (encoder tokens, decoder features or None) for ImageNet-normalized images."""

    def __init__(self, encoder: BackboneEncoder, decoder: UpsamplerBase | None = None,
                 decoder_norm: str = "layernorm", decoder_batch_size: int | None = None):
        super().__init__()
        if decoder_norm not in ("layernorm", "none"):
            raise ValueError(f"decoder_norm must be 'layernorm' or 'none', got {decoder_norm!r}.")
        self.encoder = encoder
        self.decoder = decoder.eval() if decoder is not None else None
        self.decoder_norm = decoder_norm
        self.decoder_batch_size = decoder_batch_size
        self.register_buffer("pad_value", (-torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD)).view(1, 3, 1, 1),
                             persistent=False)

    @property
    def encoder_dim(self) -> int:
        return self.encoder.feature_dim

    def decoder_dim(self, device) -> int | None:
        if self.decoder is None:
            return None
        return self(torch.zeros(1, 3, 224, 224, device=device), dense_size=(8, 8))[1].shape[1]

    @property
    def is_implicit(self) -> bool:
        """Implicit upsamplers (ViT-Up) are evaluated directly at the requested coordinates."""
        return isinstance(self.decoder, ViTUpUpsampler)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, dense_size: tuple[int, int] | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encoder tokens and dense features of size `dense_size` (default: the input resolution).

        Each model produces the requested resolution its own way: implicit upsamplers are queried
        only at the cell centers of the requested grid over the image; all others compute their
        native output (e.g. LILA at the input resolution, backbone probes at token resolution)
        and are bilinearly resized to it, as in LILA's evaluation.
        """
        encoder = self.encoder(images)
        if self.decoder is None:
            return encoder, None
        size = tuple(dense_size) if dense_size is not None else tuple(images.shape[-2:])
        decode = self._query_grid if self.is_implicit else self._decode_resized
        chunks = images.split(self.decoder_batch_size or len(images))
        first = decode(chunks[0], size)
        if len(chunks) == 1:
            return encoder, first
        # Fill one preallocated tensor instead of concatenating the parts: wide dense features are
        # several GB per batch, and torch.cat would hold every part and the result at once.
        features = first.new_empty((len(images), *first.shape[1:]))
        features[:len(first)] = first
        start = len(first)
        del first
        for part in chunks[1:]:
            features[start:start + len(part)] = decode(part, size)
            start += len(part)
        return encoder, features

    def _decode_resized(self, images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        features = self._decode(images)
        if tuple(features.shape[-2:]) != size:
            features = F.interpolate(features, size, mode="bilinear", align_corners=False)
        return features.contiguous()

    def _query_grid(self, images: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        height, width = images.shape[-2:]
        size = max(height, width)
        top, left = (size - height) // 2, (size - width) // 2
        if height != width:  # center non-square frames on a black square, as during training
            padded = self.pad_value.expand(len(images), 3, size, size).clone()
            padded[:, :, top:top + height, left:left + width] = images
            images = padded
        # Cell centers of the grid over the visible frame, as normalized (x, y) in the square; these
        # are the positions bilinear resizing (align_corners=False) samples.
        rows, cols = grid
        ys = (top + (torch.arange(rows, device=images.device) + 0.5) * height / rows) / size
        xs = (left + (torch.arange(cols, device=images.device) + 0.5) * width / cols) / size
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        queries = torch.stack((gx, gy), dim=-1).reshape(1, -1, 2).expand(len(images), -1, -1).float()
        # The wrapper's chunks split queries per image across the whole batch; divide by the batch
        # size so one chunk holds at most query_chunk_size queries in total (wide features at high
        # resolution otherwise exhaust GPU memory). Chunking does not change the features.
        chunk = max(1024, (self.decoder.query_chunk_size or queries.shape[1]) // len(images))
        # ViTUpWrapper: encodes the images at its backbone resolution under bf16 autocast.
        features = self.decoder.model(images, queries, query_chunk_size=chunk).float()
        if self.decoder_norm == "layernorm":
            # Parameter-free layer norm over channels, in place: wide dense features (e.g. 768
            # channels at 504x504) would otherwise exist in several full-size copies at once.
            mean = features.mean(dim=-1, keepdim=True)
            var = features.var(dim=-1, unbiased=False, keepdim=True)
            features.sub_(mean).div_(var.add_(1e-5).sqrt_())
        # (B, C, H, W) view of channels-last memory; convolutions and interpolation accept it as is.
        return features.reshape(len(images), rows, cols, -1).permute(0, 3, 1, 2)

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        """Native output of a non-implicit model for the frame (input resolution or token grid)."""
        height, width = images.shape[-2:]
        size = max(height, width)
        if getattr(self.decoder, "native_aspect", False):
            features = self.decoder(images, (height, width)).permute(0, 3, 1, 2)
            return self._normalize(features.float()).contiguous()
        if height != width:
            # Upsamplers take square images: center the frame on a black square and crop the output.
            top, left = (size - height) // 2, (size - width) // 2
            padded = self.pad_value.expand(len(images), 3, size, size).clone()
            padded[:, :, top:top + height, left:left + width] = images
            images = padded
        features = self.decoder(images, size).permute(0, 3, 1, 2).float()
        if height != width:
            scale = features.shape[-1] / size
            top, left = round(top * scale), round(left * scale)
            features = features[..., top:top + round(height * scale), left:left + round(width * scale)]
        return self._normalize(features).contiguous()

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.decoder_norm == "layernorm" and not getattr(self.decoder, "normalized_output", False):
            # LILA layer-normalizes its decoder features; use the parameter-free equivalent, in
            # place over channels (an out-of-place layer norm would double the peak memory).
            mean = features.mean(dim=1, keepdim=True)
            var = features.var(dim=1, unbiased=False, keepdim=True)
            features.sub_(mean).div_(var.add_(1e-5).sqrt_())
        return features
