"""ViT-Up: an implicit decoder that queries backbone-aligned features at continuous coordinates."""
import torch
import torch.nn.functional as F
from torch import nn

from .backbones import Backbone
from .layers import MLP, ContinuousRoPE2D, CrossAttention, FiLM, FourierPositionalEncoding


def closest_token_index(queries: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Flat index of the backbone token whose cell contains each normalized (x, y) query."""
    col = torch.round(queries[..., 0] * width - 0.5).long().clamp(0, width - 1)
    row = torch.round(queries[..., 1] * height - 0.5).long().clamp(0, height - 1)
    return row * width + col


def relative_position(queries: torch.Tensor, index: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Offset of each query from its closest token center, in token units."""
    center_x = (torch.remainder(index, width).float() + 0.5) / width
    center_y = (torch.div(index, width, rounding_mode="floor").float() + 0.5) / height
    return torch.stack(((queries[..., 0] - center_x) / (1.0 / width),
                        (queries[..., 1] - center_y) / (1.0 / height)), dim=-1)


class ViTUpBlock(nn.Module):
    """Refine query features with one backbone layer: global attention plus local token modulation."""

    def __init__(self, dim: int, num_heads: int, rel_pos_dim: int):
        super().__init__()
        self.transition_mlp = MLP([dim, 2 * dim, dim], input_layernorm=True, residual=True)
        self.cross_attention = CrossAttention(dim, num_heads)
        self.featx = FiLM(dim, rel_pos_dim)
        self.norm_q_global = nn.LayerNorm(dim)
        self.norm_h_local = nn.LayerNorm(dim)
        self.norm_h_global = nn.LayerNorm(dim)
        self.norm_post = nn.LayerNorm(dim)
        self.fusion_mlp = MLP([dim, 4 * dim, dim])

    def forward(self, q, tokens_hwc, closest_token, rel_pos_enc, rope):
        q = self.transition_mlp(q)
        q = q + self.cross_attention(self.norm_q_global(q), self.norm_h_global(tokens_hwc.flatten(1, 2)), rope)
        q = q + self.norm_h_local(self.featx(closest_token, rel_pos_enc))
        return q + self.fusion_mlp(self.norm_post(q))


class ViTUp(nn.Module):
    """Predict the features of `layer_indices` backbone layers at arbitrary query coordinates.

    Queries start from patch embeddings bilinearly sampled from a high-resolution
    image (`query_image_size`). Each block then moves them to the next selected
    backbone layer's feature space, and a per-layer decoder maps them to its output.
    """

    def __init__(self, dim: int, num_heads: int, layer_indices: list[int], query_image_size: int = 3584,
                 num_bands: int = 16, max_resolution: float = 10.0, rope_base: float = 100.0):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.query_image_size = query_image_size
        self.rel_pos_enc = FourierPositionalEncoding(num_bands, max_resolution)
        self.q_rope_embeddings = ContinuousRoPE2D(dim // num_heads, base=rope_base)
        self.vit_up_blocks = nn.ModuleList(
            ViTUpBlock(dim, num_heads, self.rel_pos_enc.out_dim) for _ in self.layer_indices[1:])
        self.decoder_mlp = nn.ModuleList(MLP([dim, dim], input_layernorm=True) for _ in self.layer_indices)

    def encode(self, pixel_values: torch.Tensor, backbone: Backbone, image_size: int | None = None) -> dict:
        """Compute the per-image cache shared by all queries of that image."""
        hidden_states = backbone.hidden_states(pixel_values, image_size)
        query_image = F.interpolate(pixel_values, size=(self.query_image_size, self.query_image_size),
                                    mode="bilinear", align_corners=False)
        return {
            "hidden_states": [hidden_states[i] for i in self.layer_indices],
            "patch_embeddings": backbone.embed_patches(query_image).permute(0, 3, 1, 2).contiguous(),
        }

    def forward(self, queries: torch.Tensor, cache: dict) -> list[torch.Tensor]:
        """Return features of every selected layer, each (B, N, C), for (B, N, 2) queries in [0, 1]."""
        hidden_states = cache["hidden_states"]
        patch_embeddings = cache["patch_embeddings"]
        grid = (queries * 2.0 - 1.0).unsqueeze(2).to(patch_embeddings.dtype)
        q = F.grid_sample(patch_embeddings, grid, mode="bilinear", padding_mode="border", align_corners=False)
        q = q.squeeze(-1).transpose(1, 2)

        height, width = hidden_states[0].shape[1:3]
        index = closest_token_index(queries, height, width)
        rel_pos_enc = self.rel_pos_enc(relative_position(queries, index, height, width))
        rope = self.q_rope_embeddings(queries, height, width)
        batch = torch.arange(len(queries), device=queries.device)[:, None]

        outputs = [self.decoder_mlp[0](q)]
        for block, decoder, tokens in zip(self.vit_up_blocks, self.decoder_mlp[1:], hidden_states[1:]):
            closest_token = tokens.flatten(1, 2)[batch, index]
            q = block(q, tokens, closest_token, rel_pos_enc, rope)
            outputs.append(decoder(q))
        return outputs
