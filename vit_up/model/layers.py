"""Building blocks of the ViT-Up decoder."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Linear layers with GELU between them, optional input LayerNorm and residual."""

    def __init__(self, dims: list[int], input_layernorm: bool = False, residual: bool = False,
                 zero_init_last: bool = False):
        super().__init__()
        if residual and dims[0] != dims[-1]:
            raise ValueError(f"Residual MLPs need equal input and output dims, got {dims}.")
        layers: list[nn.Module] = [nn.LayerNorm(dims[0])] if input_layernorm else []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        if zero_init_last:
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x) if self.residual else self.mlp(x)


class FiLM(nn.Module):
    """Modulate a backbone token with its query's relative position, then refine it."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.gamma_beta_mlp = MLP([cond_dim, dim, 2 * dim], zero_init_last=True)
        self.post_mlp = MLP([dim, 4 * dim, dim], input_layernorm=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = torch.chunk(self.gamma_beta_mlp(cond), 2, dim=-1)
        return self.post_mlp((1 + gamma) * self.norm_input(x) + beta)


class FourierPositionalEncoding(nn.Module):
    """Concatenate coordinates with their sines and cosines at log-spaced frequencies."""

    def __init__(self, num_bands: int = 16, max_resolution: float = 10.0):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(max_resolution), num_bands)))

    @property
    def out_dim(self) -> int:
        return 2 + 4 * len(self.freqs)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        angles = xy.unsqueeze(-1) * self.freqs * math.pi
        return torch.cat([xy, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """[x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


class ContinuousRoPE2D(nn.Module):
    """2D rotary embeddings at continuous positions, measured in backbone-token units.

    Half of the channels encode x and half encode y. Queries use their exact
    positions, keys use the centers of the backbone tokens.
    """

    def __init__(self, dim: int, base: float = 100.0, scale: float = 2 * math.pi):
        super().__init__()
        if dim % 4:
            raise ValueError(f"RoPE dim must be divisible by 4, got {dim}.")
        n_pairs = dim // 4
        self.scale = scale
        inv_freq = 1.0 / (base ** (torch.arange(n_pairs, dtype=torch.float32) / n_pairs))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = self.scale * xy.to(self.inv_freq.dtype)[..., None] * self.inv_freq
        angles = angles.repeat_interleave(2, dim=-1).flatten(-2)
        return angles.cos(), angles.sin()

    def forward(self, q_xy: torch.Tensor, h_tokens: int, w_tokens: int):
        """Return (q_cos, q_sin, k_cos, k_sin) for normalized query coordinates."""
        q_cos, q_sin = self._cos_sin(q_xy * q_xy.new_tensor([w_tokens, h_tokens]))
        ys = torch.arange(h_tokens, device=q_xy.device, dtype=q_xy.dtype) + 0.5
        xs = torch.arange(w_tokens, device=q_xy.device, dtype=q_xy.dtype) + 0.5
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(1, -1, 2)
        k_cos, k_sin = self._cos_sin(grid)
        return q_cos, q_sin, k_cos, k_sin


class CrossAttention(nn.Module):
    """Global attention from queries to all backbone tokens, with continuous 2D RoPE."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor, rope) -> torch.Tensor:
        q, k, v = self._heads(self.q_proj(queries)), self._heads(self.k_proj(tokens)), self._heads(self.v_proj(tokens))
        q_cos, q_sin, k_cos, k_sin = (x[:, None] for x in rope)
        q = q * q_cos.to(q.dtype) + _rotate_pairs(q) * q_sin.to(q.dtype)
        k = k * k_cos.to(k.dtype) + _rotate_pairs(k) * k_sin.to(k.dtype)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).flatten(2))
