import math
import torch
import torch.nn as nn


class SinusoidalPosEmb1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb1D expects even dim, got {dim}.")
        self.dim = dim

    def forward(self, x):
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        device = x.device
        half_dim = self.dim // 2
        denom = max(half_dim - 1, 1)
        emb = math.log(10000) / denom
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * -emb)
        emb = x[..., None] * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SinusoidalPosEmb2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(
                f"SinusoidalPosEmb2D expects dim divisible by 4, got {dim}."
            )
        self.dim = dim
        self._axis_emb = SinusoidalPosEmb1D(dim // 2)

    def forward(self, xy):
        if xy.shape[-1] != 2:
            raise ValueError(
                f"SinusoidalPosEmb2D expects last dim == 2, got {xy.shape[-1]}."
            )
        x_emb = self._axis_emb(xy[..., 0])
        y_emb = self._axis_emb(xy[..., 1])
        return torch.cat((x_emb, y_emb), dim=-1)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_bands=16, max_resolution=10.0):
        """Generates sine/cosine positional embeddings from spatial coordinates."""
        super().__init__()
        self.num_bands = num_bands

        # Create log-linear spaced frequencies
        freqs = torch.exp(torch.linspace(0, math.log(max_resolution), steps=num_bands))
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        # x shape: (..., 2)
        # Multiply x by frequencies -> (..., 2, num_bands)
        x_expanded = x.unsqueeze(-1) * self.freqs

        # Calculate sine and cosine
        sin_feats = torch.sin(x_expanded * math.pi)
        cos_feats = torch.cos(x_expanded * math.pi)

        # Output shape: (..., 2 + 4 * num_bands)
        out = torch.cat(
            [x, sin_feats.flatten(-2, -1), cos_feats.flatten(-2, -1)], dim=-1
        )

        return out
