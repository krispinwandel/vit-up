import math
import torch
import torch.nn as nn
from typing import Tuple
from vit_up.utils import grid_coords


def build_q_position_embeddings(
    h_tokens: int,
    w_tokens: int,
    q_xy_normalized: torch.Tensor,
    query_rope_embeddings: nn.Module,
) -> torch.Tensor:
    grid_center_xy_normalized = grid_coords.build_token_center_grid(
        h_tokens=h_tokens,
        w_tokens=w_tokens,
        device=q_xy_normalized.device,
        dtype=q_xy_normalized.dtype,
    )
    q_xy_for_rope = q_xy_normalized * q_xy_normalized.new_tensor(
        [
            float(grid_center_xy_normalized.shape[1]),
            float(grid_center_xy_normalized.shape[0]),
        ]
    )
    q_position_embeddings = query_rope_embeddings(
        q_xy_normalized=q_xy_for_rope,
        grid_center_xy_normalized=grid_center_xy_normalized,
    )
    return q_position_embeddings


class ContinuousRoPE2D(torch.nn.Module):
    """
    Continuous 2D RoPE phase generator.

    Usage:
        rope = ContinuousRoPE2D(dim=head_dim)

        q_cos, q_sin, g_cos, g_sin = rope(q_xy_normalized, grid_center_xy_normalized)

    where:
        q_xy_normalized:        (B, Nq, 2)
        grid_center_xy_normalized: (H, W, 2)

    and outputs:
        q_cos, q_sin:           (B, Nq, dim)
        g_cos, g_sin:           (1, H*W, dim)

    You can then apply RoPE to q and k with `apply_rope(...)`.

    Notes:
    - Coordinates are assumed normalized, usually in [0, 1].
    - Half of the channels are assigned to x, half to y.
    - Each 2D pair within each axis gets its own frequency.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale: float = 2.0 * math.pi,
    ) -> None:
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4 for 2D RoPE. Got dim={dim}.")

        self.dim = dim
        self.base = float(base)
        self.scale = float(scale)

        # dim_x = dim_y = dim / 2
        # each axis must itself be divisible by 2 because RoPE rotates pairs
        axis_dim = dim // 2
        n_pairs_per_axis = axis_dim // 2

        freq_idx = torch.arange(n_pairs_per_axis, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (freq_idx / n_pairs_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis_angles(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (...,)
        returns angles: (..., n_pairs_per_axis)
        """
        return self.scale * coords[..., None] * self.inv_freq

    @staticmethod
    def _interleave_pairs(
        cos_part: torch.Tensor, sin_part: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cos_part/sin_part: (..., n_pairs)
        returns cos/sin expanded to (..., 2*n_pairs), where each value is repeated
        for the two channels of the rotation pair.
        """
        cos_full = torch.repeat_interleave(cos_part, repeats=2, dim=-1)
        sin_full = torch.repeat_interleave(sin_part, repeats=2, dim=-1)
        return cos_full, sin_full

    def _build_cos_sin(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xy: (..., 2)
        returns:
            cos: (..., dim)
            sin: (..., dim)
        """
        x = xy[..., 0]
        y = xy[..., 1]

        ang_x = self._axis_angles(x)  # (..., n_pairs_per_axis)
        ang_y = self._axis_angles(y)  # (..., n_pairs_per_axis)

        cos_x, sin_x = self._interleave_pairs(torch.cos(ang_x), torch.sin(ang_x))
        cos_y, sin_y = self._interleave_pairs(torch.cos(ang_y), torch.sin(ang_y))

        cos = torch.cat([cos_x, cos_y], dim=-1)
        sin = torch.cat([sin_x, sin_y], dim=-1)
        return cos, sin

    def forward(
        self,
        q_xy_normalized: torch.Tensor,  # (B, Nq, 2)
        grid_center_xy_normalized: torch.Tensor,  # (H, W, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if q_xy_normalized.ndim != 3 or q_xy_normalized.shape[-1] != 2:
            raise ValueError(
                "q_xy_normalized must have shape (B, Nq, 2). "
                f"Got {tuple(q_xy_normalized.shape)}."
            )

        if (
            grid_center_xy_normalized.ndim != 3
            or grid_center_xy_normalized.shape[-1] != 2
        ):
            raise ValueError(
                "grid_center_xy_normalized must have shape (H, W, 2). "
                f"Got {tuple(grid_center_xy_normalized.shape)}."
            )

        q_xy = q_xy_normalized.to(dtype=self.inv_freq.dtype)
        g_xy = grid_center_xy_normalized.to(dtype=self.inv_freq.dtype)

        q_cos, q_sin = self._build_cos_sin(q_xy)  # (B, Nq, dim)
        g_cos, g_sin = self._build_cos_sin(g_xy)  # (H, W, dim)

        H, W, _ = g_cos.shape
        g_cos = g_cos.reshape(1, H * W, self.dim)
        g_sin = g_sin.reshape(1, H * W, self.dim)

        return q_cos, q_sin, g_cos, g_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the RoPE pairwise quarter-turn:
        [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]

    Works on the last dimension.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (..., dim)
    cos: broadcastable to x
    sin: broadcastable to x
    """
    return x * cos + rotate_half(x) * sin
