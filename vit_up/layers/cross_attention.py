import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from natten import na2d as natten_na2d
except ImportError:
    natten_na2d = None


def _rotate_half_pairs(x: torch.Tensor) -> torch.Tensor:
    """Pairwise quarter-turn: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(start_dim=-2)


def _apply_continuous_rotary_pos_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    g_cos: torch.Tensor,
    g_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_cos.ndim != 3 or q_sin.ndim != 3 or g_cos.ndim != 3 or g_sin.ndim != 3:
        raise ValueError(
            "Continuous RoPE tensors must have shape (B_or_1, N, Dh). "
            f"Got q_cos={tuple(q_cos.shape)}, q_sin={tuple(q_sin.shape)}, "
            f"g_cos={tuple(g_cos.shape)}, g_sin={tuple(g_sin.shape)}."
        )

    q_cos = q_cos[:, None, :, :].to(dtype=q.dtype)
    q_sin = q_sin[:, None, :, :].to(dtype=q.dtype)
    g_cos = g_cos[:, None, :, :].to(dtype=k.dtype)
    g_sin = g_sin[:, None, :, :].to(dtype=k.dtype)

    q = (q * q_cos) + (_rotate_half_pairs(q) * q_sin)
    k = (k * g_cos) + (_rotate_half_pairs(k) * g_sin)
    return q, k


class CrossAttention(nn.Module):
    """Cross-attention where q is from query_feature and k/v are from hidden_states."""

    def __init__(
        self,
        dim: int,
        dim_kv: Optional[int] = None,
        num_heads: int = 8,
        cross_attn_window_size: int = 0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        q_proj: Optional[nn.Module] = None,
        k_proj: Optional[nn.Module] = None,
        v_proj: Optional[nn.Module] = None,
        out_proj: Optional[nn.Module] = None,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.dim = int(dim)
        self.dim_kv = int(dim_kv) if dim_kv is not None else self.dim
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_kv // self.num_heads
        self.scale = self.head_dim**-0.5
        self.cross_attn_window_size = int(cross_attn_window_size)

        self.q_proj = (
            nn.Linear(self.dim, self.dim_kv, bias=qkv_bias)
            if q_proj is None
            else q_proj
        )
        self.k_proj = (
            nn.Linear(self.dim_kv, self.dim_kv, bias=qkv_bias)
            if k_proj is None
            else k_proj
        )
        self.v_proj = (
            nn.Linear(self.dim_kv, self.dim_kv, bias=qkv_bias)
            if v_proj is None
            else v_proj
        )
        self.out_proj = (
            nn.Linear(self.dim_kv, self.dim, bias=True)
            if out_proj is None
            else out_proj
        )

        self.attn_dropout = float(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        return x.view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, n_tokens, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, n_tokens, self.dim_kv)

    def _global_cross_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        return self._from_heads(out)

    def _window_cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
    ) -> Optional[torch.Tensor]:
        bsz, _, q_tokens, _ = q.shape
        _, _, k_tokens, _ = k.shape
        _, _, v_tokens, _ = v.shape

        # Local neighborhood attention requires same query/key/value spatial lattice.
        if not (q_tokens == k_tokens == v_tokens):
            return None

        side = int(math.isqrt(q_tokens))
        if side * side != q_tokens:
            return None

        if natten_na2d is None:
            raise ImportError(
                "NATTEN is not installed. Install it to use windowed cross attention."
            )

        kernel_size = 2 * (window_size // 2) + 1
        kernel_size = min(kernel_size, side)

        q_spatial = q.view(bsz, self.num_heads, side, side, self.head_dim)
        k_spatial = k.view(bsz, self.num_heads, side, side, self.head_dim)
        v_spatial = v.view(bsz, self.num_heads, side, side, self.head_dim)

        q_na = q_spatial.permute(0, 2, 3, 1, 4).contiguous()
        k_na = k_spatial.permute(0, 2, 3, 1, 4).contiguous()
        v_na = v_spatial.permute(0, 2, 3, 1, 4).contiguous()

        out_spatial = natten_na2d(
            q_na,
            k_na,
            v_na,
            kernel_size=kernel_size,
            scale=self.scale,
        )
        out_spatial = (
            out_spatial.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(bsz, self.num_heads, q_tokens, self.head_dim)
        )
        return self._from_heads(out_spatial)

    def forward(
        self,
        query_feature: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        window_size: Optional[int] = None,
    ) -> torch.Tensor:

        q = self._to_heads(self.q_proj(query_feature))
        k = self._to_heads(self.k_proj(hidden_states))
        v = self._to_heads(self.v_proj(hidden_states))

        if position_embeddings is not None:
            q_cos, q_sin, g_cos, g_sin = position_embeddings

            if (
                q_cos.shape[:2] != query_feature.shape[:2]
                or q_sin.shape[:2] != query_feature.shape[:2]
            ):
                raise ValueError(
                    "q_cos/q_sin must match query_feature (B, q). "
                    f"Got q_cos={tuple(q_cos.shape)}, q_sin={tuple(q_sin.shape)}, "
                    f"query_feature={tuple(query_feature.shape)}."
                )

            if (
                g_cos.shape[1] != hidden_states.shape[1]
                or g_sin.shape[1] != hidden_states.shape[1]
            ):
                raise ValueError(
                    "g_cos/g_sin token dim must match hidden_states token dim t. "
                    f"Got g_cos={tuple(g_cos.shape)}, g_sin={tuple(g_sin.shape)}, "
                    f"hidden_states={tuple(hidden_states.shape)}."
                )
            if g_cos.shape[0] not in (1, hidden_states.shape[0]) or g_sin.shape[
                0
            ] not in (1, hidden_states.shape[0]):
                raise ValueError(
                    "g_cos/g_sin batch dim must be 1 or B. "
                    f"Got g_cos B={g_cos.shape[0]}, g_sin B={g_sin.shape[0]}, "
                    f"expected 1 or {hidden_states.shape[0]}."
                )

            if (
                q_cos.shape[-1] != self.head_dim
                or q_sin.shape[-1] != self.head_dim
                or g_cos.shape[-1] != self.head_dim
                or g_sin.shape[-1] != self.head_dim
            ):
                raise ValueError(
                    "All RoPE head dims must equal attention head_dim. "
                    f"Got q_cos={q_cos.shape[-1]}, q_sin={q_sin.shape[-1]}, "
                    f"g_cos={g_cos.shape[-1]}, g_sin={g_sin.shape[-1]}, "
                    f"expected {self.head_dim}."
                )

            q, k = _apply_continuous_rotary_pos_emb_qk(q, k, q_cos, q_sin, g_cos, g_sin)

        ws = self.cross_attn_window_size if window_size is None else int(window_size)
        if ws > 0:
            attn_out = self._window_cross_attention(q, k, v, window_size=ws)
            if attn_out is None:
                attn_out = self._global_cross_attention(q, k, v)
        else:
            attn_out = self._global_cross_attention(q, k, v)

        return self.proj_dropout(self.out_proj(attn_out))
