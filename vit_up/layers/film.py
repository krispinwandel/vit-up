import torch
import torch.nn as nn
from typing import Optional


class SimpleFiLM(nn.Module):
    def __init__(
        self,
        dim: int,
        gamma_beta_mlp: nn.Module,
        post_mlp: Optional[nn.Module] = None,
        use_residual: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.use_residual = use_residual
        self.gamma_beta_mlp = gamma_beta_mlp
        self.post_mlp = post_mlp

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = torch.chunk(self.gamma_beta_mlp(x_cond), 2, dim=-1)
        gamma = 1 + gamma
        h = gamma * h + beta
        if self.use_residual:
            h = h + x  # FiLM with residual connection
        if self.post_mlp is not None:
            x = h  # for post_mlp residual
            h = self.post_mlp(h)
            if self.use_residual:
                h = h + x
        return h


class SimpleFiLMV2(nn.Module):
    def __init__(
        self,
        gamma_beta_mlp: nn.Module,
        post_mlp: Optional[nn.Module] = None,
        input_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.norm_input = input_module if input_module is not None else nn.Identity()
        self.gamma_beta_mlp = gamma_beta_mlp
        self.post_mlp = post_mlp

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        x = self.norm_input(x)
        gamma, beta = torch.chunk(self.gamma_beta_mlp(x_cond), 2, dim=-1)
        gamma = 1 + gamma
        x = gamma * x + beta
        if self.post_mlp is not None:
            x = self.post_mlp(x)
        return x
