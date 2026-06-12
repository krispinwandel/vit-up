from typing import List, Optional, Union
import torch.nn as nn
import torch


activation_classes = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


def build_mlp(
    dims: list[int],
    activation: Union[str, type[nn.Module]] = "gelu",
    dropout: float = 0.0,
    input_layernorm: bool = False,
    input_activation: bool = False,
    output_activation: bool = False,
    output_layernorm: bool = False,
    zero_init_last: bool = False,
    hidden_layernorm: bool = False,
) -> nn.Sequential:
    """
    Build an MLP from a list of dimensions.

    Example:
        build_mlp([128, 256, 256, 64])

    gives:
        [optional LayerNorm(128)]
        Linear(128, 256) -> Activation -> Dropout
        Linear(256, 256) -> Activation -> Dropout
        Linear(256, 64)

    Args:
        dims: Full list of dimensions, including input and output.
        activation: Activation class, e.g. nn.ReLU or nn.GELU.
        dropout: Dropout probability after hidden activations.
        input_layernorm: Whether to apply LayerNorm once at the input.
        output_activation: Whether to also apply activation/dropout after the final linear.
        zero_init_last: Whether to initialize the final Linear weight and bias to zero.
        hidden_layernorm: Whether to apply LayerNorm after each hidden linear.
    """
    dims = [int(x) for x in dims]

    activation = (
        activation_classes[activation] if isinstance(activation, str) else activation
    )
    if len(dims) < 2:
        raise ValueError(f"dims must contain at least 2 entries, got {dims}")

    layers: list[nn.Module] = []
    linear_layers: list[nn.Linear] = []

    if input_layernorm:
        layers.append(nn.LayerNorm(dims[0]))

    if input_activation:
        layers.append(activation())

    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = i == len(dims) - 2

        linear = nn.Linear(din, dout)
        layers.append(linear)
        linear_layers.append(linear)

        if not is_last:
            if hidden_layernorm:
                layers.append(nn.LayerNorm(dout))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    if output_layernorm:
        layers.append(nn.LayerNorm(dims[-1]))

    if output_activation:
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    if zero_init_last:
        last_linear = linear_layers[-1]
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    return nn.Sequential(*layers)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation: Union[str, type[nn.Module]] = "gelu",
        dropout: float = 0.0,
        input_layernorm: bool = False,
        output_layernorm: bool = False,
        hidden_layernorm: bool = False,
        input_activation: bool = False,
        output_activation: bool = False,
        zero_init_last: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.mlp = build_mlp(
            dims=dims,
            activation=activation,
            dropout=dropout,
            input_layernorm=input_layernorm,
            output_layernorm=output_layernorm,
            hidden_layernorm=hidden_layernorm,
            input_activation=input_activation,
            output_activation=output_activation,
            zero_init_last=zero_init_last,
        )
        self.use_residual = use_residual
        if self.use_residual and dims[0] != dims[-1]:
            raise ValueError(
                f"Input and output dimensions must match for residual connection. Got {dims}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.mlp(x)
        else:
            return self.mlp(x)
