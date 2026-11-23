import pytest
import torch
from torch import nn

from vit_up.model import ViTUp
from vit_up.model.backbones import Backbone


class FakeBackbone(Backbone):
    """A tiny convolutional stand-in exposing the backbone interface."""

    def __init__(self, dim=32, n_layers=12, patch_size=16):
        super().__init__()
        self.patch_size, self.n_prefix_tokens = patch_size, 0
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.layer = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))

    def embed_patches(self, pixel_values):
        return self.patch_embed(pixel_values).permute(0, 2, 3, 1)

    def forward(self, pixel_values, return_class_token=False):
        x = self.embed_patches(pixel_values)
        states = [x]
        for layer in self.layer:
            x = x + torch.tanh(layer(x))
            states.append(x)
        return (states, x.mean(dim=(1, 2))) if return_class_token else states


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return FakeBackbone()


@pytest.fixture
def vit_up():
    torch.manual_seed(0)
    return ViTUp(dim=32, num_heads=2, layer_indices=[0, 2, 4, 6, 8, 10, 12], query_image_size=128)
