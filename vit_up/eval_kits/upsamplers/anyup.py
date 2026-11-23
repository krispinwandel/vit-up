from pathlib import Path

import torch

from vit_up.model import load_backbone
from .base import UpsamplerBase


class AnyUpUpsampler(UpsamplerBase):
    """AnyUp (multi-backbone checkpoint) on the last backbone layer, guided by the output-size image."""

    def __init__(self, backbone: str, use_natten: bool = True, local_repo: str | None = None, name: str = "anyup",
                 **kwargs):
        super().__init__(name=name)
        source = dict(repo_or_dir=str(Path(local_repo).expanduser()), source="local") if local_repo else dict(repo_or_dir="wimmerth/anyup")
        self.upsampler = torch.hub.load(**source, model="anyup_multi_backbone", use_natten=use_natten)
        self.upsampler.eval().requires_grad_(False)
        self.backbone = load_backbone(backbone).eval()
        self.backbone.compile()

    @torch.no_grad()
    def forward(self, images, output_size, input_size=None):
        features = self.backbone_features(self.backbone, self.resize(images, input_size))
        return self.to_bhwc(self.upsampler(self.resize(images, output_size), features))
