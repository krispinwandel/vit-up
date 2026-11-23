from pathlib import Path

import torch

from .backbones import PretrainedViTWrapper
from .base import UpsamplerBase, as_size
from .src.jafar.jafar import JAFAR

WEIGHTS = Path(__file__).resolve().parent / "weights" / "jafar"


class JAFARUpsampler(UpsamplerBase):
    """JAFAR with the official checkpoints in `weights/jafar/<timm backbone>.pth`."""

    def __init__(self, backbone: str, feature_dim: int, name: str = "jafar", checkpoint_dir: str | None = None,
                 **kwargs):
        super().__init__(name=name)
        self.backbone = PretrainedViTWrapper(name=backbone).eval().requires_grad_(False)
        self.backbone.compile()
        self.jafar = JAFAR(input_dim=3, qk_dim=128, v_dim=feature_dim, feature_dim=feature_dim, kernel_size=1,
                           num_heads=4)
        checkpoint = torch.load(Path(checkpoint_dir or WEIGHTS) / f"{backbone}.pth", map_location="cpu",
                                weights_only=False)
        self.jafar.load_state_dict(checkpoint.get("jafar", checkpoint), strict=False)
        self.jafar.eval()

    def forward(self, images, output_size, input_size=None):
        with torch.no_grad():
            features = self.backbone(self.resize(images, input_size))
            features = features[0] if isinstance(features, tuple) else features
        return self.to_bhwc(self.jafar(images, features, as_size(output_size)))
