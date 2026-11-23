import torch
import torch.nn.functional as F
from transformers import AutoModel

from .base import UpsamplerBase, as_size


class BackboneProbe(UpsamplerBase):
    """Final-layer backbone tokens, optionally bilinearly resized to `out_size` (the bilinear baseline).

    `img_in_size` sets the backbone input resolution, e.g. 896 or 1792 px for
    56x56 or 112x112 token grids.
    """

    def __init__(self, backbone_model_name: str, name: str = "backbone", img_in_size: int | None = None,
                 out_size: int | None = None, **kwargs):
        super().__init__(name=name)
        self.backbone = AutoModel.from_pretrained(backbone_model_name).eval().requires_grad_(False)
        self.backbone.compile()
        config = self.backbone.config
        self.patch_size = config.patch_size
        self.n_prefix_tokens = 1 + getattr(config, "num_register_tokens", 0)
        self.img_in_size = img_in_size
        self.out_size = out_size

    @torch.no_grad()
    def forward(self, images, output_size, input_size=None):
        if self.img_in_size is not None:
            images = F.interpolate(images, size=as_size(self.img_in_size), mode="bilinear", align_corners=False)
        else:
            images = self.resize(images, input_size)
        tokens = self.backbone(images).last_hidden_state[:, self.n_prefix_tokens:]
        h, w = (s // self.patch_size for s in images.shape[-2:])
        features = tokens.unflatten(1, (h, w)).permute(0, 3, 1, 2)
        return self.to_bhwc(features, as_size(self.out_size) if self.out_size is not None else None)
