from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from vit_up.model import load_backbone
from .base import UpsamplerBase


class LoftUpUpsampler(UpsamplerBase):
    """Official LoftUp DINOv2-S/14, with code and weights pinned to fixed revisions.

    The checkpoint includes LoftUp's learned input ChannelNorm. `runtime_only`
    substitutes DINOv3-S+ (same width) to measure computational cost only.
    """

    def __init__(self, backbone: str = "facebook/dinov2-small", name: str = "loftup",
                 code_revision: str = "0751a9b3cf7d5d4d1be181abc6f44fe3f43c77d8",
                 revision: str = "bee2049195b60005ea4ec74fb65fcdc28daf0ba9", local_repo: str | None = None,
                 runtime_only: bool = False, attention_need_weights: bool = True, **kwargs):
        super().__init__(name=name)
        if backbone != "facebook/dinov2-small" and not runtime_only:
            raise ValueError("The official LoftUp checkpoint requires facebook/dinov2-small.")
        source = str(Path(local_repo).expanduser()) if local_repo else f"andrehuang/loftup:{code_revision}"
        self.upsampler = torch.hub.load(source, "loftup_dinov2s", pretrained=False, trust_repo=True,
                                        source="local" if local_repo else "github")
        weights = hf_hub_download("haiwen/loftup-dinov2s", "pytorch_model.bin", revision=revision)
        self.upsampler.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        if not attention_need_weights:
            # Upstream discards the attention weights; skipping them enables fused SDPA.
            for module in self.upsampler.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.forward = partial(module.forward, need_weights=False)
        self.backbone = load_backbone(backbone)
        self.requires_grad_(False).eval()
        self.backbone.compile()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images, output_size, input_size=None):
        features = self.backbone_features(self.backbone, self.resize(images, input_size))
        return self.to_bhwc(self.upsampler(features, self.resize(images, output_size)))
