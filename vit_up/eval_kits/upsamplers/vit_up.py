import torch
import torch.nn.functional as F

from vit_up.inference import ViTUpWrapper
from .base import UpsamplerBase, as_size


class ViTUpUpsampler(UpsamplerBase):
    """ViT-Up queried at the cell centers of the output grid.

    With `decode_size`, ViT-Up decodes a fixed decode_size x decode_size grid
    which is then bilinearly resized to the output size.
    """

    def __init__(self, checkpoint: str = "dinov3_splus", name: str = "vit_up", image_size: int | None = None,
                 query_chunk_size: int | None = 12544, decode_size: int | None = None, compile: bool = True,
                 **kwargs):
        super().__init__(name=name)
        self.model = ViTUpWrapper(checkpoint, use_bfloat16=True, image_size=image_size,
                                  query_chunk_size=query_chunk_size, compile=compile)
        self.feature_dim = self.model.feature_dim
        self.image_size = self.model.image_size
        self.query_chunk_size = query_chunk_size
        self.decode_size = decode_size

    @torch.no_grad()
    def get_class_token(self, images: torch.Tensor) -> torch.Tensor:
        with self.model._autocast():
            _, cls = self.model.backbone.hidden_states(self.resize(images, self.image_size), return_class_token=True)
        return cls.float()

    def forward(self, images, output_size, input_size=None):
        size = as_size(output_size)
        features = self.model.dense(self.resize(images, input_size), self.decode_size or size[0])
        if features.shape[1:3] != size:
            features = F.interpolate(features.permute(0, 3, 1, 2), size=size, mode="bilinear",
                                     align_corners=False).permute(0, 2, 3, 1).contiguous()
        return features
