import math
from pathlib import Path

import torch

from .base import UpsamplerBase, as_size


class UpLiftUpsampler(UpsamplerBase):
    """UPLiFT (official Torch Hub models) with the number of 2x iterations matching the output size."""

    def __init__(self, dino_version: int, name: str = "uplift", local_repo: str | None = None,
                 iter_values: list[int] = (1, 2, 3, 4, 5), fast: bool = True, **kwargs):
        super().__init__(name=name)
        entrypoint = {2: "uplift_dinov2_s14", 3: "uplift_dinov3_splus16"}[int(dino_version)]
        source = dict(repo_or_dir=str(Path(local_repo).expanduser()), source="local") if local_repo else dict(repo_or_dir="mwalmer-umd/UPLiFT")
        options = dict(no_transform=True, auto_resize=False, return_base_feat=False, **({"fast": True} if fast else {}))
        self.models = torch.nn.ModuleDict({
            str(iters): torch.hub.load(**source, model=entrypoint, iters=iters, **options).eval()
            for iters in iter_values})
        self.patch_size = self.models[str(iter_values[-1])].extractor.model.patch_embed.proj.kernel_size[0]

    @torch.no_grad()
    def forward(self, images, output_size, input_size=None):
        images = self.resize(images, input_size)
        size = as_size(output_size)
        iters = math.ceil(math.log2(size[0] / (images.shape[-2] / self.patch_size)))
        return self.to_bhwc(self.models[str(iters)](images), size)
