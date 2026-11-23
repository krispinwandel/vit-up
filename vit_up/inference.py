"""Feature extraction with a trained ViT-Up model."""
from contextlib import nullcontext
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from peft import PeftModel
from PIL import Image
from torch import nn

from vit_up import checkpoint
from vit_up.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from vit_up.utils import regular_grid


class ViTUpWrapper(nn.Module):
    """Query dense features from a ViT-Up checkpoint.

    Args:
        model: A release name (see `vit_up.checkpoint.RELEASES`) or a checkpoint directory.
        device: Inference device.
        use_bfloat16: Run under bf16 autocast. Weights and query coordinates stay fp32.
        image_size: Backbone input resolution; defaults to the training resolution.
        query_chunk_size: Maximum number of queries decoded at once.
        compile: Compile the backbone and the ViT-Up decoder.
    """

    def __init__(self, model: str | Path = "dinov3_splus", device: str | None = None, use_bfloat16: bool = True,
                 image_size: int | None = None, query_chunk_size: int | None = None, compile: bool = True):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        vit_up, backbone, self.config = checkpoint.load(model)
        if isinstance(backbone, PeftModel):
            backbone = backbone.merge_and_unload(safe_merge=True)
        self.vit_up = vit_up.to(self.device).eval().requires_grad_(False)
        self.backbone = backbone.to(self.device).eval()
        if compile:
            self.backbone.compile()
            self.vit_up.compile()
        self.use_bfloat16 = use_bfloat16
        self.image_size = image_size or self.config["image_size"]
        self.query_chunk_size = query_chunk_size
        self.feature_dim = self.config["vit_up"]["dim"]
        self.patch_size = self.backbone.patch_size
        self._cache = None
        self._pil_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                                         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def _autocast(self):
        enabled = self.use_bfloat16 and self.device.type in ("cuda", "cpu")
        return torch.autocast(self.device.type, dtype=torch.bfloat16) if enabled else nullcontext()

    def _prepare(self, images: torch.Tensor | Image.Image) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = self._pil_transform(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        return images.to(self.device, torch.float32)

    @torch.no_grad()
    def set_images(self, images: torch.Tensor | Image.Image) -> None:
        """Encode images once; later `forward` calls without images reuse this cache.

        Tensors must be ImageNet-normalized (B, 3, H, W) or (3, H, W). PIL images
        are normalized here. Images are resized to `image_size` for the backbone.
        """
        with self._autocast():
            self._cache = self.vit_up.encode(self._prepare(images), self.backbone, self.image_size)

    @torch.no_grad()
    def forward(self, images: torch.Tensor | Image.Image | None = None, query_coords: torch.Tensor | None = None,
                query_chunk_size: int | None = None, return_all_layers: bool = False):
        """Features at normalized (x, y) query coordinates in [0, 1], shape (B, N, 2).

        Returns (B, N, C) last-layer features, or a list with one tensor per
        selected backbone layer when `return_all_layers=True`.
        """
        if images is not None:
            self.set_images(images)
        if self._cache is None:
            raise RuntimeError("Pass images or call set_images first.")
        query_coords = query_coords.to(self.device, torch.float32)
        chunk = query_chunk_size or self.query_chunk_size or query_coords.shape[1]
        with self._autocast():
            chunks = [self.vit_up(q, self._cache) for q in query_coords.split(chunk, dim=1)]
        layers = [torch.cat(layer, dim=1).float() for layer in zip(*chunks)]
        return layers if return_all_layers else layers[-1]

    def backbone_features(self) -> torch.Tensor:
        """Last-layer backbone features of the current images, (B, h, w, C)."""
        return self._cache["hidden_states"][-1].float()

    @torch.no_grad()
    def dense(self, images: torch.Tensor | Image.Image | None = None, output_size: int = 448,
              query_chunk_size: int | None = None) -> torch.Tensor:
        """Features on a regular output_size x output_size grid of cell centers, shape (B, H, W, C)."""
        if images is not None:
            self.set_images(images)
        batch = len(self._cache["patch_embeddings"])
        queries = regular_grid(output_size, self.device).reshape(1, -1, 2).expand(batch, -1, -1)
        return self(query_coords=queries, query_chunk_size=query_chunk_size).unflatten(1, (output_size, output_size))
