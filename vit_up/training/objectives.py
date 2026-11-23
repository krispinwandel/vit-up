"""Training objectives. Each computes the loss for one student resolution and, when
training, backpropagates it immediately so that per-chunk graphs can be freed.

Both stages distill from a teacher view (full image or crop) into a student view
in which that region appears downscaled on a larger canvas (see `vit_up.data`).
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from vit_up.model.backbones import disable_adapters
from vit_up.utils import regular_grid
from .losses import FeatureLoss


def to_blocks(x: torch.Tensor, size: int) -> torch.Tensor:
    """(..., H, W, C) -> (..., H/size * W/size, size, size, C), blocks in row-major order."""
    x = x.unflatten(-3, (x.shape[-3] // size, size)).unflatten(-2, (x.shape[-2] // size, size))
    return x.transpose(-4, -3).flatten(-5, -4)


def box_coordinates(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Map normalized points inside (B, 4) xyxy boxes to the boxes' parent coordinates."""
    shape = (len(boxes),) + (1,) * (points.dim() - 2) + (2,)
    origin, extent = boxes[:, :2].reshape(shape), (boxes[:, 2:] - boxes[:, :2]).reshape(shape)
    return origin + points * extent


class GridDistillation(nn.Module):
    """Stage 1: match pooled student queries to the frozen backbone at several resolutions.

    The student is queried on a regular grid over the teacher region with
    `block = 2 ** (n_scales - 1)` queries per cell of the coarsest teacher grid
    (`grid_size` tokens per side). Average-pooling the predictions yields one
    prediction per token of the teacher at `grid_size * 2 ** s` tokens per side,
    for every scale s. Teacher features come from the backbone without adapters.
    """

    def __init__(self, loss: FeatureLoss, patch_size: int, grid_size: int = 14, n_scales: int = 3,
                 blocks_per_chunk: int = 196):
        super().__init__()
        self.loss = loss
        self.block = 2 ** (n_scales - 1)
        self.teacher_sizes = [patch_size * grid_size * 2 ** s for s in range(n_scales)]
        self.blocks_per_chunk = blocks_per_chunk
        self.register_buffer("query_blocks", to_blocks(regular_grid(grid_size * self.block), self.block),
                             persistent=False)

    def setup(self, model):
        pass

    def prepare(self, model, batch):
        queries = box_coordinates(self.query_blocks[None], batch["teacher_box"])
        with torch.no_grad(), disable_adapters(model.backbone):
            targets = []
            for scale, size in enumerate(self.teacher_sizes):
                hidden_states = model.backbone.hidden_states(batch["teacher_image"], size)
                targets.append([to_blocks(hidden_states[i], 2 ** scale) for i in model.vit_up.layer_indices])
        return queries, targets

    def _pool(self, pred: torch.Tensor, size: int) -> torch.Tensor:
        """(B, N, block, block, C) query predictions to (B, N * size * size, C) cell averages."""
        x = pred.flatten(0, 1).permute(0, 3, 1, 2)
        if size != self.block:
            x = F.avg_pool2d(x, kernel_size=self.block // size)
        return x.permute(0, 2, 3, 1).reshape(len(pred), -1, pred.shape[-1])

    def forward(self, model, batch, prepared, image_size: int, backward_weight: float | None = None) -> dict:
        queries, targets = prepared
        cache = model.vit_up.encode(batch["student_image"], model.backbone, image_size)
        starts = range(0, queries.shape[1], self.blocks_per_chunk)
        metrics = {}
        for i, start in enumerate(starts):
            chunk = slice(start, start + self.blocks_per_chunk)
            chunk_queries = queries[:, chunk]
            preds = model.vit_up(chunk_queries.flatten(1, 3), cache)
            preds = [pred.unflatten(1, chunk_queries.shape[1:4]) for pred in preds]
            chunk_targets = {scale: [layer[:, chunk].flatten(1, 3) for layer in layers]
                             for scale, layers in enumerate(targets)}
            chunk_preds = {scale: [self._pool(pred, 2 ** scale) for pred in preds] for scale in chunk_targets}
            chunk_metrics = self.loss(chunk_targets, chunk_preds)
            if backward_weight is not None:
                model.manual_backward(chunk_metrics["loss"] * (backward_weight / len(starts)),
                                      retain_graph=i < len(starts) - 1)
            for key, value in chunk_metrics.items():
                metrics[key] = metrics.get(key, 0) + value.detach()
        return {key: value / len(starts) for key, value in metrics.items()}


class OffsetHead(nn.Module):
    """Predict a bounded (dx, dy) displacement from a query's features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, max_offset: float = 0.15):
        super().__init__()
        self.max_offset = max_offset
        self.mlp = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(features).float()) * self.max_offset


class OffsetDistillation(nn.Module):
    """Stage 2: distill a frozen copy of the Stage 1 model into itself across views.

    Queries are sampled uniformly in the teacher region. The student must match
    the teacher at the same point (anchor term) and at a learned, regularized
    displacement of that point (offset term), which tolerates small
    misalignments between the low-resolution student and the teacher view.
    """

    def __init__(self, loss: FeatureLoss, offset_head: OffsetHead, num_queries: int = 3136,
                 anchor_weight: float = 0.1, offset_weight: float = 1.0, offset_regularization_weight: float = 10.0,
                 validation_seed: int = 1234):
        super().__init__()
        self.loss = loss
        self.offset_head = offset_head
        self.num_queries = num_queries
        self.anchor_weight = anchor_weight
        self.offset_weight = offset_weight
        self.offset_regularization_weight = offset_regularization_weight
        self.validation_seed = validation_seed

    def setup(self, model):
        self.teacher = copy.deepcopy(model.vit_up).requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "teacher"):
            self.teacher.eval()
        return self

    def prepare(self, model, batch):
        boxes = batch["teacher_box"]
        generator = None if model.training else torch.Generator(boxes.device).manual_seed(self.validation_seed)
        teacher_queries = torch.rand(len(boxes), self.num_queries, 2, device=boxes.device, generator=generator)
        return teacher_queries, box_coordinates(teacher_queries, boxes)

    def forward(self, model, batch, prepared, image_size: int, backward_weight: float | None = None) -> dict:
        teacher_queries, student_queries = prepared
        with torch.no_grad():
            teacher_cache = self.teacher.encode(batch["teacher_image"], model.backbone, image_size)
            targets = self.teacher(teacher_queries, teacher_cache)
        student_cache = model.vit_up.encode(batch["student_image"], model.backbone, image_size)
        preds = model.vit_up(student_queries, student_cache)
        anchor = self.loss({0: targets}, {0: preds})

        offsets = self.offset_head(preds[-1])
        shifted_queries = (teacher_queries.float() + offsets).clamp(0, 1)
        shifted_targets = self.teacher(shifted_queries, teacher_cache)
        offset = self.loss({0: shifted_targets}, {0: preds})
        regularization = offsets.float().square().sum(dim=-1).mean()

        loss = (self.anchor_weight * anchor["loss"] + self.offset_weight * offset["loss"]
                + self.offset_regularization_weight * regularization)
        if backward_weight is not None:
            model.manual_backward(loss * backward_weight)
        return {"loss": loss.detach(), **{f"anchor_{k}": v.detach() for k, v in anchor.items()},
                **{f"offset_{k}": v.detach() for k, v in offset.items()},
                "offset_regularization": regularization.detach(),
                "offset_norm": (shifted_queries - teacher_queries).detach().norm(dim=-1).mean()}
