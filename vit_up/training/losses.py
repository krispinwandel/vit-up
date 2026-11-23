"""Feature distances between teacher and student, weighted over layers and scales."""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable


class L2Distance(nn.Module):
    """MSE after standardizing both features by the teacher's per-token mean and std."""
    name = "l2"

    def __init__(self, weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.weight, self.eps = weight, eps

    def forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        mean = target.mean(dim=-1, keepdim=True)
        std = torch.sqrt(target.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return F.mse_loss((pred - mean) / std, (target - mean) / std)


class CosineDistance(nn.Module):
    name = "cos"

    def __init__(self, weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.weight, self.eps = weight, eps

    def forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=self.eps).mean()


class KLDistance(nn.Module):
    """KL divergence between the token-to-token cosine-similarity distributions."""
    name = "kl"

    def __init__(self, weight: float = 1.0, eps: float = 1e-5, tau: float = 0.5):
        super().__init__()
        self.weight, self.eps, self.tau = weight, eps, tau

    def forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        n = target.shape[1]
        target = F.normalize(target, dim=-1, eps=self.eps)
        pred = F.normalize(pred, dim=-1, eps=self.eps)
        self_mask = torch.eye(n, device=target.device, dtype=torch.bool)
        target_logits = (target @ target.transpose(1, 2) / self.tau).masked_fill_(self_mask, -1e4)
        pred_logits = (pred @ pred.transpose(1, 2) / self.tau).masked_fill_(self_mask, -1e4)
        kl = F.kl_div(F.log_softmax(pred_logits, -1), F.log_softmax(target_logits, -1),
                      reduction="batchmean", log_target=True)
        return kl / (n * max(math.log(n), 1.0))


class _AttachGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gradients, *features):
        ctx.save_for_backward(*gradients)
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        return (None, None, *(scale * gradient for gradient in ctx.saved_tensors))


class _GradientAccumulator:
    """Evaluate each distance in fp32 on detached copies and keep only its input gradients.

    The returned loss carries the accumulated gradients back into the original
    graph, so no per-distance graph (e.g. KL's N x N similarities) is retained.
    """

    def __init__(self):
        self.features, self.gradients = {}, {}

    def evaluate(self, distance, target, pred, weight):
        if not torch.is_grad_enabled() or not (target.requires_grad or pred.requires_grad):
            return distance(target.float(), pred.float())
        inputs = [x.detach().float().requires_grad_(x.requires_grad) for x in (target, pred)]
        value = distance(*inputs)
        active = [(x, leaf) for x, leaf in zip((target, pred), inputs) if leaf.requires_grad]
        gradients = torch.autograd.grad(value, [leaf for _, leaf in active], grad_outputs=value.new_tensor(weight),
                                        allow_unused=True)
        for (feature, _), gradient in zip(active, gradients):
            if gradient is not None:
                key = id(feature)
                self.features[key] = feature
                self.gradients[key] = self.gradients[key] + gradient if key in self.gradients else gradient
        return value.detach()

    def attach(self, value):
        if not self.gradients:
            return value
        return _AttachGradients.apply(value, tuple(self.gradients.values()), *self.features.values())


class FeatureLoss(nn.Module):
    """Weighted mean of each distance over scales and layers, summed over distances.

    Inputs are dicts mapping scale -> list of (B, N, C) tensors, one per layer.
    """

    def __init__(self, distances: list[nn.Module], layer_weights: list[float], scale_weights: list[float]):
        super().__init__()
        self.distances = nn.ModuleList(distances)
        self.layer_weights = [float(w) for w in layer_weights]
        self.scale_weights = [float(w) for w in scale_weights]

    def forward(self, targets: dict[int, list[torch.Tensor]], preds: dict[int, list[torch.Tensor]]) -> dict:
        accumulator = _GradientAccumulator()
        total_weight = sum(self.scale_weights) * sum(self.layer_weights)
        total = next(iter(targets.values()))[0].new_zeros(())
        metrics = {}
        for distance in self.distances:
            values, weights = [], []
            for scale_weight, scale in zip(self.scale_weights, targets):
                for layer_weight, target, pred in zip(self.layer_weights, targets[scale], preds[scale]):
                    weight = layer_weight * scale_weight
                    values.append(accumulator.evaluate(distance, target, pred, distance.weight * weight / total_weight))
                    weights.append(values[-1].new_tensor(weight))
            values, weights = torch.stack(values), torch.stack(weights)
            mean = (values * weights).sum() / weights.sum()
            total = total + distance.weight * mean
            metrics[f"{distance.name}_loss"] = mean
        metrics["loss"] = accumulator.attach(total)
        return metrics
