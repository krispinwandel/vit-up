import math
from typing import Any, Dict, List, Optional, Union, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistLossModuleBase(nn.Module):
    def __init__(
        self,
        name: str,
        apply_at: Union[int, str] = "all",
        weight: float = 1.0,
    ):
        super().__init__()
        if not isinstance(name, str) or not name:
            raise ValueError("Dist loss module 'name' must be a non-empty string.")
        self.name = name
        self.apply_at = apply_at
        self.weight = float(weight)

    def applies_to(self, layer_idx: int, n_layers: int) -> bool:
        if isinstance(self.apply_at, int):
            return layer_idx == self.apply_at

        scope = str(self.apply_at).lower()
        if scope == "all":
            return True
        if scope == "first":
            return layer_idx == 0
        if scope == "last":
            return layer_idx == n_layers - 1

        raise ValueError(
            "apply_at must be one of {'all', 'first', 'last'} or an int layer index. "
            f"Got apply_at={self.apply_at!r} for module '{self.name}'."
        )

    def forward(self, gt_feat: torch.Tensor, pred_feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class L2DistLoss(DistLossModuleBase):
    def __init__(
        self,
        name: str = "l2",
        apply_at: Union[int, str] = "all",
        weight: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__(name=name, apply_at=apply_at, weight=weight)
        self.eps = float(eps)

    def forward(self, gt_feat: torch.Tensor, pred_feat: torch.Tensor) -> torch.Tensor:
        target_mean = gt_feat.mean(dim=-1, keepdim=True)
        target_var = gt_feat.var(dim=-1, keepdim=True, unbiased=False)
        target_std = torch.sqrt(target_var + self.eps)
        target_normalized = (gt_feat - target_mean) / target_std
        pred_normalized = (pred_feat - target_mean) / target_std
        return F.mse_loss(pred_normalized, target_normalized)
        # return F.mse_loss(pred_feat, gt_feat)


class CosineDistLoss(DistLossModuleBase):
    def __init__(
        self,
        name: str = "cos",
        apply_at: Union[int, str] = "all",
        weight: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__(name=name, apply_at=apply_at, weight=weight)
        self.eps = float(eps)

    def forward(self, gt_feat: torch.Tensor, pred_feat: torch.Tensor) -> torch.Tensor:
        return (
            1.0 - F.cosine_similarity(pred_feat, gt_feat, dim=-1, eps=self.eps).mean()
        )


class KLDistLoss(DistLossModuleBase):
    def __init__(
        self,
        name: str = "kl",
        apply_at: Union[int, str] = "all",
        weight: float = 1.0,
        eps: float = 1e-5,
        tau: float = 0.5,
    ):
        super().__init__(name=name, apply_at=apply_at, weight=weight)
        self.eps = float(eps)
        self.tau = float(tau)

    def forward(self, gt_feat: torch.Tensor, pred_feat: torch.Tensor) -> torch.Tensor:
        _, s, _ = gt_feat.shape

        gt_feat_norm = F.normalize(gt_feat, p=2, dim=-1, eps=self.eps)
        pred_feat_norm = F.normalize(pred_feat, p=2, dim=-1, eps=self.eps)

        sim_gt = torch.bmm(gt_feat_norm, gt_feat_norm.transpose(1, 2))
        sim_pred = torch.bmm(pred_feat_norm, pred_feat_norm.transpose(1, 2))

        sim_gt = sim_gt / self.tau
        sim_pred = sim_pred / self.tau

        mask = torch.eye(s, device=gt_feat.device, dtype=torch.bool)
        sim_gt.masked_fill_(mask, -1e4)
        sim_pred.masked_fill_(mask, -1e4)

        log_prob_gt = F.log_softmax(sim_gt, dim=-1)
        log_prob_pred = F.log_softmax(sim_pred, dim=-1)

        kl_raw = F.kl_div(
            log_prob_pred, log_prob_gt, reduction="batchmean", log_target=True
        )
        entropy_norm = max(math.log(s), 1.0)
        return kl_raw / (s * entropy_norm)


class MultiScaleAlignedFeatureLoss(nn.Module):
    def __init__(
        self,
        dist_module_args: List[DistLossModuleBase],
        layer_weights: List[float],
        scale_weights: List[float],
    ):
        super().__init__()
        self.dist_modules = self._init_dist_modules(dist_module_args)
        self.scale_weights = [float(weight) for weight in scale_weights]
        self.layer_weights = [float(weight) for weight in layer_weights]

    def _init_dist_modules(
        self, dist_module_args: List[DistLossModuleBase]
    ) -> nn.ModuleList:
        if not isinstance(dist_module_args, list) or not dist_module_args:
            raise ValueError("dist_module_args must be a non-empty list.")

        modules: List[DistLossModuleBase] = []
        used_names = set()
        for module_idx, module_init in enumerate(dist_module_args):
            if isinstance(module_init, DistLossModuleBase):
                module_obj = module_init
            else:
                raise TypeError(
                    "Each dist module must be a DistLossModuleBase instance. "
                    f"Got {type(module_init)} at dist_module_args[{module_idx}]."
                )
            if not isinstance(module_obj, DistLossModuleBase):
                raise TypeError(
                    "Each dist module must inherit DistLossModuleBase. "
                    f"Got type={type(module_obj)} at dist_module_args[{module_idx}]."
                )
            if module_obj.name in used_names:
                raise ValueError(
                    f"Duplicate dist module name '{module_obj.name}'. Names must be unique."
                )
            used_names.add(module_obj.name)
            modules.append(module_obj)
        return nn.ModuleList(modules)

    def forward(
        self,
        gt_layers_by_scale: Dict[int, List[torch.Tensor]],
        pred_layers_by_scale: Dict[int, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:

        scales = list(gt_layers_by_scale.keys())
        if not scales:
            raise ValueError(
                "MultiScaleAlignedFeatureLoss requires non-empty per-scale aligned tensors."
            )
        if set(gt_layers_by_scale.keys()) != set(pred_layers_by_scale.keys()):
            raise ValueError(
                "gt/pred scale keys must match in MultiScaleAlignedFeatureLoss. "
                f"Got gt_scales={list(gt_layers_by_scale.keys())}, "
                f"pred_scales={list(pred_layers_by_scale.keys())}."
            )
        if len(self.scale_weights) != len(scales):
            raise ValueError(
                "scale_weights length must match number of scales. "
                f"Got len(scale_weights)={len(self.scale_weights)}, n_scales={len(scales)}."
            )
        if any(weight < 0 for weight in self.layer_weights):
            raise ValueError("layer_weights must be non-negative.")
        if any(weight < 0 for weight in self.scale_weights):
            raise ValueError("scale_weights must be non-negative.")

        reference_scale = scales[0]
        n_layers = len(gt_layers_by_scale[reference_scale])
        if n_layers == 0:
            raise ValueError("Each scale must contain at least one layer tensor.")
        if len(self.layer_weights) != n_layers:
            raise ValueError(
                "layer_weights length must match number of layers per scale. "
                f"Got len(layer_weights)={len(self.layer_weights)}, n_layers={n_layers}."
            )

        zero = gt_layers_by_scale[reference_scale][0].new_zeros(())
        loss_total = zero
        out: Dict[str, torch.Tensor] = {"loss": zero}

        for module in self.dist_modules:
            module = cast(DistLossModuleBase, module)  # for type checker
            module_layer_losses: List[torch.Tensor] = []
            module_layer_weight_values: List[torch.Tensor] = []
            module_last = zero

            for scale_idx, scale in enumerate(scales):
                gt_layers = gt_layers_by_scale[scale]
                pred_layers = pred_layers_by_scale[scale]
                if len(gt_layers) != len(pred_layers):
                    raise ValueError(
                        "gt/pred layer counts must match per scale. "
                        f"Got scale={scale}, len(gt_layers)={len(gt_layers)}, "
                        f"len(pred_layers)={len(pred_layers)}."
                    )
                if len(gt_layers) != n_layers:
                    raise ValueError(
                        "Each scale must contain the same number of layers. "
                        f"Got reference n_layers={n_layers}, scale={scale} has {len(gt_layers)}."
                    )

                scale_weight = float(self.scale_weights[scale_idx])
                for layer_idx, (gt_layer, pred_layer) in enumerate(
                    zip(gt_layers, pred_layers)
                ):
                    if gt_layer.shape != pred_layer.shape:
                        raise ValueError(
                            "Aligned gt/pred shapes must match exactly before loss computation. "
                            f"Got gt={tuple(gt_layer.shape)}, pred={tuple(pred_layer.shape)}, scale={scale}."
                        )
                    if not module.applies_to(layer_idx=layer_idx, n_layers=n_layers):
                        continue
                    layer_loss = module(gt_layer.float(), pred_layer.float())
                    module_layer_losses.append(layer_loss)
                    combined_weight = float(
                        self.layer_weights[layer_idx] * scale_weight
                    )
                    module_layer_weight_values.append(
                        layer_loss.new_tensor(combined_weight)
                    )
                    module_last = layer_loss

            if not module_layer_losses:
                raise ValueError(
                    f"Dist module '{module.name}' did not apply to any layer. "
                    "Check its apply_at configuration."
                )

            stacked_losses = torch.stack(module_layer_losses)
            stacked_weights = torch.stack(module_layer_weight_values)
            weight_sum = stacked_weights.sum()
            if float(weight_sum.item()) <= 0.0:
                raise ValueError(
                    f"Sum of selected layer/scale weights must be > 0 for module '{module.name}'."
                )
            module_mean = (stacked_losses * stacked_weights).sum() / weight_sum
            loss_total = loss_total + module.weight * module_mean
            out[f"{module.name}_loss"] = module_mean
            out[f"{module.name}_loss_last"] = module_last

        out["loss"] = loss_total
        return out


class AlignedFeatureLoss(MultiScaleAlignedFeatureLoss):
    def __init__(
        self,
        dist_module_args: List[DistLossModuleBase],
        layer_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            dist_module_args=dist_module_args,
            layer_weights=(
                layer_weights if layer_weights is not None else [1.0]
            ),  # default to 1.0 for single scale/layer
            scale_weights=[1.0],  # single scale, so weight is 1.0
        )

    def forward(
        self,
        gt_layers_by_scale: Dict[int, List[torch.Tensor]],
        pred_layers_by_scale: Dict[int, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if len(gt_layers_by_scale) != 1 or len(pred_layers_by_scale) != 1:
            raise ValueError(
                "AlignedFeatureLoss expects exactly one scale of features. "
                f"Got gt_scales={list(gt_layers_by_scale.keys())}, "
                f"pred_scales={list(pred_layers_by_scale.keys())}."
            )
        return super().forward(gt_layers_by_scale, pred_layers_by_scale)
