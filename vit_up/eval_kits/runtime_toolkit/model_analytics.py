from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn


def _count_params(params: Iterable[torch.nn.Parameter]) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in params:
        num = int(param.numel())
        total += num
        if param.requires_grad:
            trainable += num
    return total, trainable


def _format_param_count(value: int) -> str:
    return f"{value:,}"


def _resolve_backbone(model: nn.Module) -> Optional[nn.Module]:
    if hasattr(model, "backbone"):
        backbone = getattr(model, "backbone")
        if isinstance(backbone, nn.Module):
            return backbone
    if hasattr(model, "vit_up"):
        vit_up = getattr(model, "vit_up")
        if isinstance(vit_up, nn.Module) and hasattr(vit_up, "backbone"):
            backbone = getattr(vit_up, "backbone")
            if isinstance(backbone, nn.Module):
                return backbone
    return None


def _resolve_uplift_models(model: nn.Module) -> list[nn.Module]:
    uplift_models: list[nn.Module] = []
    uplift_dict = getattr(model, "model_uplift", None)
    if isinstance(uplift_dict, dict):
        for item in uplift_dict.values():
            if isinstance(item, nn.Module):
                uplift_models.append(item)
    return uplift_models


def _resolve_uplift_backbone(models: list[nn.Module]) -> Optional[nn.Module]:
    for model in models:
        extractor = getattr(model, "extractor", None)
        if isinstance(extractor, nn.Module) and hasattr(extractor, "model"):
            backbone = getattr(extractor, "model")
            if isinstance(backbone, nn.Module):
                return backbone
        if isinstance(extractor, nn.Module):
            return extractor
    return None


def _is_lora_param(name: str) -> bool:
    name_lc = name.lower()
    return "lora" in name_lc


def _count_module_params(module: nn.Module, recurse: bool) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters(recurse=recurse):
        num = int(param.numel())
        total += num
        if param.requires_grad:
            trainable += num
    return total, trainable


def _print_param_tree(roots: List[Tuple[str, nn.Module]]) -> None:
    def _children_with_params(module: nn.Module) -> List[Tuple[str, nn.Module]]:
        children: List[Tuple[str, nn.Module]] = []
        for name, child in module.named_children():
            subtree_total, _ = _count_module_params(child, recurse=True)
            if subtree_total > 0:
                children.append((name, child))
        return children

    def _format_counts(module: nn.Module) -> str:
        direct_total, direct_trainable = _count_module_params(module, recurse=False)
        subtree_total, subtree_trainable = _count_module_params(module, recurse=True)
        return (
            f"subtree={_format_param_count(subtree_total)}"
            f" (train={_format_param_count(subtree_trainable)}), "
            f"direct={_format_param_count(direct_total)}"
            f" (train={_format_param_count(direct_trainable)})"
        )

    def _print_node(
        name: str,
        module: nn.Module,
        prefix: str,
        is_last: bool,
    ) -> None:
        branch = "`-- " if is_last else "|-- "
        print(
            f"{prefix}{branch}{name} [{module.__class__.__name__}] "
            f"{_format_counts(module)}"
        )
        child_prefix = prefix + ("    " if is_last else "|   ")
        children = _children_with_params(module)
        for idx, (child_name, child) in enumerate(children):
            _print_node(child_name, child, child_prefix, idx == len(children) - 1)

    print("Param tree (only modules with parameters):")
    for idx, (name, root) in enumerate(roots):
        print(f"{name} [{root.__class__.__name__}] {_format_counts(root)}")
        root_children = _children_with_params(root)
        for child_idx, (child_name, child) in enumerate(root_children):
            _print_node(
                child_name,
                child,
                "",
                child_idx == len(root_children) - 1,
            )
        if idx != len(roots) - 1:
            print("")


def _print_param_table(
    rows: List[Tuple[str, int, int]],
) -> None:
    name_width = max(len(row[0]) for row in rows)
    total_width = max(len(_format_param_count(row[1])) for row in rows)
    train_width = max(len(_format_param_count(row[2])) for row in rows)

    header = (
        f"{'Component':<{name_width}}  "
        f"{'Total':>{total_width}}  "
        f"{'Trainable':>{train_width}}"
    )
    print("Param summary:")
    print(header)
    print(f"{'-' * name_width}  " f"{'-' * total_width}  " f"{'-' * train_width}")
    for name, total, trainable in rows:
        print(
            f"{name:<{name_width}}  "
            f"{_format_param_count(total):>{total_width}}  "
            f"{_format_param_count(trainable):>{train_width}}"
        )


def print_model_and_params(model: nn.Module) -> None:
    model_params = list(model.parameters())
    uplift_models = []
    if not model_params:
        uplift_models = _resolve_uplift_models(model)
        if uplift_models:
            model_params = [
                param
                for uplift_model in uplift_models
                for param in uplift_model.parameters()
            ]

    if uplift_models:
        all_named_params: list[tuple[str, nn.Parameter]] = []
        seen_params: set[int] = set()
        for uplift_model in uplift_models:
            for name, param in uplift_model.named_parameters():
                param_id = id(param)
                if param_id in seen_params:
                    continue
                seen_params.add(param_id)
                all_named_params.append((name, param))

        model_total, model_trainable = _count_params(
            [param for _, param in all_named_params]
        )

        backbone = _resolve_uplift_backbone(uplift_models)
        if backbone is None:
            backbone_param_ids: set[int] = set()
            backbone_total = 0
            backbone_trainable = 0
            lora_total = 0
            lora_trainable = 0
        else:
            backbone_param_ids = {id(param) for param in backbone.parameters()}
            backbone_total, backbone_trainable = _count_params(backbone.parameters())
            lora_params = [
                param
                for name, param in backbone.named_parameters()
                if _is_lora_param(name)
            ]
            lora_total, lora_trainable = _count_params(lora_params)

        model_no_backbone_params = [
            param
            for _, param in all_named_params
            if id(param) not in backbone_param_ids
        ]
        model_no_backbone_total, model_no_backbone_trainable = _count_params(
            model_no_backbone_params
        )
    else:
        model_total, model_trainable = _count_params(model_params)

        backbone = _resolve_backbone(model)
        if backbone is None:
            backbone_total = 0
            backbone_trainable = 0
            lora_total = 0
            lora_trainable = 0
        else:
            backbone_total, backbone_trainable = _count_params(backbone.parameters())
            lora_params = [
                param
                for name, param in backbone.named_parameters()
                if _is_lora_param(name)
            ]
            lora_total, lora_trainable = _count_params(lora_params)

        model_no_backbone_total = max(0, model_total - backbone_total)
        model_no_backbone_trainable = max(0, model_trainable - backbone_trainable)

    backbone_no_lora_total = max(0, backbone_total - lora_total)
    backbone_no_lora_trainable = max(0, backbone_trainable - lora_trainable)

    roots: List[Tuple[str, nn.Module]] = [("<model>", model)]
    if not list(model.parameters()) and uplift_models:
        roots = [
            (f"<uplift:{idx}>", uplift) for idx, uplift in enumerate(uplift_models)
        ]

    _print_param_tree(roots)
    print("")

    _print_param_table(
        [
            (
                "Model (excl backbone)",
                model_no_backbone_total,
                model_no_backbone_trainable,
            ),
            (
                "Backbone (excl LoRA)",
                backbone_no_lora_total,
                backbone_no_lora_trainable,
            ),
            ("LoRA", lora_total, lora_trainable),
        ]
    )
