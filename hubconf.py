"""PyTorch Hub entry points, e.g.

    model = torch.hub.load("krispinwandel/vit-up", "vit_up_dinov3_splus", trust_repo=True)
    features = model(images, query_coords)  # (B, N, C) for (B, N, 2) coordinates in [0, 1]

Keyword arguments are passed to `vit_up.inference.ViTUpWrapper`.
"""

dependencies = ["torch", "torchvision", "transformers", "huggingface_hub", "safetensors", "peft", "hydra", "yaml"]


def _load(release: str, pretrained: bool, **kwargs):
    if not pretrained:
        raise ValueError("ViT-Up hub models are always pretrained.")
    from vit_up.inference import ViTUpWrapper
    return ViTUpWrapper(release, **kwargs)


def vit_up_dinov3_splus(pretrained: bool = True, **kwargs):
    """DINOv3-S+, Stage 2 (main model)."""
    return _load("dinov3_splus", pretrained, **kwargs)


def vit_up_dinov3_splus_stage1(pretrained: bool = True, **kwargs):
    """DINOv3-S+, Stage 1."""
    return _load("dinov3_splus_stage1", pretrained, **kwargs)


def vit_up_dinov3_splus_g28_56(pretrained: bool = True, **kwargs):
    """DINOv3-S+, Stage 2, trained at 448 and 896 px (28x28 and 56x56 backbone grids); defaults to 896 px."""
    return _load("dinov3_splus_g28_56", pretrained, **kwargs)


def vit_up_dinov3_base(pretrained: bool = True, **kwargs):
    """DINOv3-B, Stage 2."""
    return _load("dinov3_base", pretrained, **kwargs)


def vit_up_dinov2_small(pretrained: bool = True, **kwargs):
    """DINOv2-S, Stage 2."""
    return _load("dinov2_small", pretrained, **kwargs)
