"""
PyTorch Hub configuration for ViTUp models.

This module enables loading ViTUp models via torch.hub.load():

    # Load ViTUp-DINOv3-Base
    model = torch.hub.load('krispinwandel/vit-up', 'vit_up_dinov3_base', pretrained=True)

    # Load ViTUp-DINOv3-SPlus
    model = torch.hub.load('krispinwandel/vit-up', 'vit_up_dinov3_splus', pretrained=True)

    # Run inference
    images = torch.randn(1, 3, 448, 448)
    query_coords = torch.rand(1, 100, 2)  # Normalized coordinates (0-1)
    features = model(images, query_coords)

"""

import torch

dependencies = [
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "omegaconf",
    "peft",
]


def vit_up_dinov3_base(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pretrained: bool = True,
    use_bfloat16: bool = True,
    hidden_layer_img_size: int = 448,
    query_chunk_size: int | None = None,
) -> "ViTUpWrapper":
    """
    Load ViTUp model with DINOv3-Base backbone.

    Args:
        device: Device to load model on ('cpu', 'cuda', 'mps')
        pretrained: Kept for standard torch.hub compatibility. ViT-Up Hub
            models are always loaded with pretrained weights.
        use_bfloat16: Whether to use bf16 inference and model weights.
        hidden_layer_img_size: Image size used to compute hidden backbone states.
        query_chunk_size: Number of query points to process per chunk.

    Returns:
        ViTUpWrapper instance ready for inference
    """
    from vit_up.inference.vit_up_wrapper import ViTUpWrapper

    if not pretrained:
        raise ValueError("torch.hub ViT-Up models require pretrained=True.")

    return ViTUpWrapper(
        "vit_up_dinov3_base",
        device=device,
        use_bfloat16=use_bfloat16,
        hidden_layer_img_size=hidden_layer_img_size,
        query_chunk_size=query_chunk_size,
    )


def vit_up_dinov3_splus(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pretrained: bool = True,
    use_bfloat16: bool = True,
    hidden_layer_img_size: int = 448,
    query_chunk_size: int | None = None,
) -> "ViTUpWrapper":
    """
    Load ViTUp model with DINOv3-SPlus backbone.

    Args:
        device: Device to load model on ('cpu', 'cuda', 'mps')
        pretrained: Kept for standard torch.hub compatibility. ViT-Up Hub
            models are always loaded with pretrained weights.
        use_bfloat16: Whether to use bf16 inference and model weights.
        hidden_layer_img_size: Image size used to compute hidden backbone states.
        query_chunk_size: Number of query points to process per chunk.

    Returns:
        ViTUpWrapper instance ready for inference
    """
    from vit_up.inference.vit_up_wrapper import ViTUpWrapper

    if not pretrained:
        raise ValueError("torch.hub ViT-Up models require pretrained=True.")

    return ViTUpWrapper(
        "vit_up_dinov3_splus",
        device=device,
        use_bfloat16=use_bfloat16,
        hidden_layer_img_size=hidden_layer_img_size,
        query_chunk_size=query_chunk_size,
    )


# Additional model variants can be added here
# def vit_up_dinov2_base(...):
#     ...
