import os
import importlib
import sys
from typing import List, Dict
import numpy as np
import torch
from PIL import Image
from PIL import Image as PILImage
import torchvision.transforms as T
import matplotlib.pyplot as plt

from vit_up.visualization.utils import layout, img_ops
from vit_up.utils import pil_img_utils as pu

from vit_up.layers.backbones import dino_vit_base
from vit_up.layers.backbones.dinov2_vit import DINOv2ViT
from vit_up.visualization.helpers.pca_vis_helper import PCAVisHelper
from transformers import AutoImageProcessor, Dinov2Config

# ============================================
# Backbone
# ============================================


def load_backbone(
    local_model_path="/media/user/ssd2t/huggingface/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056",
    device="cuda",
):
    backbone = DINOv2ViT.init_from_hf(
        backbone_model_name=local_model_path,
        freeze_weights=True,
    )
    backbone = backbone.to(device)
    backbone.eval()

    patch_size = backbone.get_patch_size()
    num_layers = backbone.get_num_layers()
    print(f"✓ Backbone loaded successfully")
    print(f"  Patch size: {patch_size}")
    print(f"  Num layers: {num_layers}")

    # Setup image processor (use model name for processor config)
    processor = AutoImageProcessor.from_pretrained(local_model_path)
    print(f"✓ Image processor configured")

    def image_to_pixel_values(
        image: PILImage.Image,
        device: torch.device,
    ) -> torch.Tensor:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        )
        return transform(image).unsqueeze(0).to(device)

    return {
        "backbone": backbone,
        "processor": processor,
        "image_to_pixel_values": image_to_pixel_values,
    }


def compute_hidden_states_by_scale(
    backbone: DINOv2ViT,
    pixel_values: torch.Tensor,
    image_sizes: List[int],
) -> Dict[int, List[torch.Tensor]]:
    layer_indices = list(range(backbone.get_num_layers()))
    img_size_to_hidden_states_hwc: Dict[int, List[torch.Tensor]] = {}
    for image_size in image_sizes:
        pixel_values_resized = T.Resize(image_size)(pixel_values)
        img_size_to_hidden_states_hwc[image_size] = DINOv2ViT._compute_gt_features(
            backbone=backbone,
            pixel_values=pixel_values_resized,
            layer_indices=layer_indices,
            img_size=image_size,
            window_size=0,
            flatten_hw_to_seq=False,
        )
    return img_size_to_hidden_states_hwc


# ============================================
# Image IO
# ============================================


def load_imagenet1k_image(
    img_idx: int,
    imagenet_path="/media/user/ssd2t/datasets2/imagenet-1k/",
    make_square=True,
):

    import pyarrow.parquet as pq
    import io

    try:
        # Read the first parquet file
        parquet_file = f"{imagenet_path}/data/test-00000-of-00028.parquet"
        table = pq.read_table(parquet_file)

        print(f"✓ Loaded parquet file with {len(table)} images")

        # Extract the first image
        image_struct = table.column("image")[img_idx].as_py()
        image_bytes = image_struct["bytes"]

        # Decode image from bytes
        original_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"✓ Extracted image, original size: {original_image.size}")

    except Exception as e:
        print(f"Error loading image: {e}")
        import traceback

        traceback.print_exc()
        original_image = PILImage.new("RGB", (600, 600), color=(73, 109, 137))

    # Crop to a square without resizing so we keep the original pixel scale
    if make_square:
        square_size = min(original_image.size)
        original_image = T.CenterCrop(square_size)(original_image)

    return original_image


# ============================================
# Visualization
# ============================================


def build_spectral_analysis_image(
    img_size_to_hidden_states_hwc,
    img_size_to_hidden_states_bhwc_semantics,
    img_size_to_spectral_info_raw,
    img_size_to_spectral_info_semantics,
    channel_idx: int,
    layer_idx: int = -1,
    cell_size: int = 128,
):
    import PIL.Image as PILImage

    img_sizes = sorted(img_size_to_spectral_info_raw.keys())

    return pu.concat_images(
        [
            pu.concat_images(
                [
                    pu.concat_images(
                        [
                            pu.heatmap_to_rgb(
                                torch.log1p(
                                    img_size_to_spectral_info_raw[img_size][
                                        "energy_shifted_hwc"
                                    ].mean(dim=-1)
                                )
                                .cpu()
                                .numpy()
                            ),
                            pu.heatmap_to_rgb(
                                torch.log1p(
                                    img_size_to_spectral_info_semantics[img_size][
                                        "energy_shifted_hwc"
                                    ].mean(dim=-1)
                                )
                                .cpu()
                                .numpy()
                            ),
                        ],
                        target_width=cell_size,
                        interpolate_resample=PILImage.Resampling.NEAREST,
                        mode="row",
                    )
                    for img_size in img_sizes
                ],
                # target_width=cell_size,
                interpolate_resample=PILImage.Resampling.NEAREST,
                mode="row",
            ),
            pu.concat_images(
                [
                    pu.concat_images(
                        [
                            pu.heatmap_to_rgb(
                                torch.log1p(
                                    img_size_to_spectral_info_raw[img_size][
                                        "energy_shifted_hwc"
                                    ][:, :, channel_idx]
                                )
                                .cpu()
                                .numpy()
                            ),
                            pu.heatmap_to_rgb(
                                torch.log1p(
                                    img_size_to_spectral_info_semantics[img_size][
                                        "energy_shifted_hwc"
                                    ][:, :, channel_idx]
                                )
                                .cpu()
                                .numpy()
                            ),
                        ],
                        target_width=cell_size,
                        interpolate_resample=PILImage.Resampling.NEAREST,
                        mode="row",
                    )
                    for img_size in img_sizes
                ],
                # target_width=cell_size,
                interpolate_resample=PILImage.Resampling.NEAREST,
                mode="row",
            ),
            pu.concat_images(
                [
                    pu.concat_images(
                        [
                            pu.heatmap_to_rgb(
                                img_size_to_hidden_states_hwc[img_size][layer_idx][
                                    0, :, :, channel_idx
                                ]
                                .cpu()
                                .numpy()
                            ),
                            pu.heatmap_to_rgb(
                                img_size_to_hidden_states_bhwc_semantics[img_size][
                                    layer_idx
                                ][0, :, :, channel_idx]
                                .cpu()
                                .numpy()
                            ),
                        ],
                        target_width=cell_size,
                        interpolate_resample=PILImage.Resampling.NEAREST,
                        mode="row",
                    )
                    for img_size in img_sizes
                ],
                # target_width=cell_size,
                interpolate_resample=PILImage.Resampling.NEAREST,
                mode="row",
            ),
        ],
        mode="col",
        # target_width=cell_size,
        interpolate_resample=PILImage.Resampling.NEAREST,
    )


def project_and_color_layer_features(
    layer_features_by_size: Dict[int, List[torch.Tensor]],
    layer_pca_and_rgb_stats,
    pca_helper: PCAVisHelper,
) -> Dict[int, List[torch.Tensor]]:
    projected_by_size = {}
    for image_size, layer_features in layer_features_by_size.items():
        layer_pca = pca_helper._project_features_to_pca_space(
            layer_features=layer_features,
            layer_pca_and_rgb_stats=layer_pca_and_rgb_stats,
        )
        layer_rgb = pca_helper._layer_pca_features_to_color(
            layer_pca_features=layer_pca,
            layer_pca_and_rgb_stats=layer_pca_and_rgb_stats,
        )
        projected_by_size[image_size] = layer_rgb
    return projected_by_size


def build_pca_row_image(
    input_image: PILImage.Image,
    layer_rgb_features: List[torch.Tensor],
    label: str,
    desc: str,
    cell_width: int,
) -> PILImage.Image:
    input_cell = layout.add_cell_desc(
        input_image,
        desc=desc,
        textbox_width=cell_width,
    )
    layer_cells = []
    for layer_idx, layer_rgb in enumerate(layer_rgb_features):
        layer_cells.append(
            layout.add_cell_desc(
                img_ops.to_pil_rgb(layer_rgb[0]),
                desc=f"layer={layer_idx}",
                textbox_width=cell_width,
                interpolate_resample=PILImage.Resampling.NEAREST,
            )
        )
    return layout.make_row_image(
        [input_cell, *layer_cells], label=label, cell_width=cell_width
    )


def build_pca_image(
    img_size_to_hidden_states_hwc: Dict[int, torch.Tensor],
    img_size_to_hidden_states_bhwc_semantics: Dict[int, torch.Tensor],
    pca_img_sizes=None,
    cell_width=200,
):
    img_sizes = list(img_size_to_hidden_states_hwc.keys())
    if pca_img_sizes is None:
        pca_img_sizes = img_sizes
    pca_vis_helper = PCAVisHelper(
        gt_img_sizes=img_sizes,
        pca_img_sizes=pca_img_sizes,
        pred_out_sizes=[],
        lr_hidden_layer_img_size=448,
        cell_width=cell_width,
    )

    pca_vis_helper.backbone_layer_features_bhwc_by_img_in_size = (
        img_size_to_hidden_states_hwc
    )
    pca_vis_helper.layer_pca_and_rgb_stats = (
        pca_vis_helper._compute_layer_pca_and_rgb_stats(
            backbone_layer_features_bhwc_by_img_in_size=img_size_to_hidden_states_hwc,
            scales_to_use=pca_img_sizes,
            n_components=3,
        )
    )

    # Project the original image features and the averaged canvas-crop features using the same PCA basis.
    img_size_to_raw_pca_rgb = project_and_color_layer_features(
        layer_features_by_size=img_size_to_hidden_states_hwc,
        layer_pca_and_rgb_stats=pca_vis_helper.layer_pca_and_rgb_stats,
        pca_helper=pca_vis_helper,
    )
    img_size_to_denoised_pca_rgb = project_and_color_layer_features(
        layer_features_by_size=img_size_to_hidden_states_bhwc_semantics,
        layer_pca_and_rgb_stats=pca_vis_helper.layer_pca_and_rgb_stats,
        pca_helper=pca_vis_helper,
    )
    return img_size_to_raw_pca_rgb, img_size_to_denoised_pca_rgb


def build_multi_scale_pca_vis(
    original_image: PILImage.Image,
    img_size_to_hidden_states_hwc: Dict[int, torch.Tensor],
    img_size_to_hidden_states_bhwc_semantics: Dict[int, torch.Tensor],
    cell_width: int = 200,
    pca_img_sizes=None,
) -> PILImage.Image:

    img_size_to_raw_pca_rgb, img_size_to_denoised_pca_rgb = build_pca_image(
        img_size_to_hidden_states_hwc=img_size_to_hidden_states_hwc,
        img_size_to_hidden_states_bhwc_semantics=img_size_to_hidden_states_bhwc_semantics,
        pca_img_sizes=pca_img_sizes,
        cell_width=cell_width,
    )

    img_sizes = sorted(img_size_to_hidden_states_hwc.keys())
    rows = []
    for img_size in img_sizes:
        patch_size = img_size_to_hidden_states_hwc[img_size][0].shape[1]
        rows.append(
            build_pca_row_image(
                input_image=original_image.resize(
                    (img_size, img_size), resample=PILImage.Resampling.BICUBIC
                ),
                layer_rgb_features=img_size_to_raw_pca_rgb[img_size],
                # layer_indices=layer_indices,
                label=f"PCA of raw hidden states ({patch_size}x{patch_size})",
                desc=f"RGB ({img_size}x{img_size})",
                cell_width=cell_width,
            )
        )
        rows.append(
            build_pca_row_image(
                input_image=original_image.resize(
                    (img_size, img_size), resample=PILImage.Resampling.BICUBIC
                ),
                layer_rgb_features=img_size_to_denoised_pca_rgb[img_size],
                # layer_indices=layer_indices,
                label=f"PCA of denoised hidden states ({patch_size}x{patch_size})",
                desc=f"RGB ({img_size}x{img_size})",
                cell_width=cell_width,
            )
        )

    return pu.concat_images(
        rows,
        mode="col",
        pad=2,
        pad_color=(0, 0, 0),
    )
