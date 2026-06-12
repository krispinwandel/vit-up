import requests
from PIL import Image
from io import BytesIO

import torch
import torch.nn.functional as F

from vit_up.inference import inference_wrappers
from vit_up.utils import pca_utils
from vit_up.layers.backbones.dino_vit_base import DinoViTBackboneBase


from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def reset_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def get_hydra_run_dir(project_root: str):
    return f"{project_root}/output/debug/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}"


def build_cfg(
    dataset_name: str,
    data_root: str,
    img_size: int,
    batch_size: int,
    config_dir: str,
    hydra_run_dir: str,
):
    reset_hydra()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="probing",
            overrides=[
                "schedule/mode=eval",
                f"schedule/dataset={dataset_name}",
                f"hydra.run.dir={hydra_run_dir}",
                f"img_size={img_size}",
                f"target_size={img_size}",
                f"data_root={data_root}",
                f"train_dataloader.batch_size={batch_size}",
                f"val_dataloader.batch_size={batch_size}",
            ],
        )
    return cfg


def get_probing_dataset(
    dataset_name: str,
    img_size: int = 448,
    batch_size=1,
    hydra_run_dir=None,
    datasets_root=None,
    device="cuda",
):
    from vit_up.eval_kits.probing_toolkit.utils.training import get_dataloaders
    from pathlib import Path

    pkg_dir = Path(__file__).parent / "../.."

    probing_config_dir = pkg_dir / "nf_dino" / "eval_kits" / "config" / "probing"
    datasets_root = (
        Path(datasets_root) if datasets_root is not None else pkg_dir / "data"
    )

    hydra_run_dir = (
        Path(hydra_run_dir) if hydra_run_dir is not None else get_hydra_run_dir(pkg_dir)
    )

    cfg = build_cfg(
        dataset_name=dataset_name,
        data_root=datasets_root,
        img_size=img_size,
        batch_size=batch_size,
        config_dir=probing_config_dir,
        hydra_run_dir=hydra_run_dir,
    )

    backbone = instantiate(cfg.backbone).to(device).eval()

    _, val_loader = get_dataloaders(cfg, backbone, is_evaluation=True, shuffle=False)
    return val_loader, cfg


def download_image(url, save_path=None):
    """Download an image from URL and return PIL Image object"""
    # Make sure URL is a string, not a tuple
    if isinstance(url, tuple):
        url = url[0]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")

    if save_path:
        image.save(save_path)
        print(f"Image saved to: {save_path}")

    return image


def compute_baseline_features(
    backbone: DinoViTBackboneBase,
    pixel_values: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Compute baseline features from the ViTUpPL backbone using _compute_gt_features.
    Then bilinear upsample to target resolution.

    Args:
        backbone: DINO backbone model
        pixel_values: (B, C, H, W) input images
        target_h: target height
        target_w: target width

    Returns:
        features: (B, C, target_h, target_w)
    """
    if backbone is None:
        raise ValueError("Backbone is not available. Ensure ViTUpPL is loaded.")

    with torch.no_grad():
        # Use the backbone API directly for ground-truth features
        feats_hwc = backbone._compute_gt_features(
            backbone,
            pixel_values=pixel_values,
            layer_indices=[12],
            flatten_hw_to_seq=False,
        )[0]

        # Convert (B, H, W, C) -> (B, C, H, W)
        if feats_hwc.ndim == 3:
            feats_hwc = feats_hwc.unsqueeze(0)
        feats = feats_hwc.permute(0, 3, 1, 2)

        # Upsample to target resolution with bilinear interpolation
        feats_upsampled = F.interpolate(
            feats,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return feats_upsampled


def compute_vitup_features(
    vit_up_method,
    pixel_values: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Compute features from ViTUpPL Lightning module using inference wrappers.

    Args:
        vit_up_method: ViTUpPL Lightning module
        pixel_values: (B, C, H, W) input images
        target_h: target height
        target_w: target width

    Returns:
        features: (B, C, target_h, target_w)
    """
    with torch.no_grad():
        # Use inference_wrappers to compute features with proper coordinates
        # First, compute query coordinates for target resolution
        q_xy_normalized_by_out_size = (
            inference_wrappers.compute_query_coords_by_out_res([target_h, target_w])
        )  # {target_h: coords, target_w: coords, ...}

        # Compute upsampled features from ViTUpPL for all query coords
        q_fts_by_out_size = inference_wrappers.compute_layer_query_features(
            q_xy_normalized_by_out_size,
            pixel_values,
            vit_up_method,
        )  # {out_res: [layer_0_feats, layer_1_feats, ...]}

        # Get features at target resolution, use last layer
        target_res_key = target_h  # Assuming square output
        feats_list = q_fts_by_out_size[target_res_key]
        feats_hwc = feats_list[-1]  # (B, H, W, C) or (H, W, C) if B=1

        # Ensure batch dimension and convert to (B, C, H, W)
        if feats_hwc.ndim == 3:  # (H, W, C) when B=1
            feats_hwc = feats_hwc.unsqueeze(0)  # -> (1, H, W, C)

        feats = feats_hwc.permute(0, 3, 1, 2)  # (B, C, H, W)

        return feats


def compute_baseline_multi_scale_PCA(
    backbone: DinoViTBackboneBase,
    pixel_values: torch.Tensor,
    pca_hidden_states_sizes=[16, 32, 64],
):
    patch_size = backbone.get_patch_size()
    with torch.no_grad():
        # Fit PCA on baseline features from multiple scales
        h_size_to_feats_bhwc = {}

        for h_size in pca_hidden_states_sizes:
            input_size = h_size * patch_size
            pixel_values_in = F.interpolate(
                pixel_values,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            layer_feats_bhwc = backbone._compute_gt_features(
                backbone,
                pixel_values=pixel_values_in,
                layer_indices=[12],
                flatten_hw_to_seq=False,
            )
            last_feats_bhwc = layer_feats_bhwc[0]  # (B, H*W, C)
            h_size_to_feats_bhwc[h_size] = last_feats_bhwc

        baseline_fit_tokens = torch.cat(
            [x.reshape(-1, x.shape[-1]) for x in h_size_to_feats_bhwc.values()], dim=0
        )
        pca_feats, pca_mean, pca_std, _, pca_eig = pca_utils.pca(
            baseline_fit_tokens, k=3, std_normalize=True
        )

        pca_rgb, cmin, cmax = pca_utils.tensor_to_rgb(pca_feats)

    return {
        "pca_mean": pca_mean,
        "pca_std": pca_std,
        "pca_eig": pca_eig,
        "pca_color_min": cmin,
        "pca_color_max": cmax,
        "h_size_to_feats_bhwc": h_size_to_feats_bhwc,
    }


def apply_pca(
    hidden_states_bhwc: torch.Tensor,
    pca_data: dict,
) -> torch.Tensor:
    # hidden_states_bhwc: (B, H, W, C)
    B, H, W, C = hidden_states_bhwc.shape
    hidden_states_flat = hidden_states_bhwc.reshape(-1, C)  # (B*H*W, C)
    hidden_states_pca = pca_utils.apply_pca(
        hidden_states_flat,
        pca_data["pca_eig"],
        pca_data["pca_mean"],
        pca_data["pca_std"],
    )  # (B*H*W, 3)
    hidden_states_pca_color, _, _ = pca_utils.tensor_to_rgb(
        hidden_states_pca, pca_data["pca_color_min"], pca_data["pca_color_max"]
    )  # (B*H*W, 3)

    hidden_states_pca_reshaped = hidden_states_pca.reshape(B, H, W, 3)  # (B, H, W, 3)
    hidden_states_pca_color = hidden_states_pca_color.reshape(
        B, H, W, 3
    )  # (B, H, W, 3)
    return {
        "pca_feats": hidden_states_pca_reshaped,
        "pca_color": hidden_states_pca_color,
    }
