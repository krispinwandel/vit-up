from typing import Any, Dict, List, Optional, Tuple, cast
from abc import abstractmethod
import torch
from PIL import Image

import torch.nn as nn
from lightning import LightningModule
from vit_up.training.lightning_module import ViTUpPL

from vit_up.utils import pca_utils, pil_img_utils
from vit_up.layers.backbones.dino_vit_base import DinoViTBackboneBase
from ...inference import inference_wrappers
from ..utils.layout import (
    add_cell_desc,
    make_row_image,
)
from ..utils.img_ops import (
    to_pil_rgb,
    overlay_imgs,
    overlay_points,
)
from ..utils.colors import (
    numpy_color_palette,
)
from .base import VisHelper
from vit_up.layers.backbones.pe_inv_bias import PEInvBias


class PCAVisHelperBase(VisHelper):

    def __init__(
        self,
        gt_img_sizes=[896],
        pca_img_sizes=[224, 448, 896],
        lr_hidden_layer_img_size: int = 448,
        cell_width: int = 300,
    ):
        super().__init__()

        self.cell_width = cell_width
        self.pca_img_sizes = pca_img_sizes
        self.gt_img_sizes = gt_img_sizes
        self.all_img_sizes = list(set(pca_img_sizes + gt_img_sizes))
        self.lr_hidden_layer_img_size = lr_hidden_layer_img_size

        self.backbone_layer_features_bhwc_by_img_in_size = None
        self.layer_pca_and_rgb_stats = None
        self.gt_backbone_layer_features_by_img_in_size = None
        self.gt_layer_pca_features_by_img_in_size = None
        self.gt_layer_rgb_by_img_in_size = {}

    def _compute_layer_pca_and_rgb_stats(
        self,
        backbone_layer_features_bhwc_by_img_in_size: Dict[int, List[torch.Tensor]],
        scales_to_use: Optional[List[int]] = None,
        n_components: int = 3,
    ) -> List[
        List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]
    ]:
        """
        Returns:
            layer_pca_and_rgb_stats: (n_layer, batch_size, 5) in list form
        """
        if scales_to_use is None:
            scales_to_use = list(backbone_layer_features_bhwc_by_img_in_size.keys())
        b = backbone_layer_features_bhwc_by_img_in_size[scales_to_use[0]][0].shape[0]
        n_layers = len(backbone_layer_features_bhwc_by_img_in_size[scales_to_use[0]])
        layer_pca_and_rgb_stats = []
        for layer_idx in range(n_layers):
            per_layer_stats = []
            for bidx in range(b):
                fts_flat = []
                for img_size in scales_to_use:
                    fts_bhwc = backbone_layer_features_bhwc_by_img_in_size[img_size][
                        layer_idx
                    ][bidx]
                    fts_flat.append(fts_bhwc.reshape(-1, fts_bhwc.shape[-1]))
                all_fts_flat = torch.cat(fts_flat, dim=0)
                pca_all, pca_mean, pca_std, _, pca_eigvec = pca_utils.pca(
                    all_fts_flat,
                    k=n_components,
                    std_normalize=True,
                )
                _, rgb_min_val, rgb_max_val = pca_utils.tensor_to_rgb(pca_all)
                per_layer_stats.append(
                    (pca_mean, pca_std, pca_eigvec, rgb_min_val, rgb_max_val)
                )
            layer_pca_and_rgb_stats.append(per_layer_stats)
        return layer_pca_and_rgb_stats

    def _project_features_to_pca_space(
        self,
        layer_features: List[torch.Tensor],
        layer_pca_and_rgb_stats: List[
            List[
                Tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
                ]
            ]
        ],
    ) -> List[torch.Tensor]:
        """
        Returns:
            layer_pca_features: List of (B, out_res, out_res, n_components) tensors for each layer
        """
        assert len(layer_features[0]) == len(
            layer_pca_and_rgb_stats[0]
        ), "Batch size of pred_features must match that of pca_stats"
        out_res = layer_features[0].shape[1]
        layer_pca_features = []
        n_layers = len(layer_features)
        for l in range(n_layers):
            pred_pca_out_layer = []
            for bidx in range(layer_features[0].shape[0]):
                pca_mean, pca_std, pca_eigvec, _, _ = layer_pca_and_rgb_stats[l][bidx]
                pred_pca = pca_utils.apply_pca(
                    X_new=layer_features[l][bidx].flatten(0, 1),
                    original_mean=pca_mean,
                    original_std=pca_std,
                    pca_eigenvectors=pca_eigvec,
                )
                pred_pca = pred_pca.reshape(out_res, out_res, -1)
                pred_pca_out_layer.append(pred_pca)
            pred_pca_out_layer = torch.stack(
                pred_pca_out_layer, dim=0
            )  # (B, out_res, out_res, n_components)
            layer_pca_features.append(pred_pca_out_layer)
        return layer_pca_features

    def _build_pca_row_image(
        self,
        rgb_tensor: torch.Tensor,
        label: str,
        cell_width: int,
        desc: str = "layer=last",
    ) -> Image.Image:
        cell = add_cell_desc(
            to_pil_rgb(rgb_tensor), desc=desc, textbox_width=self.cell_width
        )
        return make_row_image([cell], label=label, cell_width=cell_width)

    def _layer_pca_features_to_color(self, layer_pca_features, layer_pca_and_rgb_stats):
        assert len(layer_pca_features[0]) == len(
            layer_pca_and_rgb_stats[0]
        ), "Batch size of pred_pca must match that of base_rgb_stats"
        n_layer = len(layer_pca_and_rgb_stats)
        layer_rgb_out = []
        for l in range(n_layer):
            pred_rgb_out = []
            for bidx in range(len(layer_pca_features[0])):
                _, _, _, rgb_min_val, rgb_max_val = layer_pca_and_rgb_stats[l][bidx]
                pred_rgb, _, _ = pca_utils.tensor_to_rgb(
                    layer_pca_features[l][bidx].flatten(0, 1),
                    tensor_min=rgb_min_val,
                    tensor_max=rgb_max_val,
                )
                pred_rgb_out.append(
                    pred_rgb.reshape(
                        layer_pca_features[l][bidx].shape[0],
                        layer_pca_features[l][bidx].shape[1],
                        -1,
                    )
                )
            layer_rgb_out.append(torch.stack(pred_rgb_out, dim=0))
        return layer_rgb_out

    @abstractmethod
    def get_layer_indices(self, pl_module: LightningModule) -> List[int]:
        """Get the list of layer indices to visualize from the pl_module."""
        raise NotImplementedError

    @abstractmethod
    def get_backbone(self, pl_module: LightningModule) -> DinoViTBackboneBase:
        """Get the backbone model from the pl_module."""
        raise NotImplementedError

    def set_input_images(self, imgs: List[Image.Image], pl_module: LightningModule):
        backbone = self.get_backbone(pl_module)
        layer_indices = self.get_layer_indices(pl_module)
        self.imgs = [img.copy() for img in imgs]
        pixel_values = self._transform_images(imgs=imgs, device=pl_module.device)
        self.pixel_values = pixel_values
        self.backbone_layer_features_bhwc_by_img_in_size = (
            inference_wrappers.compute_backbone_layer_features_bhwc_by_scale(
                layer_indices=layer_indices,
                pixel_values=pixel_values,
                base_img_in_sizes=self.all_img_sizes,
                backbone=backbone,
            )
        )
        self.layer_pca_and_rgb_stats = self._compute_layer_pca_and_rgb_stats(
            backbone_layer_features_bhwc_by_img_in_size=self.backbone_layer_features_bhwc_by_img_in_size,
            scales_to_use=self.pca_img_sizes,
            n_components=3,
        )
        assert len(self.layer_pca_and_rgb_stats[0]) == pixel_values.shape[0]

        self.gt_backbone_layer_features_by_img_in_size = {
            img_size: self.backbone_layer_features_bhwc_by_img_in_size[img_size]
            for img_size in self.gt_img_sizes
        }
        self.gt_layer_pca_features_by_img_in_size = {
            img_size: self._project_features_to_pca_space(
                layer_features=self.gt_backbone_layer_features_by_img_in_size[img_size],
                layer_pca_and_rgb_stats=self.layer_pca_and_rgb_stats,
            )
            for img_size in self.gt_img_sizes
        }
        self.gt_layer_rgb_by_img_in_size = {
            img_size: self._layer_pca_features_to_color(
                layer_pca_features=self.gt_layer_pca_features_by_img_in_size[img_size],
                layer_pca_and_rgb_stats=self.layer_pca_and_rgb_stats,
            )
            for img_size in self.gt_img_sizes
        }


class PCAVisHelper(PCAVisHelperBase):
    def __init__(
        self,
        gt_img_sizes=[896],
        pca_img_sizes=[224, 448, 896],
        pred_out_sizes=[448],
        lr_hidden_layer_img_size: int = 448,
        cell_width: int = 300,
    ):
        super().__init__(
            gt_img_sizes=gt_img_sizes,
            pca_img_sizes=pca_img_sizes,
            lr_hidden_layer_img_size=lr_hidden_layer_img_size,
            cell_width=cell_width,
        )
        self.pred_out_sizes = pred_out_sizes

    def get_layer_indices(self, pl_module: LightningModule):
        pl_module = cast(ViTUpPL, pl_module)
        return pl_module.vit_up_layer_indices

    def get_backbone(self, pl_module: LightningModule):
        pl_module = cast(ViTUpPL, pl_module)
        return pl_module.backbone

    def compute_pred_pca_and_rgb(
        self,
        pixel_values: torch.Tensor,
        vit_up_pl: ViTUpPL,
        pred_out_sizes=None,
        hidden_layer_img_size=None,
        query_chunk_size=None,
    ):
        assert (
            self.layer_pca_and_rgb_stats is not None
        ), "PCA and RGB stats must be initialized by calling init_images() before computing predicted PCA and RGB"
        if pred_out_sizes is None:
            pred_out_sizes = self.pred_out_sizes
        q_xy_normalized_by_out_size = (
            inference_wrappers.compute_query_coords_by_out_res(pred_out_sizes)
        )
        pred_layer_features_by_out_size = (
            inference_wrappers.compute_layer_query_features(
                q_xy_normalized_by_out_size=q_xy_normalized_by_out_size,
                pixel_values=pixel_values,
                vit_up_pl=vit_up_pl,
                hidden_layer_img_size=hidden_layer_img_size,
                query_chunk_size=query_chunk_size,
            )
        )
        pred_layer_pca_by_out_size = {}
        pred_layer_rgb_by_out_size = {}
        for out_size, pred_layer_features in pred_layer_features_by_out_size.items():
            pred_layer_pca = self._project_features_to_pca_space(
                layer_features=pred_layer_features,
                layer_pca_and_rgb_stats=self.layer_pca_and_rgb_stats,
            )
            pred_layer_rgb = self._layer_pca_features_to_color(
                layer_pca_features=pred_layer_pca,
                layer_pca_and_rgb_stats=self.layer_pca_and_rgb_stats,
            )
            pred_layer_pca_by_out_size[out_size] = pred_layer_pca
            pred_layer_rgb_by_out_size[out_size] = pred_layer_rgb
        return pred_layer_pca_by_out_size, pred_layer_rgb_by_out_size

    def generate_vis(self, img_idx: int, pl_module: LightningModule) -> Image.Image:
        pl_module = cast(ViTUpPL, pl_module)
        if self.pixel_values is None or self.imgs is None:
            raise RuntimeError("init_images must be called before generate_image.")

        layer_indices = pl_module.vit_up_layer_indices
        patch_size = pl_module.backbone.get_patch_size()
        n_layer = len(layer_indices)
        # TODO we predict for all images here but only visualize for one image.
        pred_layer_pca_by_out_size, pred_layer_rgb_by_out_size = (
            self.compute_pred_pca_and_rgb(
                pixel_values=self.pixel_values,
                vit_up_pl=pl_module,
                pred_out_sizes=self.pred_out_sizes,
                hidden_layer_img_size=self.lr_hidden_layer_img_size,
                query_chunk_size=None,
            )
        )
        rows: List[Image.Image] = []
        for gt_size in sorted(self.gt_img_sizes):
            if gt_size not in self.gt_layer_rgb_by_img_in_size:
                continue
            input_cell = add_cell_desc(
                self.imgs[img_idx].resize(
                    (gt_size, gt_size), resample=Image.Resampling.BICUBIC
                ),
                desc=f"Input image ({gt_size}x{gt_size})",
                textbox_width=self.cell_width,
            )
            gt_layer_cells = []
            for l in range(n_layer):
                gt_layer_cells.append(
                    add_cell_desc(
                        to_pil_rgb(
                            self.gt_layer_rgb_by_img_in_size[gt_size][l][img_idx]
                        ),
                        desc=f"layer={layer_indices[l]}",
                        textbox_width=self.cell_width,
                        interpolate_resample=Image.Resampling.NEAREST,
                    )
                )
            rows.append(
                make_row_image(
                    [input_cell, *gt_layer_cells],
                    label=f"PCA of hidden states ({gt_size//patch_size}x{gt_size//patch_size})",
                    cell_width=self.cell_width,
                )
            )

        for pred_size in sorted(pred_layer_rgb_by_out_size.keys()):
            pred_layer_rgb = pred_layer_rgb_by_out_size[pred_size]
            input_cell = add_cell_desc(
                self.imgs[img_idx].resize(
                    (
                        self.lr_hidden_layer_img_size,
                        self.lr_hidden_layer_img_size,
                    ),
                    resample=Image.Resampling.BICUBIC,
                ),
                desc=f"Input image ({self.lr_hidden_layer_img_size}x{self.lr_hidden_layer_img_size})",
                textbox_width=self.cell_width,
            )
            pred_layer_cells = []
            for l in range(n_layer):
                pred_layer_cells.append(
                    add_cell_desc(
                        to_pil_rgb(pred_layer_rgb[l][img_idx]),
                        desc=f"layer={layer_indices[l]}",
                        textbox_width=self.cell_width,
                        interpolate_resample=Image.Resampling.NEAREST,
                    )
                )
            rows.append(
                make_row_image(
                    [input_cell, *pred_layer_cells],
                    label=f"PCA of upsampled hidden states ({self.lr_hidden_layer_img_size//patch_size}x{self.lr_hidden_layer_img_size//patch_size} => {pred_size}x{pred_size})",
                )
            )

        if len(rows) == 0:
            return Image.new("RGB", (self.cell_width, self.cell_width), color=(0, 0, 0))

        return pil_img_utils.concat_images(
            rows,
            mode="col",
            pad=2,
            pad_color=(0, 0, 0),
        )


class PCAVisHelperInvBias(PCAVisHelperBase):
    """Visualize original vs de-biased hidden-state PCA for each gt_img_size.

    Usage: call `set_input_images(imgs, vit_up_pl, pe_inv_bias)` where `vit_up_pl`
    is a `ViTUpPL` (provides backbone and layer indices) and `pe_inv_bias` is an
    instance of `PEInvBias` already initialized/loaded to the backbone.
    """

    def get_layer_indices(self, pl_module: LightningModule) -> List[int]:
        pl_module = cast(LightningModule, pl_module)
        return pl_module.layer_indices

    def get_backbone(self, pl_module: LightningModule) -> DinoViTBackboneBase:
        pl_module = cast(LightningModule, pl_module)
        return pl_module.backbone

    def compute_pred(self, pl_module: LightningModule):
        pl_module = cast(LightningModule, pl_module)
        if self.pixel_values is None:
            raise RuntimeError(
                "pixel_values must be initialized by calling set_input_images() before computing pred features."
            )
        if self.gt_backbone_layer_features_by_img_in_size is None:
            raise RuntimeError(
                "gt_backbone_layer_features_by_img_in_size must be initialized by calling set_input_images() before computing pred features."
            )
        if self.layer_pca_and_rgb_stats is None:
            raise RuntimeError(
                "layer_pca_and_rgb_stats must be initialized by calling set_input_images() before computing pred features."
            )
        # compute de-biased features by applying PEInvBias to original per-layer lists
        pred_backbone_layer_features_debiased_by_img_in_size = {}
        for img_size in self.gt_img_sizes:
            original_layers = self.gt_backbone_layer_features_by_img_in_size[img_size]
            # ensure tensors live on same device as pe_inv_bias buffers
            device = self.pixel_values.device
            original_layers = [t.to(device) for t in original_layers]
            # pass a cloned list to avoid in-place modification of cached originals
            cloned = [t.clone() for t in original_layers]

            # Support two de-biasing APIs:
            # - legacy: pl_module.pe_inv_bias(list[layer_tensors]) -> list
            # - artifact-basis: pl_module.smoother(list[layer_tensors]) -> (list, bases)
            if hasattr(pl_module, "pe_inv_bias"):
                debiased_layers = pl_module.pe_inv_bias(cloned)
            elif hasattr(pl_module, "smoother"):
                # smoother returns (smoothed, bases)
                out = pl_module.smoother(cloned)
                # out[0] is the smoothed list/tensor
                debiased_layers = out[0]
            else:
                # Fallback: try calling forward(pl_module, cloned)
                try:
                    out = pl_module.forward(cloned)
                    if isinstance(out, tuple):
                        debiased_layers = out[0]
                    elif isinstance(out, dict):
                        debiased_layers = out.get("debiased", cloned)
                    else:
                        debiased_layers = out
                except Exception as e:
                    raise RuntimeError(
                        "pl_module does not expose pe_inv_bias or smoother and forward failed"
                    ) from e

            pred_backbone_layer_features_debiased_by_img_in_size[img_size] = (
                debiased_layers
            )
        pred_layer_pca_features_debiased_by_img_in_size = {
            img_size: self._project_features_to_pca_space(
                layer_features=pred_backbone_layer_features_debiased_by_img_in_size[
                    img_size
                ],
                layer_pca_and_rgb_stats=self.layer_pca_and_rgb_stats,
            )
            for img_size in self.gt_img_sizes
        }
        pred_layer_rgb_debiased_by_img_in_size = {
            img_size: self._layer_pca_features_to_color(
                layer_pca_features=pred_layer_pca_features_debiased_by_img_in_size[
                    img_size
                ],
                layer_pca_and_rgb_stats=self.layer_pca_and_rgb_stats,
            )
            for img_size in self.gt_img_sizes
        }
        return pred_layer_rgb_debiased_by_img_in_size

    def generate_vis(self, img_idx: int, pl_module: LightningModule) -> Image.Image:
        pl_module = cast(ViTUpPL, pl_module)
        if self.pixel_values is None or self.imgs is None:
            raise RuntimeError("init_images must be called before generate_image.")
        pred_layer_rgb_debiased_by_img_in_size = self.compute_pred(pl_module=pl_module)

        layer_indices = pl_module.layer_indices
        patch_size = pl_module.backbone.get_patch_size()
        n_layer = len(layer_indices)

        rows: List[Image.Image] = []
        for gt_size in sorted(self.gt_img_sizes):
            if gt_size not in self.gt_layer_rgb_by_img_in_size:
                continue
            input_cell = add_cell_desc(
                self.imgs[img_idx].resize(
                    (gt_size, gt_size), resample=Image.Resampling.BICUBIC
                ),
                desc=f"Input image ({gt_size}x{gt_size})",
                textbox_width=self.cell_width,
            )

            # original PCA cells
            orig_cells = []
            for l in range(n_layer):
                orig_cells.append(
                    add_cell_desc(
                        to_pil_rgb(
                            self.gt_layer_rgb_by_img_in_size[gt_size][l][img_idx]
                        ),
                        desc=f"layer={layer_indices[l]} (orig)",
                        textbox_width=self.cell_width,
                        interpolate_resample=Image.Resampling.NEAREST,
                    )
                )

            # de-biased PCA cells
            deb_cells = []
            for l in range(n_layer):
                deb_cells.append(
                    add_cell_desc(
                        to_pil_rgb(
                            pred_layer_rgb_debiased_by_img_in_size[gt_size][l][img_idx]
                        ),
                        desc=f"layer={layer_indices[l]} (de-bias)",
                        textbox_width=self.cell_width,
                        interpolate_resample=Image.Resampling.NEAREST,
                    )
                )

            # one wide row: input | orig layers... | debiased layers...
            rows.append(
                make_row_image(
                    [input_cell, *orig_cells],
                    label=f"PCA original ({gt_size//patch_size}x{gt_size//patch_size})",
                    cell_width=self.cell_width,
                )
            )
            rows.append(
                make_row_image(
                    [input_cell, *deb_cells],
                    label=f"PCA de-biased ({gt_size//patch_size}x{gt_size//patch_size})",
                    cell_width=self.cell_width,
                )
            )

        if len(rows) == 0:
            return Image.new("RGB", (self.cell_width, self.cell_width), color=(0, 0, 0))

        return pil_img_utils.concat_images(rows, mode="col", pad=2, pad_color=(0, 0, 0))
