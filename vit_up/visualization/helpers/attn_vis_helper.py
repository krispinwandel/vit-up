from typing import List, Optional, Tuple, cast
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lightning import LightningModule
from vit_up.training.lightning_module import ViTUpPL
from ...inference import inference_wrappers
from .base import VisHelper
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


class AttnVisHelper(VisHelper):
    """Self-contained helper for attention map computation and visualization image rendering."""

    def __init__(
        self,
        n_q_per_side: int = 128,
        gt_hidden_layer_img_size: int = 128,
        pred_hidden_layer_img_size: int = 448,
        n_cells_per_side: int = 4,
        query_chunk_size: int = 4096,
        cell_width: int = 300,
    ):
        super().__init__()
        self.cell_width = cell_width
        self.n_q_per_side = n_q_per_side
        self.gt_hidden_layer_img_size = gt_hidden_layer_img_size
        self.n_cells_per_side = n_cells_per_side
        self.pred_hidden_layer_img_size = pred_hidden_layer_img_size
        self.query_chunk_size = query_chunk_size
        self.gt_hidden_states_bhwc: Optional[List[torch.Tensor]] = None

    @staticmethod
    def _compute_layer_attentions(
        q_latent_layers: List[Optional[torch.Tensor]],
        k_layers: List[Optional[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Compute per-layer query-token attention maps with shape (n_q, embd_size, embd_size).
        Args:
            q_latent_layers: list of (n_q, c)
            k_layers: list of (embd_size**2, c)
        Out:
            q_layer_attn_maps: list of (n_q, embd_size, embd_size) attention maps for each layer
        """
        n_layers = len(q_latent_layers)
        embd_size = (
            math.isqrt(k_layers[-1].shape[0]) if k_layers[-1] is not None else None
        )
        n_q = q_latent_layers[-1].shape[0] if q_latent_layers[-1] is not None else None
        if embd_size is None:
            raise RuntimeError(
                "Unable to infer embedding map size from k layers in attention visualization."
            )
        if n_q is None:
            raise RuntimeError(
                "Unable to infer query count from q_latent layers in attention visualization."
            )

        # Layer 0 has no global context; keep a zero map for indexing consistency.
        q_layer_attn_maps = [
            torch.zeros(
                n_q,
                embd_size,
                embd_size,
                dtype=torch.float32,
            )
            for l in range(n_layers)
        ]

        for layer_idx in range(1, n_layers):
            q_tokens = q_latent_layers[layer_idx]
            k_tokens = k_layers[layer_idx]
            if q_tokens is None or k_tokens is None:
                print(f"Layer {layer_idx}: One or more attention tensors are None.")
                continue

            attn = torch.matmul(
                F.normalize(q_tokens.float(), dim=-1),
                F.normalize(k_tokens.float(), dim=-1).transpose(0, 1),
            )  # (n_q, h*w)

            attn_2d = attn.reshape(n_q, embd_size, embd_size).cpu()
            q_layer_attn_maps[layer_idx][:n_q] = attn_2d

        return q_layer_attn_maps

    @staticmethod
    def _select_grid_max_error_q_indices(
        q_features_last_layer: torch.Tensor,
        q_xy_normalized: torch.Tensor,
        backbone_last_hidden_states_hwc: torch.Tensor,
        n_cells_per_side: int,
    ) -> List[int]:
        """Select one query index per grid cell by maximum feature error vs sampled GT feature."""
        cell_size = 1.0 / n_cells_per_side
        q_indices = torch.arange(
            q_features_last_layer.shape[0], device=q_features_last_layer.device
        )

        cell_centers = (
            torch.linspace(
                0.5,
                n_cells_per_side - 0.5,
                n_cells_per_side,
                device=q_features_last_layer.device,
            )
            * cell_size
        )
        grid_y, grid_x = torch.meshgrid(cell_centers, cell_centers, indexing="ij")
        cells_xy = torch.stack((grid_x, grid_y), dim=-1).reshape(
            -1, 2
        )  # (n_cells, 2), x/y in [0,1]
        cells_xy_grid = cells_xy * 2.0 - 1.0  # grid_sample expects [-1, 1]

        sampled = F.grid_sample(
            backbone_last_hidden_states_hwc.permute(2, 0, 1).unsqueeze(
                0
            ),  # (1, c, h, w)
            cells_xy_grid.unsqueeze(0).unsqueeze(0),  # (1, 1, n_cells, 2)
            align_corners=False,
        )  # (1, c, 1, n_cells)
        cell_fts = (
            sampled.squeeze(0)
            .squeeze(1)
            .transpose(0, 1)
            .reshape(n_cells_per_side, n_cells_per_side, -1)
        )  # (n_cells_per_side, n_cells_per_side, c)

        selected_q_indices: List[int] = []
        for i in range(n_cells_per_side):
            for j in range(n_cells_per_side):
                q_in_cell = (
                    (q_xy_normalized[:, 0] >= j * cell_size)
                    & (q_xy_normalized[:, 0] < (j + 1) * cell_size)
                    & (q_xy_normalized[:, 1] >= i * cell_size)
                    & (q_xy_normalized[:, 1] < (i + 1) * cell_size)
                )
                cell_q_indices = q_indices[q_in_cell]

                if int(cell_q_indices.numel()) == 0:
                    # Fallback for sparse or non-divisible coordinate grids.
                    center = torch.tensor(
                        [(j + 0.5) * cell_size, (i + 0.5) * cell_size],
                        device=q_xy_normalized.device,
                        dtype=q_xy_normalized.dtype,
                    )
                    dist = torch.linalg.norm(
                        q_xy_normalized - center.unsqueeze(0), dim=-1
                    )
                    selected_q_indices.append(int(torch.argmin(dist).item()))
                    continue

                cell_q_fts = q_features_last_layer[q_in_cell]
                cell_gt_ft = cell_fts[i, j].unsqueeze(0)
                cell_err = torch.linalg.norm(
                    cell_q_fts.float() - cell_gt_ft.float(), dim=-1
                )
                max_err_idx_in_cell = int(torch.argmax(cell_err).item())
                selected_q_idx = int(cell_q_indices[max_err_idx_in_cell].item())
                selected_q_indices.append(selected_q_idx)

        return selected_q_indices

    @staticmethod
    def _build_colored_attention_images(
        q_layer_attn_maps: List[torch.Tensor],
        n_cells_per_side: int,
    ) -> Tuple[List[Image.Image], List[np.ndarray]]:
        """Render one RGB attention image per layer and return query point colors."""
        n_q, h_embd, w_embd = q_layer_attn_maps[0].shape[:3]
        assert (
            h_embd % n_cells_per_side == 0 and w_embd % n_cells_per_side == 0
        ), "Embedding map size must be divisible by n_cells_per_side."
        assert (
            n_q == n_cells_per_side * n_cells_per_side
        ), "Expected exactly one selected query per grid cell."

        n_tokens_per_cell_h = h_embd // n_cells_per_side
        n_tokens_per_cell_w = w_embd // n_cells_per_side
        max_err_attn_imgs: List[Image.Image] = []

        palette = numpy_color_palette(n_colors=n_q)
        q_colors = [np.round(color * 255.0).astype(np.uint8) for color in palette]

        for layer_attn in q_layer_attn_maps:
            attn_img_zero = np.zeros((h_embd, w_embd, 3), dtype=np.float32)
            for i in range(n_q):
                cell_i = i // n_cells_per_side
                cell_j = i % n_cells_per_side
                attn_color = palette[i]
                h_slice = slice(
                    cell_i * n_tokens_per_cell_h, (cell_i + 1) * n_tokens_per_cell_h
                )
                w_slice = slice(
                    cell_j * n_tokens_per_cell_w, (cell_j + 1) * n_tokens_per_cell_w
                )
                cell_attn = layer_attn[i][h_slice, w_slice]
                attn_img_zero[h_slice, w_slice] = (
                    (cell_attn - cell_attn.min())
                    / (cell_attn.max() - cell_attn.min() + 1e-8)
                )[..., None] * attn_color

            attn_img_pil = Image.fromarray((attn_img_zero * 255).astype(np.uint8))
            max_err_attn_imgs.append(attn_img_pil)

        return max_err_attn_imgs, q_colors

    def set_input_images(self, imgs: List[Image.Image], pl_module: LightningModule):
        pl_module = cast(ViTUpPL, pl_module)
        self.imgs = [img.copy() for img in imgs]
        pixel_values = self._transform_images(imgs=imgs, device=pl_module.device)
        self.pixel_values = pixel_values
        self.gt_hidden_states_bhwc = (
            inference_wrappers.compute_backbone_layer_features_bhwc_by_scale(
                layer_indices=pl_module.vit_up_layer_indices,
                pixel_values=pixel_values,
                base_img_in_sizes=[self.gt_hidden_layer_img_size],
                backbone=pl_module.backbone,
                use_lora=False,
            )
        )[self.gt_hidden_layer_img_size]

    def _build_attention_visualization_data(
        self,
        img_idx: int,
        vit_up_pl: ViTUpPL,
    ) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor]:
        """Attention-map responsibility: compute selected queries, attentions, and colorized maps."""
        if self.pixel_values is None or self.gt_hidden_states_bhwc is None:
            raise RuntimeError("init_images must be called before generate_image.")

        pixel_values = self.pixel_values[img_idx]
        q_xy_normalized = (
            inference_wrappers.compute_query_coords_by_out_res([self.n_q_per_side])
        )[self.n_q_per_side]
        q_xy_normalized_flat = q_xy_normalized.view(-1, 2)  # (n_q, 2)

        vis_meta = inference_wrappers.compute_layer_query_features_and_meta(
            q_xy_normalized_by_out_size={self.n_q_per_side: q_xy_normalized},
            pixel_values=pixel_values,
            vit_up_pl=vit_up_pl,
            hidden_layer_img_size=self.pred_hidden_layer_img_size,
        )[self.n_q_per_side]

        q_layer_fts = cast(List[torch.Tensor], vis_meta["q_layer_features"])
        q_layer_fts_flat = [ql.flatten(0, 1) for ql in q_layer_fts]  # (n_q, c)

        hidden_states_hwc = [
            layer_states[img_idx] for layer_states in self.gt_hidden_states_bhwc
        ]
        q_indices = self._select_grid_max_error_q_indices(
            q_features_last_layer=q_layer_fts_flat[-1],
            q_xy_normalized=q_xy_normalized_flat,
            backbone_last_hidden_states_hwc=hidden_states_hwc[-1],
            n_cells_per_side=self.n_cells_per_side,
        )

        q_xy_normalized_selected = q_xy_normalized_flat[q_indices]

        q_latent_layers = cast(List[Optional[torch.Tensor]], vis_meta["q_latent"])
        k_layers = cast(List[Optional[torch.Tensor]], vis_meta["k"])
        v_layers = cast(List[Optional[torch.Tensor]], vis_meta["v"])

        q_latent_layers_selected: List[Optional[torch.Tensor]] = []
        for q_latent in q_latent_layers:
            if q_latent is None:
                q_latent_layers_selected.append(None)
            elif q_latent.ndim == 2:
                q_latent_layers_selected.append(q_latent[q_indices])
            elif q_latent.ndim == 3:
                q_latent_layers_selected.append(q_latent[:, q_indices])
            else:
                q_latent_layers_selected.append(None)

        q_layer_attn_maps = self._compute_layer_attentions(
            q_latent_layers=q_latent_layers_selected, k_layers=k_layers
        )
        max_err_attn_imgs, q_colors = self._build_colored_attention_images(
            q_layer_attn_maps=q_layer_attn_maps,
            n_cells_per_side=self.n_cells_per_side,
        )
        q_colors_tensor = torch.tensor(np.stack(q_colors), dtype=torch.float32)
        return max_err_attn_imgs, q_xy_normalized_selected, q_colors_tensor

    def _compose_visualization_image(
        self,
        img_idx: int,
        max_err_attn_imgs: List[Image.Image],
        q_xy_normalized_selected: torch.Tensor,
        q_colors: torch.Tensor,
        layer_indices: List[int],
    ) -> Image.Image:
        """Image responsibility: build final visualization mosaic from precomputed attention artifacts."""
        if self.imgs is None:
            raise RuntimeError("init_images must be called before generate_image.")

        blended_imgs = []
        for i, attn_img in enumerate(max_err_attn_imgs):
            blended_imgs.append(
                overlay_imgs(
                    self.imgs[img_idx],
                    attn_img,
                    alpha=1.0 if i == 0 else 0.55,
                )
            )

        query_point_img = overlay_points(
            base_img=self.imgs[img_idx].resize(
                (2 * self.cell_width, 2 * self.cell_width),
                resample=Image.Resampling.BICUBIC,
            ),
            points_xz_normalized=q_xy_normalized_selected,
            point_colors=q_colors,
        )

        row_imgs = [
            add_cell_desc(
                query_point_img,
                desc="query points",
                textbox_width=self.cell_width,
            )
        ] + [
            add_cell_desc(
                blended,
                desc=(
                    f"layer={layer_indices[i]}"
                    if i > 0
                    else f"layer={layer_indices[i]}, no cross-attention"
                ),
                textbox_width=self.cell_width,
            )
            for i, blended in enumerate(blended_imgs)
        ]

        return make_row_image(
            row_imgs,
            label="Cross-attention between query points and hidden states",
        )

    def generate_vis(
        self,
        img_idx: int,
        pl_module: LightningModule,
    ) -> Image.Image:
        pl_module = cast(ViTUpPL, pl_module)
        max_err_attn_imgs, q_xy_selected, q_colors = (
            self._build_attention_visualization_data(
                img_idx=img_idx,
                vit_up_pl=pl_module,
            )
        )
        return self._compose_visualization_image(
            img_idx=img_idx,
            max_err_attn_imgs=max_err_attn_imgs,
            q_xy_normalized_selected=q_xy_selected,
            q_colors=q_colors,
            layer_indices=pl_module.vit_up_layer_indices,
        )
