from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
from peft import LoraConfig
import torch
import torch.nn as nn

from ..layers.backbones.dinov2_vit import DINOv2ViT
from ..layers.backbones.dinov3_vit import DINOv3ViT

from ..model.vit_up import ViTUp
from ..supervision.strategies.base import SupervisionStrategy
from ..layers.layer_init_utils import normalize_class_init

# ==================================
# utils
# ==================================


def _normalize_underscored_class_init(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a dict with _class_path/_init_args.")

    class_path = raw.get("_class_path")
    if not isinstance(class_path, str) or not class_path:
        raise ValueError(f"{name}._class_path must be a non-empty string.")

    init_args = raw.get("_init_args", {})
    if init_args is None:
        init_args = {}
    if not isinstance(init_args, dict):
        raise TypeError(f"{name}._init_args must be a dict when provided.")

    return {
        "class_path": class_path,
        "init_args": dict(init_args),
    }


def query_bbox_from_grid(
    query_xy_grid: torch.Tensor,
    query_size_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if query_xy_grid.ndim != 4 or query_xy_grid.shape[-1] != 2:
        raise ValueError(
            "query_xy_grid must have shape (B,H,W,2). "
            f"Got shape={tuple(query_xy_grid.shape)}."
        )

    if query_size_grid is None:
        x = query_xy_grid[..., 0]
        y = query_xy_grid[..., 1]
        x1 = x.amin(dim=(1, 2))
        y1 = y.amin(dim=(1, 2))
        x2 = x.amax(dim=(1, 2))
        y2 = y.amax(dim=(1, 2))
        return torch.stack((x1, y1, x2, y2), dim=-1)

    if query_size_grid.ndim == 4 and query_size_grid.shape[-1] == 1:
        query_size_grid = query_size_grid[..., 0]
    if query_size_grid.ndim != 3:
        raise ValueError(
            "query_size_grid must have shape (B,H,W) or (B,H,W,1). "
            f"Got shape={tuple(query_size_grid.shape)}."
        )
    if query_size_grid.shape != query_xy_grid.shape[:3]:
        raise ValueError(
            "query_size_grid shape must match query_xy_grid spatial shape. "
            f"Got query_xy_grid={tuple(query_xy_grid.shape)} and "
            f"query_size_grid={tuple(query_size_grid.shape)}."
        )

    half = 0.5 * query_size_grid
    x1 = (query_xy_grid[..., 0] - half).amin(dim=(1, 2)).clamp(0.0, 1.0)
    y1 = (query_xy_grid[..., 1] - half).amin(dim=(1, 2)).clamp(0.0, 1.0)
    x2 = (query_xy_grid[..., 0] + half).amax(dim=(1, 2)).clamp(0.0, 1.0)
    y2 = (query_xy_grid[..., 1] + half).amax(dim=(1, 2)).clamp(0.0, 1.0)
    return torch.stack((x1, y1, x2, y2), dim=-1)


def merge_chunk_metrics(
    acc: Optional[Dict[str, torch.Tensor]],
    chunk_metrics: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if acc is None:
        return {k: v.detach() for k, v in chunk_metrics.items()}
    for key, value in chunk_metrics.items():
        if key in acc:
            acc[key] = acc[key] + value.detach()
        else:
            acc[key] = value.detach()
    return acc


def normalize_chunk_metrics(
    metrics: Dict[str, torch.Tensor], n_chunks: int
) -> Dict[str, torch.Tensor]:
    return {k: v / float(n_chunks) for k, v in metrics.items()}


def load_checkpoint_if_present(model: nn.Module, ckpt_path: Optional[str]) -> None:
    if ckpt_path is None:
        return
    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.exists():
        print(f"warning: ckpt_path not found, skipping load: {ckpt_path}")
        return

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        raw_state_dict = cast(Dict[str, torch.Tensor], checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        raw_state_dict = cast(Dict[str, torch.Tensor], checkpoint)
    else:
        raise TypeError(
            "Checkpoint must be a dict or Lightning checkpoint containing a dict 'state_dict'."
        )

    model_state_keys = set(model.state_dict().keys())
    candidate_prefixes = (
        "",
        "model.",
        "module.",
        "vit_up_pl.",
        "vit_up_pl_v2.",
    )

    selected_state_dict: Dict[str, torch.Tensor] = {}
    selected_prefix = ""
    for prefix in candidate_prefixes:
        matched = {
            key[len(prefix) :]: value
            for key, value in raw_state_dict.items()
            if key.startswith(prefix) and key[len(prefix) :] in model_state_keys
        }
        if len(matched) > len(selected_state_dict):
            selected_state_dict = matched
            selected_prefix = prefix

    if not selected_state_dict:
        raise ValueError(
            "No matching model weights found in checkpoint for ViTUpPL. "
            f"Tried prefixes: {candidate_prefixes}"
        )

    missing_keys, unexpected_keys = model.load_state_dict(
        selected_state_dict,
        strict=False,
    )
    if missing_keys or unexpected_keys:
        print(
            "warning: checkpoint load mismatch for ViTUpPL from "
            f"'{ckpt_path}' with prefix '{selected_prefix}': "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )


@dataclass(frozen=True)
class LRSchedulerConfig:
    scheduler: Dict[str, Any]
    interval: str = "step"
    frequency: int = 1
    monitor: Optional[str] = None
    name: Optional[str] = None
    strict: Optional[bool] = None

    def __post_init__(self) -> None:
        scheduler_spec = normalize_class_init(
            _normalize_underscored_class_init(
                self.scheduler, name="lr_scheduler.scheduler"
            ),
            name="lr_scheduler.scheduler",
        )
        object.__setattr__(self, "scheduler", scheduler_spec)

        if self.interval not in ("step", "epoch"):
            raise ValueError("lr_scheduler interval must be 'step' or 'epoch'.")
        if self.frequency < 1:
            raise ValueError("lr_scheduler frequency must be >= 1.")

    @classmethod
    def from_raw(cls, lr_scheduler: Any) -> Optional["LRSchedulerConfig"]:
        if lr_scheduler is None:
            return None
        if isinstance(lr_scheduler, cls):
            return lr_scheduler
        if isinstance(lr_scheduler, dict):
            return cls(**lr_scheduler)
        if hasattr(lr_scheduler, "__dict__"):
            return cls(**dict(vars(lr_scheduler)))
        raise TypeError(
            "lr_scheduler must be a LRSchedulerConfig or dict-like object when provided."
        )

    def to_lightning_config(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        scheduler = _instantiate_lr_scheduler(
            optimizer=optimizer,
            scheduler_spec=self.scheduler,
        )
        cfg: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": self.interval,
            "frequency": self.frequency,
        }
        if self.monitor is not None:
            cfg["monitor"] = self.monitor
        if self.name is not None:
            cfg["name"] = self.name
        if self.strict is not None:
            cfg["strict"] = self.strict
        return cfg


def _instantiate_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_spec: Dict[str, Any],
) -> Any:
    class_path = scheduler_spec.get("class_path")
    init_args = dict(scheduler_spec.get("init_args", {}) or {})
    init_args.pop("optimizer", None)

    if class_path == "torch.optim.lr_scheduler.SequentialLR":
        schedulers_spec = init_args.get("schedulers")
        milestones = init_args.get("milestones")
        if not isinstance(schedulers_spec, list) or milestones is None:
            raise ValueError(
                "SequentialLR requires init_args.schedulers (list) and init_args.milestones."
            )
        schedulers = []
        for idx, spec in enumerate(schedulers_spec):
            child_name = f"lr_scheduler.scheduler.init_args.schedulers[{idx}]"
            child_spec = normalize_class_init(
                _normalize_underscored_class_init(spec, name=child_name),
                name=child_name,
            )
            child_init_args = dict(child_spec.get("init_args", {}) or {})
            child_init_args.pop("optimizer", None)
            child_spec = {
                "class_path": child_spec["class_path"],
                "init_args": child_init_args,
            }
            schedulers.append(instantiate_class(args=(optimizer,), init=child_spec))
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )

    return instantiate_class(
        args=(optimizer,),
        init={"class_path": class_path, "init_args": init_args},
    )


# ==================================
# main module
# ==================================


class ViTUpPL(LightningModule):
    @staticmethod
    def _resolve_backbone_class(backbone_model_name: str):
        model_name = str(backbone_model_name).strip().lower()
        if "dinov3" in model_name:
            return DINOv3ViT
        if "dinov2" in model_name:
            return DINOv2ViT
        raise ValueError(
            "Unable to infer backbone family from backbone_model_name. "
            "Expected model name to include 'dinov2' or 'dinov3'. "
            f"Got: {backbone_model_name}"
        )

    def __init__(
        self,
        optimizer: Dict[str, Any],
        vit_up: ViTUp,
        supervision_strategy: SupervisionStrategy,
        loss_fn: nn.Module,
        lr_img_size: int,
        lr_scheduler: Optional[LRSchedulerConfig] = None,
        backbone_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        window_size: int = 0,
        ckpt_path: Optional[str] = None,
        backbone_lora_config: Optional[Dict[str, Any]] = None,
        compile_model: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.lr_img_size = lr_img_size
        self.window_size = window_size

        # backbone
        lora_config = None
        if backbone_lora_config is not None:
            lora_kwargs = dict(backbone_lora_config)
            lora_config = LoraConfig(**lora_kwargs)
        # remember whether backbone was instantiated with LoRA
        self._backbone_uses_lora = lora_config is not None
        self.backbone_class = self._resolve_backbone_class(backbone_model_name)
        self.backbone = self.backbone_class.init_from_hf(
            backbone_model_name=backbone_model_name,
            backbone_lora_config=lora_config,
            freeze_weights=True,
        )
        # self.backbone.compile()
        self.backbone_patch_size = self.backbone.get_patch_size()

        # vit-up
        self.vit_up: ViTUp = vit_up
        self.vit_up_layer_indices = self.vit_up.layer_indices
        if compile_model and hasattr(self.vit_up, "compile"):
            self.vit_up.compile()

        # supervision strategy
        self.supervision_strategy = supervision_strategy

        # loss
        self.loss_fn = loss_fn

        # optimizer
        self.optimizer_init = normalize_class_init(
            _normalize_underscored_class_init(optimizer, name="optimizer"),
            name="optimizer",
        )
        self.lr_scheduler_cfg = LRSchedulerConfig.from_raw(
            lr_scheduler=cast(Any, lr_scheduler),
        )

        # load checkpoint if provided
        load_checkpoint_if_present(self, ckpt_path)

    def forward(
        self,
        pixel_values: torch.Tensor,
        q_xy_normalized: torch.Tensor,
        hidden_layer_img_size: Optional[int] = None,
        cache_data: Any = None,
        query_chunk_size: Optional[int] = None,
        return_all_layers: bool = False,
    ) -> Tuple[torch.Tensor | List[torch.Tensor], Any]:
        """
        Args:
            pixel_values: (B, C, H, W) tensor of input images.
            q_xy_normalized: (B, N_q, 2) tensor of normalized query
                coordinates in [0, 1], where N_q is the number of query points.
            hidden_layer_img_size: Optional int specifying the image size to use
                when computing hidden states from the backbone. If None, uses the original image size.
            cache_data: Optional dict containing precomputed cache data for the backbone and vit-up. If provided, this data will be used instead of recomputing it.
            query_chunk_size: Optional int specifying the number of query points to process in each chunk.
                If None, processes all query points in a single chunk. Can be used to reduce memory usage at the cost of speed.
        Returns:
            Tuple:
                q_fts: (B, N_q, D) tensor of query features output by ViTUp, where D is the feature dimension.
                cache_data: dict containing cache data that can be reused in subsequent forward passes with the same input images to save computation.
        """
        cache_data = self.vit_up.maybe_compute_cache_data(
            pixel_values=pixel_values,
            backbone=cast(Any, self.backbone),
            hidden_layer_img_size=hidden_layer_img_size,
            cache_data=cache_data,
        )

        if query_chunk_size is None:
            query_chunk_size = q_xy_normalized.shape[1]

        q_chunks = []
        for q_start in range(0, q_xy_normalized.shape[1], query_chunk_size):
            q_end = min(q_start + query_chunk_size, q_xy_normalized.shape[1])
            q_xy_normalized_chunk = q_xy_normalized[:, q_start:q_end, :]
            q_layers_chunk = self.vit_up(
                pixel_values=pixel_values,
                q_xy_normalized=q_xy_normalized_chunk,
                cache_data=cache_data,
            )
            if not return_all_layers:
                q_layers_chunk = q_layers_chunk[-1]
            q_chunks.append(q_layers_chunk)
        if return_all_layers:
            q_chunks = [
                torch.cat([chunk[layer_idx] for chunk in q_chunks], dim=1)
                for layer_idx in range(len(q_chunks[0]))
            ]
            return q_chunks, cache_data
        return torch.cat(q_chunks, dim=1), cache_data

    def _log_losses(self, prefix, loss_dict, on_step, on_epoch):
        for key, value in loss_dict.items():
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
            )

    def _log_learning_rate(self):
        if not self.trainer or not self.trainer.optimizers:
            return
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=True)

    def _prepare_supervision_data(self, batch: Dict[str, Any]) -> Tuple[
        List[torch.Tensor],
        List[Dict[int, torch.Tensor]],
        Dict[int, List[torch.Tensor]],
        Any,
    ]:
        # get supervision blocks from strategy
        # NOTE do not use query_xy if use_full_image
        if not self.supervision_strategy.use_full_image:
            bbox_x1y2x2y2 = query_bbox_from_grid(
                batch["query_xy"],
                batch.get("query_size"),
            )
        else:
            device = batch["pixel_values"].device
            bbox_x1y2x2y2 = torch.tensor(
                [[0.0, 0.0, 1.0, 1.0]],
                device=device,
            ).expand(batch["pixel_values"].shape[0], -1)
        q_xy_blocks = self.supervision_strategy.get_transformed_q_xy_blocks(
            bbox_x1y2x2y2=bbox_x1y2x2y2
        )
        hidden_ij_blocks = self.supervision_strategy.get_hidden_ij_blocks()

        # chunk blocks
        block_chunk_size = self.supervision_strategy.block_chunk_size
        n_total_blocks = q_xy_blocks.shape[1]
        q_xy_block_chunks: List[torch.Tensor] = []
        hidden_ij_blocks_chunks: List[Dict[int, torch.Tensor]] = []
        for block_start in range(0, n_total_blocks, block_chunk_size):
            block_end = min(block_start + block_chunk_size, n_total_blocks)
            q_xy_block_chunk = q_xy_blocks[:, block_start:block_end, :, :, :]
            hidden_ij_blocks_chunk = {
                scale: ij_blocks[block_start:block_end, :, :, :]
                for scale, ij_blocks in hidden_ij_blocks.items()
            }
            q_xy_block_chunks.append(q_xy_block_chunk)
            hidden_ij_blocks_chunks.append(hidden_ij_blocks_chunk)

        gt_img_sizes = self.supervision_strategy.gt_img_sizes()
        gt_layers_flat_by_scale: Dict[int, List[torch.Tensor]] = {}
        for scale, gt_img_size in gt_img_sizes.items():
            gt_layers_flat_by_scale[scale] = self.backbone_class._compute_gt_features(
                backbone=self.backbone,
                pixel_values=batch["pixel_values"],
                layer_indices=self.vit_up_layer_indices,
                img_size=gt_img_size,
                window_size=self.window_size,
            )

        gt_layers_hwc_by_scale: Dict[int, List[torch.Tensor]] = {}
        for scale, gt_layers_flat in gt_layers_flat_by_scale.items():
            gt_embd_size = gt_img_sizes[scale] // self.backbone_patch_size
            gt_layers_hwc_by_scale[scale] = [
                layer.view(layer.shape[0], gt_embd_size, gt_embd_size, layer.shape[-1])
                for layer in gt_layers_flat
            ]

        pred_pixel_values = (
            batch["pixel_values"]
            if self.supervision_strategy.use_full_image
            else batch["pixel_values_container"]
        )

        pred_cache_data = self.vit_up.compute_cache_data(
            pixel_values=pred_pixel_values,
            backbone=cast(Any, self.backbone),
            hidden_layer_img_size=self.lr_img_size,
        )

        return (
            q_xy_block_chunks,
            hidden_ij_blocks_chunks,
            gt_layers_hwc_by_scale,
            pred_cache_data,
        )

    def _compute_chunk_loss_dict(
        self,
        batch: Dict[str, Any],
        q_xy_block_chunk: torch.Tensor,
        hidden_ij_blocks_chunk: Dict[int, torch.Tensor],
        gt_layers_hwc_by_scale: Dict[int, List[torch.Tensor]],
        pred_cache_data: Any,
    ) -> Dict[str, torch.Tensor]:
        bsz, n_blocks_chunk, block_size_q_h, block_size_q_w, _ = q_xy_block_chunk.shape
        q_xy_flat_chunk = q_xy_block_chunk.reshape(
            bsz, n_blocks_chunk * block_size_q_h * block_size_q_w, 2
        )
        pred_pixel_values = (
            batch["pixel_values"]
            if self.supervision_strategy.use_full_image
            else batch["pixel_values_container"]
        )
        pred_layers_flat = cast(
            List[torch.Tensor],
            self.vit_up(
                pixel_values=pred_pixel_values,
                q_xy_normalized=q_xy_flat_chunk,
                cache_data=pred_cache_data,
            ),
        )

        gt_layers_by_scale: Dict[int, List[torch.Tensor]] = {
            scale: [] for scale in hidden_ij_blocks_chunk
        }
        pred_layers_by_scale: Dict[int, List[torch.Tensor]] = {
            scale: [] for scale in hidden_ij_blocks_chunk
        }

        for layer_pred_flat in pred_layers_flat:
            pred_q_chunk = layer_pred_flat.view(
                bsz,
                n_blocks_chunk,
                block_size_q_h,
                block_size_q_w,
                layer_pred_flat.shape[-1],
            )
            # Required by new supervision API: map query features to hidden grid.
            pred_hidden_by_scale = self.supervision_strategy.query_features_to_hidden(
                pred_q_chunk
            )

            for scale, ij_blocks_chunk in hidden_ij_blocks_chunk.items():
                ij_blocks_chunk = ij_blocks_chunk.to(device=layer_pred_flat.device)
                i_idx = ij_blocks_chunk[..., 0].long()
                j_idx = ij_blocks_chunk[..., 1].long()

                gt_layer_hwc = gt_layers_hwc_by_scale[scale][
                    len(gt_layers_by_scale[scale])
                ]
                gt_chunk = gt_layer_hwc[:, i_idx, j_idx, :]
                pred_chunk = pred_hidden_by_scale[scale]

                gt_layers_by_scale[scale].append(
                    gt_chunk.reshape(bsz, -1, gt_chunk.shape[-1])
                )
                pred_layers_by_scale[scale].append(
                    pred_chunk.reshape(bsz, -1, pred_chunk.shape[-1])
                )

        chunk_loss_dict = self.loss_fn(
            gt_layers_by_scale=gt_layers_by_scale,
            pred_layers_by_scale=pred_layers_by_scale,
        )
        if not isinstance(chunk_loss_dict, dict):
            raise TypeError("Loss module must return a dict[str, Tensor].")
        if "loss" not in chunk_loss_dict:
            raise KeyError("Loss module output must include key 'loss'.")
        return cast(Dict[str, torch.Tensor], chunk_loss_dict)

    def _compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        (
            q_xy_block_chunks,
            hidden_ij_blocks_chunks,
            gt_layers_hwc_by_scale,
            pred_cache_data,
        ) = self._prepare_supervision_data(batch)

        accumulated_metrics: Optional[Dict[str, torch.Tensor]] = None
        for q_xy_block_chunk, hidden_ij_blocks_chunk in zip(
            q_xy_block_chunks, hidden_ij_blocks_chunks
        ):
            chunk_loss_dict = self._compute_chunk_loss_dict(
                batch=batch,
                q_xy_block_chunk=q_xy_block_chunk,
                hidden_ij_blocks_chunk=hidden_ij_blocks_chunk,
                gt_layers_hwc_by_scale=gt_layers_hwc_by_scale,
                pred_cache_data=pred_cache_data,
            )
            accumulated_metrics = merge_chunk_metrics(
                acc=accumulated_metrics,
                chunk_metrics=cast(Dict[str, torch.Tensor], chunk_loss_dict),
            )

        if accumulated_metrics is None:
            raise RuntimeError("No chunk losses were accumulated.")
        return normalize_chunk_metrics(
            accumulated_metrics,
            n_chunks=len(q_xy_block_chunks),
        )

    def training_step(self, batch, batch_idx):
        self._log_learning_rate()
        optimizer = cast(torch.optim.Optimizer, self.optimizers())

        accumulate_grad_batches = max(
            int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1), 1
        )
        is_first_accum_step = batch_idx % accumulate_grad_batches == 0
        should_step = (batch_idx + 1) % accumulate_grad_batches == 0 or getattr(
            self.trainer, "is_last_batch", False
        )

        if is_first_accum_step:
            optimizer.zero_grad()

        (
            q_xy_block_chunks,
            hidden_ij_blocks_chunks,
            gt_layers_hwc_by_scale,
            pred_cache_data,
        ) = self._prepare_supervision_data(batch)
        n_chunks = len(q_xy_block_chunks)
        if n_chunks == 0:
            raise RuntimeError("No query blocks were produced.")

        accumulated_metrics: Optional[Dict[str, torch.Tensor]] = None
        backward_divisor = float(n_chunks * accumulate_grad_batches)

        for chunk_idx, (q_xy_block_chunk, hidden_ij_blocks_chunk) in enumerate(
            zip(q_xy_block_chunks, hidden_ij_blocks_chunks)
        ):
            chunk_loss_dict = self._compute_chunk_loss_dict(
                batch=batch,
                q_xy_block_chunk=q_xy_block_chunk,
                hidden_ij_blocks_chunk=hidden_ij_blocks_chunk,
                gt_layers_hwc_by_scale=gt_layers_hwc_by_scale,
                pred_cache_data=pred_cache_data,
            )

            retain_graph = chunk_idx < (n_chunks - 1)
            self.manual_backward(
                chunk_loss_dict["loss"] / backward_divisor,
                retain_graph=retain_graph,
            )
            accumulated_metrics = merge_chunk_metrics(
                acc=accumulated_metrics,
                chunk_metrics=chunk_loss_dict,
            )

        if accumulated_metrics is None:
            raise RuntimeError("No chunk losses were accumulated.")
        loss_dict = normalize_chunk_metrics(accumulated_metrics, n_chunks=n_chunks)

        if should_step:
            gradient_clip_val = 1.0
            if float(gradient_clip_val) > 0.0:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=float(gradient_clip_val),
                    gradient_clip_algorithm=getattr(
                        self.trainer, "gradient_clip_algorithm", "norm"
                    ),
                )
            optimizer.step()
            if (
                self.lr_scheduler_cfg is not None
                and self.lr_scheduler_cfg.interval == "step"
            ):
                scheduler = cast(Any, self.lr_schedulers())
                if scheduler is not None:
                    if isinstance(scheduler, list):
                        for item in scheduler:
                            item.step()
                    else:
                        scheduler.step()

        self._log_losses(
            prefix="train", loss_dict=loss_dict, on_step=True, on_epoch=True
        )
        return loss_dict["loss"]

    def on_train_epoch_end(self):
        if self.lr_scheduler_cfg is None or self.lr_scheduler_cfg.interval != "epoch":
            return
        scheduler = cast(Any, self.lr_schedulers())
        if scheduler is None:
            return
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
            return
        scheduler.step()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        del batch_idx
        loss_dict = self._compute_loss(batch)
        self._log_losses(
            prefix="val",
            loss_dict={"loss": loss_dict["loss"]},
            on_step=False,
            on_epoch=True,
        )
        return loss_dict["loss"]

    def configure_optimizers(self) -> Any:
        optimizer = instantiate_class(
            args=(self.parameters(),),
            init=self.optimizer_init,
        )
        if not self.lr_scheduler_cfg:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.lr_scheduler_cfg.to_lightning_config(optimizer),
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Filter saved state_dict so that when LoRA is used we only persist
        trainable backbone parameters (the LoRA adapters) and keep all other
        non-backbone parameters as usual. This prevents saving the full frozen
        backbone weights when using PEFT/LoRA.
        """
        if not getattr(self, "_backbone_uses_lora", False):
            return

        # Lightning stores model params under "state_dict" in the checkpoint
        state = checkpoint.get("state_dict", None)
        if state is None:
            return

        trainable_names = {
            name for name, p in self.named_parameters() if p.requires_grad
        }

        filtered_state: Dict[str, Any] = {}
        for k, v in state.items():
            # keep everything that is not part of the backbone
            if not k.startswith("backbone."):
                filtered_state[k] = v
                continue

            # for backbone keys, only keep those that correspond to trainable params
            if k in trainable_names:
                filtered_state[k] = v

        checkpoint["state_dict"] = filtered_state
