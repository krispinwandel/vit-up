import lightning as L
import torch
from hydra.utils import instantiate
from peft import PeftModel
from torch import nn
from torch._functorch import config as functorch_config

from vit_up import checkpoint
from vit_up.model import load_backbone


class ViTUpTraining(L.LightningModule):
    """Train a ViT-Up decoder (and optionally backbone LoRA adapters) with a distillation objective.

    Args:
        vit_up: Config of the decoder (a `vit_up.model.ViTUp` spec).
        backbone: Hugging Face model id of the frozen backbone.
        objective: `GridDistillation` (Stage 1) or `OffsetDistillation` (Stage 2).
        optimizer: Partially instantiated optimizer, called with the trainable parameters.
        schedule: Batch accumulation and the warmup + cosine learning-rate schedule, in optimizer steps.
        image_size: Backbone input resolution, stored with exported checkpoints.
        image_sizes: Train at each of these resolutions and average the losses (default: `[image_size]`).
        lora: LoRA config for the backbone, or "inherit" to take it from `initialization`.
        train_lora: Train the adapters; otherwise they are merged into the frozen backbone.
        initialization: Checkpoint (release name or directory) whose weights initialize the model.
    """

    def __init__(self, vit_up: dict, backbone: str, objective: nn.Module, optimizer, schedule: dict,
                 image_size: int, image_sizes: list[int] | None = None, lora: dict | str | None = None,
                 train_lora: bool = True, initialization: str | None = None, gradient_clip_val: float = 1.0,
                 compile: bool = True):
        super().__init__()
        self.automatic_optimization = False
        self.vit_up_config = vit_up
        self.backbone_name = backbone
        self.objective = objective
        self.optimizer_factory = optimizer
        self.schedule = schedule
        self.image_size = image_size
        self.image_sizes = image_sizes or [image_size]
        self.gradient_clip_val = gradient_clip_val

        self.lora = checkpoint.load_config(initialization)["lora"] if lora == "inherit" else lora
        self.vit_up = instantiate(vit_up)
        self.backbone = load_backbone(backbone, self.lora)
        if initialization is not None:
            checkpoint.load_weights(initialization, self.vit_up, self.backbone)
        self.train_lora = isinstance(self.backbone, PeftModel) and train_lora
        if isinstance(self.backbone, PeftModel) and not train_lora:
            self.backbone.merge_adapter(safe_merge=True)
            self.backbone.requires_grad_(False)
        self.objective.setup(self)
        if compile:
            self.vit_up.compile()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode and self.train_lora)
        return self

    def _loss(self, batch, backward: bool) -> dict:
        prepared = self.objective.prepare(self, batch)
        weight = 1.0 / len(self.image_sizes)
        backward_weight = weight / self.schedule["accumulate_grad_batches"] if backward else None
        metrics = {}
        for size in self.image_sizes:
            size_metrics = self.objective(self, batch, prepared, size, backward_weight)
            for key, value in size_metrics.items():
                metrics[key] = metrics.get(key, 0) + weight * value
            if len(self.image_sizes) > 1:
                metrics[f"loss_{size}"] = size_metrics["loss"]
        return metrics

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        accumulation = self.schedule["accumulate_grad_batches"]
        if batch_idx % accumulation == 0:
            optimizer.zero_grad()
        self.log("train/lr", optimizer.param_groups[0]["lr"], on_step=True, on_epoch=False)
        # Chunks and resolutions backpropagate separately through a shared image cache,
        # which compiled backward graphs must not donate.
        with functorch_config.patch(donated_buffer=False):
            metrics = self._loss(batch, backward=True)
        if (batch_idx + 1) % accumulation == 0 or self.trainer.is_last_batch:
            self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            optimizer.step()
            self.lr_schedulers().step()
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        metrics = self._loss(batch, backward=False)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_factory([p for p in self.parameters() if p.requires_grad])
        s = self.schedule
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=s["start_factor"], end_factor=1.0,
                                              total_iters=s["warmup_steps"]),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s["cosine_steps"], eta_min=s["eta_min"]),
        ], milestones=[s["warmup_steps"]])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @staticmethod
    def _is_frozen_backbone_weight(key: str) -> bool:
        return key.startswith("backbone.") and "lora_" not in key

    def on_save_checkpoint(self, ckpt):
        # Frozen backbone weights are reloaded from the Hub, which keeps checkpoints small.
        ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items() if not self._is_frozen_backbone_weight(k)}

    def on_load_checkpoint(self, ckpt):
        for key, value in self.state_dict().items():
            if self._is_frozen_backbone_weight(key):
                ckpt["state_dict"].setdefault(key, value)

    def export(self, directory) -> None:
        """Save the current model as a checkpoint directory (see `vit_up.checkpoint`)."""
        config = {"backbone": self.backbone_name, "backbone_revision": self.backbone.config._commit_hash,
                  "lora": self.lora, "image_size": self.image_size, "vit_up": self.vit_up_config}
        checkpoint.save(directory, self.vit_up, self.backbone, config)
