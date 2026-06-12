from __future__ import annotations

import datetime
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from rich.console import Console
from torchmetrics.classification import Accuracy, JaccardIndex
from tqdm import tqdm

from vit_up.eval_kits.probing_toolkit.metrics import eval_metrics
from vit_up.eval_kits.probing_toolkit.loss import GradientLoss, SigLoss
from vit_up.eval_kits.probing_toolkit.utils.training import get_batch

LOG_INTERVAL = 100


def maybe_data_parallel(module):
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        wrapped = nn.DataParallel(module)
        if hasattr(module, "config"):
            wrapped.config = module.config
        return wrapped
    return module


def unwrap_parallel(module):
    return module.module if isinstance(module, nn.DataParallel) else module


class UpsamplerEvaluator:
    def __init__(self, model, backbone, device, cfg, writer, console):
        self.model, self.backbone, self.device, self.cfg, self.writer, self.console = (
            model,
            backbone,
            device,
            cfg,
            writer,
            console,
        )

        self.mean = backbone.config["mean"]
        self.std = backbone.config["std"]

        # Initialize task-specific components
        if "seg" == cfg.task:
            self.accuracy_metric = Accuracy(
                num_classes=cfg.metrics.seg.num_classes, task="multiclass"
            ).to(device)
            self.iou_metric = JaccardIndex(
                num_classes=cfg.metrics.seg.num_classes, task="multiclass"
            ).to(device)
            self.classifier = nn.Conv2d(
                cfg.model.feature_dim, cfg.metrics.seg.num_classes, 1
            ).to(device)
            self.classifier = maybe_data_parallel(self.classifier)
        elif "depth" == cfg.task:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.classifier = nn.Conv2d(
                (2 * cfg.model.feature_dim),
                256,
                kernel_size=1,
                padding=0,
                stride=1,
            ).to(device)
            self.classifier = maybe_data_parallel(self.classifier)
            self.image_processor = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            self.depth_model = (
                AutoModelForDepthEstimation.from_pretrained(
                    "depth-anything/Depth-Anything-V2-Small-hf"
                )
                .to(device)
                .eval()
            )
            self.sigloss = SigLoss(
                valid_mask=True,
                loss_weight=1.0,
                warm_up=True,
                max_depth=cfg.metrics.depth.max_depth,
            )
            self.gradientloss = GradientLoss(
                valid_mask=True, loss_weight=0.5, max_depth=cfg.metrics.depth.max_depth
            )

    def set_up_classifier(self, checkpoint_path):
        """Load classifier weights from a checkpoint."""
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            unwrap_parallel(self.classifier).load_state_dict(
                checkpoint["model_state_dict"]
            )
            self.console.print(f"Loaded classifier from checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    def set_optimizer(self, cfg, loader):
        params_classifier = self.classifier.parameters()
        optimizer = instantiate(cfg.optimizer, params=list(params_classifier))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs * len(loader)
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

        num_params = sum(p.numel() for p in params_classifier if p.requires_grad)
        self.log_print(
            f"[bold cyan]Number of optimized parameters: {num_params:,}[/bold cyan]"
        )

    def log_print(self, *args, **kwargs):
        """Log to both file and terminal with immediate updates"""
        Console(force_terminal=True).print(*args, **kwargs)
        self.console.print(*args, **kwargs)
        if hasattr(self.console, "file") and self.console.file:
            self.console.file.flush()

    def log_tensorboard(self, step, loss=None, metrics=None, lr=None):
        """Log losses and metrics to TensorBoard."""
        if loss is not None:
            self.writer.add_scalar("Loss/Step", loss, step)
        if lr is not None:
            self.writer.add_scalar("LR", lr, step)
        if metrics is not None:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, step)
        if hasattr(self, "wandb") and self.wandb is not None:
            to_log = {}
            if loss is not None:
                to_log["Loss/Step"] = loss
            if lr is not None:
                to_log["LR"] = lr
            if metrics is not None:
                for metric_name, metric_value in metrics.items():
                    to_log[f"Metrics/{metric_name}"] = metric_value
            if to_log:
                try:
                    self.wandb.log(to_log, step=step)
                except Exception:
                    pass

    def process_batch(self, image_batch, target, is_training=True):
        H, W = target.shape[-2:]
        with torch.no_grad():
            pred_bhwc = self.model(image_batch, (H, W))
            pred = pred_bhwc.permute(0, 3, 1, 2).contiguous()

            cls_token = None
            if self.cfg.task == "depth":
                cls_token = self.backbone(image_batch)[1]

        if self.cfg.task == "depth":
            cls_token = F.normalize(cls_token, dim=2)[:, 0, :]
            cls_token = cls_token[:, :, None, None]
            pred = torch.cat([pred, cls_token.expand_as(pred)], dim=1)

        pred = self.classifier(pred)

        if pred.shape[-2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode="bilinear")

        if self.cfg.task == "seg":
            if target.shape[-2:] != pred.shape[-2:]:
                target = (
                    F.interpolate(
                        target.unsqueeze(1),
                        size=pred.shape[-2:],
                        mode="nearest-exact",
                    )
                    .squeeze(1)
                    .to(target.dtype)
                )

            valid_mask = target != 255

            pred = rearrange(pred, "b c h w -> (b h w) c")
            target = rearrange(target, "b h w -> (b h w)")
            valid_mask = rearrange(valid_mask, "b h w -> (b h w)")

            pred = pred[valid_mask]
            target = target[valid_mask]

            return pred, target

        if self.cfg.task == "depth":
            depth_image_batch = (
                255 * image_batch.permute(0, 2, 3, 1).cpu().numpy()
            ).astype(np.uint8)
            inputs = self.image_processor(
                images=depth_image_batch, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                pseudo_depth = self.depth_model(**inputs)["predicted_depth"]

            target = F.interpolate(
                pseudo_depth.unsqueeze(1),
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            bins = torch.linspace(
                self.cfg.metrics.depth.min_depth,
                self.cfg.metrics.depth.max_depth,
                256,
                device="cuda",
            )
            pred = F.relu(pred)
            eps = 0.1
            pred = pred + eps
            pred = pred / pred.sum(dim=1, keepdim=True)
            pred = torch.einsum("ikmn,k->imn", [pred, bins]).unsqueeze(dim=1)
            return pred, target

        return NotImplementedError

    def train(self, train_dataloader, progress, epoch, start_time):
        self.log_print(f"[yellow]Training model epoch {epoch+1}...[/yellow]")
        self.backbone.eval()
        self.model.eval()
        self.classifier.train()

        self.steps_per_epoch = len(train_dataloader)

        epoch_task = progress.add_task(
            f"Epoch {epoch+1}/{self.cfg.num_epochs}",
            total=len(train_dataloader),
            loss=0.0,
            step=0,
        )
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            if random.random() < 0.5:
                image_batch = torch.flip(image_batch, dims=[3])
                target = torch.flip(target, dims=[2])

            self.optimizer.zero_grad()

            pred, target = self.process_batch(image_batch, target, is_training=True)

            if self.cfg.task == "seg":
                loss = F.cross_entropy(pred, target)
            elif self.cfg.task == "depth":
                loss = self.sigloss(pred, target) + self.gradientloss(pred, target)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            avg_loss = total_loss / (batch_idx + 1)

            if (batch_idx + 1) % LOG_INTERVAL == 0 or batch_idx == len(
                train_dataloader
            ) - 1:
                elapsed_time = datetime.datetime.now() - start_time
                elapsed_str = str(elapsed_time).split(".")[0]
                current_lr = self.optimizer.param_groups[0]["lr"]

                progress.update(
                    epoch_task,
                    advance=LOG_INTERVAL,
                    loss=avg_loss,
                    step=batch_idx + 1,
                )

                self.log_print(
                    f"[cyan]Iteration {batch_idx + 1}[/cyan] - "
                    f"Loss: {avg_loss:.6f} - "
                    f"LR: {current_lr:.5e} - "
                    f"Elapsed Time: {elapsed_str}"
                )

                if self.console and hasattr(self.console, "file"):
                    self.console.file.flush()

                progress.refresh()

                global_step = epoch * len(train_dataloader) + batch_idx
                self.log_tensorboard(global_step, loss=avg_loss, lr=current_lr)

            if self.cfg.sanity and batch_idx == 0:
                break

            self.scheduler.step()

            if self.cfg.sanity:
                break

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log_print(
            f"[bold cyan]Epoch {epoch+1} Summary:[/bold cyan] "
            f"Loss = {avg_loss:.6f} - "
            f"LR = {current_lr:.2e}"
        )

        return

    def save_checkpoint(self, checkpoint_path):
        console = self.console
        checkpoint = {
            "epoch": self.cfg.num_epochs - 1,
            "model_state_dict": unwrap_parallel(self.classifier).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "task": self.cfg.task,
            "backbone": self.cfg.backbone.name,
        }
        torch.save(checkpoint, checkpoint_path)
        self.log_print(
            f"[bold green]Training completed. Model saved at: {checkpoint_path}[/bold green]"
        )
        console.file.flush()
        return

    @torch.inference_mode()
    def evaluate(self, dataloader, epoch):
        self.log_print("[yellow]Evaluating model...[/yellow]")
        torch.cuda.empty_cache()

        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        nsamples = 0
        results = {}
        if self.cfg.task == "seg":
            self.accuracy_metric.reset()
            self.iou_metric.reset()
        elif self.cfg.task == "depth":
            results = {
                "d1": 0,
                "d2": 0,
                "d3": 0,
                "abs_rel": 0,
                "sq_rel": 0,
                "rmse": 0,
                "rmse_log": 0,
                "log_10": 0,
                "silog": 0,
            }

            nsamples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            pred, target = self.process_batch(image_batch, target, is_training=False)

            if self.cfg.task == "seg":
                self.accuracy_metric(pred, target)
                self.iou_metric(pred, target)

            elif self.cfg.task == "depth":
                cur_results = eval_metrics(
                    target.cpu().detach().numpy(), pred.cpu().detach().numpy()
                )

                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1

            if self.cfg.sanity and batch_idx == 0:
                break

        metrics = {}
        if self.cfg.task == "seg":
            metrics.update(
                {
                    "accuracy": self.accuracy_metric.compute().item(),
                    "iou": self.iou_metric.compute().item(),
                }
            )
        elif self.cfg.task == "depth":
            for k in results.keys():
                metrics[k] = results[k].item() / nsamples

        steps_per_epoch = getattr(self, "steps_per_epoch", len(dataloader))
        global_step = (epoch + 1) * steps_per_epoch
        self.log_tensorboard(step=global_step, metrics=metrics)

        self.log_print(f"[bold green]Results: {metrics}[/bold green]")
        return metrics

    @torch.inference_mode()
    def simple_inference(self, image_batch):
        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        H, W = image_batch.shape[-2:]
        with torch.no_grad():
            features = self.model(image_batch, (H, W))
            features = features.permute(0, 3, 1, 2)

        pred = features
        pred = self.classifier(pred)
        pred = pred.argmax(dim=1)

        return pred, features, None
