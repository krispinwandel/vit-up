from __future__ import annotations

import datetime
import os
import random
from pathlib import Path
import wandb

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vit_up.eval_kits.probing_toolkit.metrics import eval_metrics
from vit_up.eval_kits.probing_toolkit.probe_model import (
    UpsamplerEvaluator,
    maybe_data_parallel,
)
from vit_up.eval_kits.probing_toolkit.utils.training import get_dataloaders

LOG_INTERVAL = 100


def _set_seed(cfg):
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For full determinism (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _init_wandb(cfg):
    wandb_enabled = OmegaConf.select(cfg, "logger.wandb.use_wandb", default=False)
    print(f"WandB enabled: {wandb_enabled}")
    if not wandb_enabled:
        return None
    try:
        import wandb

        wb_cfg = OmegaConf.to_container(
            OmegaConf.select(cfg, "logger.wandb"), resolve=True
        )
        project = wb_cfg.get("project") or "nf-dino-probing"
        run_dir = HydraConfig.get().run.dir
        # Build default tags/run_name similar to scripts/probing.py
        mode = OmegaConf.select(cfg, "mode", default="eval")
        dataset = OmegaConf.select(cfg, "dataset", default=None)
        task = OmegaConf.select(cfg, "task", default=None)

        # Determine model spec (may be a string like 'dinov3/uplift_probe' or a config node)
        model_raw = OmegaConf.select(cfg, "model", default=None)
        model_spec = None
        if isinstance(model_raw, str):
            model_spec = model_raw
        else:
            # try common fields
            model_spec = OmegaConf.select(cfg, "model.name", default=None)

        def split_model(spec: str) -> tuple[str, str]:
            if not spec:
                return "unknown", "unknown"
            if "/" in spec:
                b, m = spec.split("/", 1)
            elif "." in spec:
                b, m = spec.split(".", 1)
            else:
                b, m = "default", spec
            return b, m.replace("/", ".")

        backbone, method = split_model(model_spec)
        backbone_spec = OmegaConf.select(cfg, "backbone.name", default="default")
        # backbone = "unknown"
        # if "dinov2" in backbone_spec:
        #     backbone = "dinov2"
        # elif "dinov3" in backbone_spec:
        #     backbone = "dinov3"
        backbone = str(backbone_spec)
        run_date = datetime.datetime.now().strftime("%Y-%m-%d")
        run_name = wb_cfg.get("run_name") or f"{method}-{run_date}"
        tags = wb_cfg.get("tags") or [mode, backbone, task, dataset, method]
        if isinstance(tags, (list, tuple)):
            tags = [str(t) for t in tags if t is not None]
        else:
            tags = [str(tags)]
        print("WandB config: " f"project={project}, run_name={run_name}, tags={tags}")
        wandb.init(
            project=project,
            name=run_name,
            tags=tags or None,
            config=OmegaConf.to_container(cfg, resolve=False),
            dir=run_dir,
            reinit=True,
        )
        return wandb
    except Exception as exc:
        print(f"WandB init failed: {exc}")
        return None


def _build_io(cfg, mode: str, task: str, checkpoint_exists: bool, run_dir: str):
    terminal_console = Console()
    log_dir = os.path.join(run_dir, "logs")
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(log_dir, exist_ok=True)
    if mode == "eval":
        file_name = f"eval_{cfg.model.name}_{task}.log"
    else:
        file_name = (
            f"eval_{cfg.model.name}_{task}.log"
            if checkpoint_exists
            else f"train_{cfg.model.name}_{task}.log"
        )
    source_path = os.path.join(run_dir, file_name)
    symlink_path = os.path.join(log_dir, file_name)
    if not os.path.exists(symlink_path):
        os.symlink(source_path, symlink_path)
    file_console = Console(file=open(source_path, "w"))
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, file_name.replace(".log", "")))

    def log_print(*args, **kwargs):
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    return log_print, file_console, writer


def _build_backbone_and_model(cfg, device, log_print):
    backbone = instantiate(cfg.backbone).to(device)
    backbone.requires_grad_(False)
    backbone.eval()

    model = instantiate(cfg.model).to(device)
    model_ckpt = OmegaConf.select(cfg, "model_ckpt", default=None)
    if model_ckpt:
        checkpoint = torch.load(model_ckpt)
        if cfg.model.name == "jafar":
            model.load_state_dict(checkpoint["jafar"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        log_print(f"[green]Loaded upsampler from checkpoint: {model_ckpt}[/green]")
    else:
        model.eval()
        log_print(
            "[yellow]No upsampler checkpoint provided, using instantiated weights.[/yellow]"
        )

    return backbone, model


def _build_loaders(cfg, backbone):
    if cfg.task == "depth":
        mean = [0, 0, 0]
        std = [1, 1, 1]
    else:
        mean, std = None, None
    return get_dataloaders(cfg, backbone, is_evaluation=True, mean=mean, std=std)


def _run_eval_mode(
    cfg, head_ckpt, log_print, writer, file_console, device, backbone, model, val_loader
):
    evaluator = UpsamplerEvaluator(model, backbone, device, cfg, writer, file_console)
    wandb_run = _init_wandb(cfg)
    if wandb_run is not None:
        evaluator.wandb = wandb_run
    evaluator.set_up_classifier(head_ckpt)
    log_print(f"[green]Loaded probe head from checkpoint: {head_ckpt}[/green]")
    metrics = evaluator.evaluate(val_loader, epoch=0)
    return metrics


def _run_train_mode(
    cfg,
    head_ckpt,
    log_print,
    writer,
    file_console,
    device,
    backbone,
    model,
    train_loader,
    val_loader,
    checkpoint_path: str,
):
    evaluator = UpsamplerEvaluator(model, backbone, device, cfg, writer, file_console)
    wandb_run = _init_wandb(cfg)
    if wandb_run is not None:
        evaluator.wandb = wandb_run

    task = cfg.task
    if Path(checkpoint_path).exists():
        log_print(f"[green]Loading classifier from {checkpoint_path}[/green]")
        evaluator.set_up_classifier(checkpoint_path)
        metrics = evaluator.evaluate(val_loader, epoch=0)
        return metrics

    log_print(f"[yellow]Training classifier... {checkpoint_path} not found[/yellow]\n")
    if head_ckpt is not None:
        evaluator.set_up_classifier(head_ckpt)
        log_print(f"[green]Initialized probe head from checkpoint: {head_ckpt}[/green]")
    evaluator.set_optimizer(cfg, loader=train_loader)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[yellow]Loss: {task.fields[loss]:.6f}"),
        TextColumn("[green]Step: {task.fields[step]:.6e}"),
        console=file_console,
    )

    start_time = datetime.datetime.now()
    metrics = {}
    with progress:
        log_print(f"[yellow]Training for {cfg.num_epochs} epochs[/yellow]\n")

        for epoch in range(cfg.num_epochs):
            evaluator.train(train_loader, progress, epoch, start_time)
            metrics = evaluator.evaluate(val_loader, epoch)
            if cfg.sanity:
                break

        os.makedirs(Path(checkpoint_path).parent, exist_ok=True)
        evaluator.save_checkpoint(checkpoint_path)
        # if wandb_run is not None:
        #     artifact_name = f"probe_head-{task}-{cfg.model.name}"
        #     artifact = wandb.Artifact(artifact_name, type="checkpoint")
        #     artifact.add_file(checkpoint_path)
        #     logged_artifact = wandb_run.log_artifact(artifact)
        #     logged_artifact.wait()  # Ensure the artifact is uploaded before proceeding
        #     log_print(
        #         f"[green]Logged probe head checkpoint to WandB: {artifact_name}[/green]"
        #     )
    return metrics


def run(cfg):
    _set_seed(cfg)

    mode = str(cfg.get("mode", "eval"))
    # Use top-level `task`, `model_ckpt`, `head_ckpt` from the config
    task = cfg.task
    head_ckpt = OmegaConf.select(cfg, "head_ckpt", default=None)
    # if not head_ckpt:
    #     raise ValueError(
    #         "`head_ckpt` must be provided and point to a probe head checkpoint."
    #     )

    run_dir = HydraConfig.get().run.dir
    checkpoint_path = os.path.join(
        run_dir, "checkpoints", task, f"{cfg.model.name}.pth"
    )
    checkpoint_exists = Path(checkpoint_path).exists()
    log_print, file_console, writer = _build_io(
        cfg, mode, task, checkpoint_exists, run_dir
    )

    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(
        f"[bold blue]Starting {('eval' if mode == 'eval' else 'probing')} run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold blue]"
    )
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")
    log_print(f"\n[bold cyan]Processing {task} task:[/bold cyan]")
    log_print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    backbone, model = _build_backbone_and_model(cfg, device, log_print)
    train_loader, val_loader = _build_loaders(cfg, backbone)
    log_print(f"[bold cyan]Train Dataset size: {len(train_loader.dataset)}[/bold cyan]")
    log_print(f"[bold cyan]Val Dataset size: {len(val_loader.dataset)}[/bold cyan]")

    if mode == "eval":
        metrics = _run_eval_mode(
            cfg,
            head_ckpt,
            log_print,
            writer,
            file_console,
            device,
            backbone,
            model,
            val_loader,
        )
    elif mode in {"train", "finetune"}:
        metrics = _run_train_mode(
            cfg,
            head_ckpt,
            log_print,
            writer,
            file_console,
            device,
            backbone,
            model,
            train_loader,
            val_loader,
            checkpoint_path,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # save metrics
    metrics_path = os.path.join(run_dir, f"{mode}_metrics.json")
    with open(metrics_path, "w") as f:
        import json

        json.dump(metrics, f, indent=2)

    file_console.file.close()
    writer.close()


@hydra.main(config_path="../config/probing", config_name="probing", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
