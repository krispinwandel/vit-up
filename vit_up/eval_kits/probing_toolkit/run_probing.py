"""Train or evaluate linear probes on frozen upsampled features (segmentation and depth)."""
import datetime
import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from torch.utils.tensorboard import SummaryWriter

from vit_up.eval_kits.probing_toolkit.probe_model import UpsamplerEvaluator
from vit_up.eval_kits.probing_toolkit.utils.training import get_dataloaders


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_io(cfg, run_dir):
    name = f"{cfg.mode}_{cfg.model.name}_{cfg.task}"
    terminal = Console()
    file_console = Console(file=open(os.path.join(run_dir, f"{name}.log"), "w"))
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb", name))

    def log_print(*args, **kwargs):
        terminal.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    return log_print, file_console, writer


def _train(cfg, evaluator, train_loader, val_loader, head_path, log_print, file_console):
    if Path(head_path).exists():
        log_print(f"[green]Evaluating existing probe head {head_path}[/green]")
        evaluator.set_up_classifier(head_path)
        return evaluator.evaluate(val_loader, epoch=0)
    if cfg.head_ckpt is not None:
        evaluator.set_up_classifier(cfg.head_ckpt)
    evaluator.set_optimizer(cfg, loader=train_loader)
    progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                        TaskProgressColumn(), TextColumn("[yellow]Loss: {task.fields[loss]:.6f}"),
                        TextColumn("[green]Step: {task.fields[step]:.6e}"), console=file_console)
    start_time = datetime.datetime.now()
    with progress:
        for epoch in range(cfg.num_epochs):
            evaluator.train(train_loader, progress, epoch, start_time)
            if cfg.sanity:
                break
            if cfg.eval_every_n_epochs and (epoch + 1) % cfg.eval_every_n_epochs == 0 and epoch + 1 < cfg.num_epochs:
                evaluator.evaluate(val_loader, epoch)
    Path(head_path).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_checkpoint(head_path)
    return evaluator.evaluate(val_loader, epoch)


@hydra.main(config_path="../config/probing", config_name="probing", version_base=None)
def main(cfg: DictConfig) -> None:
    _set_seed(cfg.seed)
    run_dir = HydraConfig.get().run.dir
    log_print, file_console, writer = _build_io(cfg, run_dir)
    log_print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = instantiate(cfg.backbone).to(device).eval().requires_grad_(False)
    model = instantiate(cfg.model).to(device).eval()
    # Depth targets come from Depth Anything, which expects unnormalized RGB.
    mean_std = ([0, 0, 0], [1, 1, 1]) if cfg.task == "depth" else (None, None)
    train_loader, val_loader = get_dataloaders(cfg, backbone, is_evaluation=True, mean=mean_std[0], std=mean_std[1])
    evaluator = UpsamplerEvaluator(model, backbone, device, cfg, writer, file_console)

    if cfg.mode == "eval":
        evaluator.set_up_classifier(cfg.head_ckpt)
        metrics = evaluator.evaluate(val_loader, epoch=0)
    else:
        head_path = os.path.join(run_dir, "checkpoints", cfg.task, f"{cfg.model.name}.pth")
        metrics = _train(cfg, evaluator, train_loader, val_loader, head_path, log_print, file_console)

    Path(run_dir, f"{cfg.mode}_metrics.json").write_text(json.dumps(metrics, indent=2))
    file_console.file.close()
    writer.close()


if __name__ == "__main__":
    main()
