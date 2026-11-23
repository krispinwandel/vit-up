"""Train from a composed Hydra config (see `configs/train.yaml` and `experiments/`)."""
import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from .module import ViTUpTraining


def train(cfg: DictConfig) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    torch.set_float32_matmul_precision("medium")
    seed_everything(cfg["seed"], workers=True)
    output_dir = Path(cfg["output_dir"])
    model, schedule = cfg["model"], cfg["schedule"]

    module = ViTUpTraining(
        vit_up=model["vit_up"],
        backbone=model["backbone"],
        objective=instantiate(cfg["objective"]),
        optimizer=instantiate(cfg["optimizer"]),
        schedule=schedule,
        image_size=model["image_size"],
        image_sizes=model["image_sizes"],
        lora=cfg["adaptation"]["lora"],
        train_lora=cfg["adaptation"]["train_lora"],
        initialization=cfg["initialization"],
        compile=cfg["compile"],
    )
    data = instantiate(cfg["data"], batch_size=schedule["batch_size"], views=cfg["views"])
    trainer = instantiate(
        cfg["trainer"],
        max_steps=schedule["max_steps"],
        max_epochs=-1,
        default_root_dir=str(output_dir),
        logger=instantiate(cfg["logger"]),
        callbacks=[instantiate(callback) for callback in cfg["callbacks"]],
    )
    trainer.fit(module, datamodule=data, ckpt_path=cfg["ckpt_path"], weights_only=False)
