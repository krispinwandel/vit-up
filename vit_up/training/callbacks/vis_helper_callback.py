from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, List
import random
from PIL import Image
import numpy as np

import torch

from lightning.pytorch.callbacks import Callback

from vit_up.utils import img_transforms
from vit_up.visualization.helpers.base import VisHelper


def log_wandb_image(trainer: Any, key: str, img: Image.Image, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import wandb
    except Exception:
        return

    loggers = trainer.loggers if getattr(trainer, "loggers", None) else []
    seen_experiments: set[int] = set()
    step = int(getattr(trainer, "global_step", 0))
    for logger in loggers:
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            continue
        if not hasattr(experiment, "log"):
            continue
        exp_id = id(experiment)
        if exp_id in seen_experiments:
            continue
        seen_experiments.add(exp_id)
        experiment.log(
            {key: wandb.Image(np.array(img))},
            step=step,
        )


class VisHelperCallback(Callback):
    """Generic callback that delegates visualization generation to a VisHelper class."""

    def __init__(
        self,
        vis_helper: VisHelper,
        num_images: int = 4,
        output_subdir: str = "val_vis_helper",
        run_every_n_val_epochs: int = 1,
        fixed_seed: int = 1234,
        log_to_wandb: bool = True,
    ):
        super().__init__()

        self.vis_helper = vis_helper
        self.num_images = int(num_images)
        self.output_subdir = output_subdir
        self.run_every_n_val_epochs = max(1, int(run_every_n_val_epochs))
        self.fixed_seed = int(fixed_seed)
        self.log_to_wandb = bool(log_to_wandb)

        self._fixed_batch_cache: Optional[Dict[str, Any]] = None

    def _slice_tensor_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[0] > 0:
                out[key] = value[: self.num_images]
        if "pixel_values" not in out:
            raise RuntimeError(
                "Validation batch is missing required key 'pixel_values'."
            )
        return out

    def _collect_dataset_batch(
        self, val_iter: Any, n_items: int
    ) -> Dict[str, torch.Tensor]:
        if n_items <= 0:
            raise RuntimeError("Expected num_images > 0 for fixed val batch.")

        first_item = next(val_iter)
        if not isinstance(first_item, dict):
            raise RuntimeError("Validation dataset items must be dictionaries.")

        tensor_keys = [
            key for key, value in first_item.items() if torch.is_tensor(value)
        ]
        if "pixel_values" not in tensor_keys:
            raise RuntimeError(
                "Validation dataset items are missing required key 'pixel_values'."
            )

        out: Dict[str, torch.Tensor] = {}
        for key in sorted(tensor_keys):
            value = first_item[key]
            out[key] = torch.empty(
                (n_items, *value.shape),
                dtype=value.dtype,
                device=value.device,
            )
            out[key][0].copy_(value)

        for idx in range(1, n_items):
            item = next(val_iter)
            if not isinstance(item, dict):
                raise RuntimeError("Validation dataset items must be dictionaries.")
            for key, batch_tensor in out.items():
                value = item.get(key)
                if not torch.is_tensor(value):
                    raise RuntimeError(
                        "Validation dataset item is missing a required tensor key "
                        f"during fixed-batch collection: {key}."
                    )
                if value.shape != batch_tensor.shape[1:]:
                    raise RuntimeError(
                        "Validation dataset tensor shape changed within fixed-batch collection. "
                        f"key={key}, expected={tuple(batch_tensor.shape[1:])}, got={tuple(value.shape)}."
                    )
                batch_tensor[idx].copy_(value)

        return out

    def _get_fixed_val_batch(self, trainer: Any) -> Dict[str, torch.Tensor]:
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "val_dataset"):
            val_loader = trainer.val_dataloaders
            if isinstance(val_loader, (list, tuple)):
                val_loader = val_loader[0]
            batch = next(iter(val_loader))
            if not isinstance(batch, dict):
                raise RuntimeError("Validation loader must return dictionary batches.")
            return self._slice_tensor_batch(batch)

        state_py = random.getstate()
        state_torch = torch.get_rng_state()
        random.seed(self.fixed_seed)
        torch.manual_seed(self.fixed_seed)
        try:
            val_iter = iter(datamodule.val_dataset)
            fixed_batch = self._collect_dataset_batch(
                val_iter=val_iter,
                n_items=self.num_images,
            )
        finally:
            random.setstate(state_py)
            torch.set_rng_state(state_torch)
        return fixed_batch

    def _initialize_helper_inputs(self, pl_module: Any):
        if self._fixed_batch_cache is None:
            raise RuntimeError("Helper and fixed batch must be initialized first.")

        pixel_values = self._fixed_batch_cache["pixel_values"].to(pl_module.device)
        imgs = [
            img_transforms.pixel_values_to_pil(pixel_values[idx].detach().float().cpu())
            for idx in range(int(pixel_values.shape[0]))
        ]
        self.vis_helper.set_input_images(imgs=imgs, pl_module=pl_module)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self._fixed_batch_cache = self._get_fixed_val_batch(trainer=trainer)
        self._initialize_helper_inputs(pl_module=pl_module)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return
        if ((trainer.current_epoch + 1) % self.run_every_n_val_epochs) != 0:
            return

        if self._fixed_batch_cache is None:
            self._fixed_batch_cache = self._get_fixed_val_batch(trainer=trainer)
        if self.vis_helper.pixel_values is None:
            self._initialize_helper_inputs(pl_module=pl_module)

        pixel_values = self._fixed_batch_cache["pixel_values"].to(pl_module.device)

        root_dir = Path(trainer.default_root_dir or ".")
        out_dir = root_dir / self.output_subdir / f"epoch_{trainer.current_epoch:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(int(pixel_values.shape[0])):
            vis_img = self.vis_helper.generate_vis(img_idx=img_idx, pl_module=pl_module)
            out_path = out_dir / f"val_img_{img_idx:02d}.png"
            vis_img.save(out_path)
            log_wandb_image(
                trainer,
                key=f"{self.output_subdir}/img_{img_idx:02d}",
                img=vis_img,
                enabled=self.log_to_wandb,
            )
