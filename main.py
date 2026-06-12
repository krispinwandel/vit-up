import os
import resource
import psutil
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningDataModule, LightningModule


def _configure_host_ram_limit_from_env() -> None:
    cap_percent_raw = os.getenv("NF_DINO_RAM_CAP_PERCENT", "0")
    try:
        cap_percent = float(cap_percent_raw)
    except ValueError:
        print(
            f"[main.py] Ignoring invalid NF_DINO_RAM_CAP_PERCENT={cap_percent_raw!r}."
        )
        return

    if cap_percent <= 0:
        print("[main.py] Host RAM cap disabled (NF_DINO_RAM_CAP_PERCENT <= 0).")
        return

    vm = psutil.virtual_memory()
    cap_bytes = int(vm.available * (cap_percent / 100.0))
    min_cap_bytes = 2 * 1024**3
    cap_bytes = max(cap_bytes, min_cap_bytes)

    try:
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        target_soft = cap_bytes
        if current_soft not in (-1, resource.RLIM_INFINITY):
            target_soft = min(current_soft, target_soft)

        target_hard = current_hard
        if current_hard in (-1, resource.RLIM_INFINITY):
            target_hard = cap_bytes
        else:
            target_hard = min(current_hard, cap_bytes)

        resource.setrlimit(resource.RLIMIT_AS, (target_soft, target_hard))
        print(
            "[main.py] Applied process host RAM cap: "
            f"{target_soft / 1024**3:.2f} GiB ({cap_percent:.1f}% of currently available RAM)."
        )
    except Exception as exc:
        print(f"[main.py] Failed to apply RAM cap via RLIMIT_AS: {exc}")


def main():
    """
    Example usage:
    python main.py fit --config configs/runs/dinov3_splus.yaml
    """
    _configure_host_ram_limit_from_env()
    torch.set_float32_matmul_precision("medium")
    config_dir = os.path.join(os.path.dirname(__file__), "configs", "defaults")
    default_config_files = [
        os.path.join(config_dir, "seed.yaml"),
        os.path.join(config_dir, "data.yaml"),
        os.path.join(config_dir, "trainer.yaml"),
        os.path.join(config_dir, "callbacks.yaml"),
        os.path.join(config_dir, "logger.yaml"),
        os.path.join(config_dir, "model.yaml"),
    ]

    LightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        load_from_checkpoint_support=True,  # Allows loading from checkpoints via the CLI
        save_config_kwargs={
            "overwrite": True
        },  # Allows overwriting config.yaml on save
        # This prevents the CLI from automatically calling fit() if you just want to instantiate
        # run=False,
        # Allows you to pass configurations via YAML easily
        parser_kwargs={
            "parser_mode": "omegaconf",
            "default_config_files": default_config_files,
        },
    )


if __name__ == "__main__":
    main()
