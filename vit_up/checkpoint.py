"""ViT-Up checkpoints: a directory with `model.safetensors` and `config.yaml`.

`model.safetensors` holds the ViT-Up weights (`vit_up.*`) and the backbone's LoRA
adapters (`backbone.*`), if any. The frozen backbone itself is loaded from the
Hugging Face Hub. `config.yaml` records everything needed to rebuild the model:

    backbone: facebook/dinov3-vits16plus-pretrain-lvd1689m
    backbone_revision: <commit>
    lora: {r: 16, lora_alpha: 32, ...}   # or null
    image_size: 448                      # backbone input resolution
    vit_up: {_target_: vit_up.model.ViTUp, dim: 384, ...}
"""
from pathlib import Path

import yaml
from hydra.utils import instantiate
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch import nn

from vit_up.model.backbones import lora_state_dict, load_backbone, load_lora_state_dict

HF_REPO = "Krispin/vit-up"
WEIGHTS, CONFIG = "model.safetensors", "config.yaml"

RELEASES = {
    "dinov3_splus": "DINOv3-S+, Stage 2 (main model)",
    "dinov3_splus_stage1": "DINOv3-S+, Stage 1",
    "dinov3_splus_g28_56": "DINOv3-S+, Stage 2, trained on 28x28 and 56x56 backbone grids (448 and 896 px)",
    "dinov3_base": "DINOv3-B, Stage 2",
    "dinov3_base_stage1": "DINOv3-B, Stage 1",
    "dinov2_small": "DINOv2-S, Stage 2",
}


def resolve(checkpoint: str | Path) -> Path:
    """Return the local directory of a checkpoint given as a path or a release name."""
    path = Path(checkpoint)
    if path.name == WEIGHTS:
        path = path.parent
    if str(checkpoint) in RELEASES and not path.exists():
        path = Path(snapshot_download(HF_REPO, allow_patterns=f"{checkpoint}/*")) / str(checkpoint)
    if not (path / WEIGHTS).is_file():
        raise FileNotFoundError(f"No checkpoint at {checkpoint!r}; releases are {sorted(RELEASES)}.")
    return path


def load_config(checkpoint: str | Path) -> dict:
    return yaml.safe_load((resolve(checkpoint) / CONFIG).read_text())


def build(config: dict, load_lora: bool = True) -> tuple[nn.Module, nn.Module]:
    """Instantiate the architecture of a checkpoint config: (vit_up, backbone)."""
    lora = config["lora"] if load_lora else None
    return instantiate(config["vit_up"]), load_backbone(config["backbone"], lora, config["backbone_revision"])


def load_weights(checkpoint: str | Path, vit_up: nn.Module, backbone: nn.Module) -> None:
    state = load_file(resolve(checkpoint) / WEIGHTS)
    vit_up.load_state_dict({k.removeprefix("vit_up."): v for k, v in state.items() if k.startswith("vit_up.")})
    load_lora_state_dict(backbone, {k.removeprefix("backbone."): v for k, v in state.items()
                                    if k.startswith("backbone.")})


def load(checkpoint: str | Path) -> tuple[nn.Module, nn.Module, dict]:
    """Build a checkpoint's model and load its weights: (vit_up, backbone, config)."""
    config = load_config(checkpoint)
    vit_up, backbone = build(config)
    load_weights(checkpoint, vit_up, backbone)
    return vit_up, backbone, config


def save(directory: str | Path, vit_up: nn.Module, backbone: nn.Module, config: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {f"vit_up.{k}": v for k, v in vit_up.state_dict().items()}
    state |= {f"backbone.{k}": v for k, v in lora_state_dict(backbone).items()}
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, directory / WEIGHTS)
    (directory / CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))
