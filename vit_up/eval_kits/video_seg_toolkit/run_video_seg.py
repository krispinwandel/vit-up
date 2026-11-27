"""Run one video segmentation benchmark (LILA protocol) for one eval-kit model (upsampler).

    python -m vit_up.eval_kits.video_seg_toolkit.run_video_seg benchmark=vos_knn model=dinov2_small/vit_up
    python -m vit_up.eval_kits.video_seg_toolkit.run_video_seg benchmark=vos_knn model=none \
        encoder.backbone_model_name=facebook/dinov2-small   # backbone baseline

`metrics.json` and the resolved `config.yaml` are written to the Hydra run directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# Dense features of wide models at high resolution are multi-GB tensors of varying size;
# expandable segments avoid fragmentation-induced out-of-memory errors. Set before CUDA starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from vit_up.eval_kits.upsamplers.vit_up import ViTUpUpsampler
from vit_up.eval_kits.video_seg_toolkit.benchmarks import BENCHMARKS, Context
from vit_up.eval_kits.video_seg_toolkit.features import BackboneEncoder, FeatureExtractor


def _encoder_backbone(cfg: DictConfig, decoder) -> str:
    """Hugging Face backbone of the probes' encoder branch: explicit, or taken from the model."""
    name = cfg.encoder.backbone_model_name
    if name != "auto":
        return name
    if isinstance(decoder, ViTUpUpsampler):
        return decoder.model.config["backbone"]
    if getattr(decoder, "backbone_model_name", None):  # LILA, backbone probes
        return decoder.backbone_model_name
    for key in ("backbone_model_name", "backbone"):  # baselines name their backbone in the config
        value = cfg.model.get(key)
        if isinstance(value, str) and "/" in value:  # a Hugging Face repository, not a timm name
            return value
    raise ValueError("Cannot infer the encoder backbone for this model; set encoder.backbone_model_name.")


@hydra.main(config_path="../config/video_seg", config_name="video_seg", version_base=None)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The video segmentation benchmarks require a CUDA device.")
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    output_dir = Path(HydraConfig.get().run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    decoder = instantiate(cfg.model).to(device) if "_target_" in cfg.model else None
    backbone = _encoder_backbone(cfg, decoder)
    encoder = BackboneEncoder(backbone, cfg.encoder.revision).to(device)
    features = FeatureExtractor(encoder, decoder, cfg.decoder_norm, cfg.decoder_batch_size).to(device)

    choices = HydraConfig.get().runtime.choices
    print(f"{choices.benchmark}: encoder {backbone}, model {choices.model}")
    BENCHMARKS[choices.benchmark](Context(features, cfg.benchmark, output_dir, device, choices.benchmark))


if __name__ == "__main__":
    main()
