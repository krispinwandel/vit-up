from typing import List, Optional, Any
from pathlib import Path
import torch


from vit_up.layers.backbones.vit_wrapper import PretrainedViTWrapper
from vit_up.layers.backbones.radio import RadioWrapper

from .src.jafar.jafar import JAFAR as JAFARCore
from .base import UpsamplerBase


def _infer_feature_dim(backbone_name: str) -> int:
    backbone_name_lc = backbone_name.lower()
    if "radio_v2.5-h" in backbone_name_lc:
        return 1280
    if "radio_v2.5-l" in backbone_name_lc:
        return 1024
    if "radio_v2.5-b" in backbone_name_lc:
        return 768
    if "vit_large" in backbone_name_lc or "vitl" in backbone_name_lc:
        return 1024
    if "vit_base" in backbone_name_lc or "vitb" in backbone_name_lc:
        return 768
    if "vit_small" in backbone_name_lc or "vits" in backbone_name_lc:
        return 384
    raise ValueError(f"Unsupported backbone name: {backbone_name}")


class JAFARUpsampler(UpsamplerBase):
    def __init__(
        self,
        backbone: str,
        name: str = "jafar",
        checkpoint_dir: str | None = None,
        feature_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(name=name)

        self.backbone_name = backbone
        # default to repo-local weights folder: nf_dino/eval_kits/upsamplers/weights/jafar
        if checkpoint_dir is None:
            default_dir = Path(__file__).resolve().parent / "weights" / "jafar"
            self.checkpoint_dir = default_dir
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        if feature_dim is None:
            feature_dim = _infer_feature_dim(backbone)
        self.feature_dim = feature_dim

        backbone_name_lc = backbone.lower()
        if "radio" in backbone_name_lc:
            self.backbone = RadioWrapper(name=backbone)
        else:
            self.backbone = PretrainedViTWrapper(name=backbone)

        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.backbone.compile()

        self.jafar = JAFARCore(
            input_dim=3,
            qk_dim=kwargs.pop("qk_dim", 128),
            v_dim=self.feature_dim,
            feature_dim=self.feature_dim,
            kernel_size=kwargs.pop("kernel_size", 1),
            num_heads=kwargs.pop("num_heads", 4),
            **kwargs,
        )

        checkpoint_path = self.checkpoint_dir / f"{backbone}.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"JAFAR checkpoint not found for backbone '{backbone}': {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "jafar" in checkpoint:
            state_dict = checkpoint["jafar"]
        else:
            state_dict = checkpoint
        self.jafar.load_state_dict(state_dict, strict=False)
        self.jafar.eval()

    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int,
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
    ):

        with torch.no_grad():
            pixel_values_bchw_backbone = self._maybe_resize_pixel_values(
                pixel_values_bchw, input_size
            )
            backbone_out = self.backbone(pixel_values_bchw_backbone)
            if isinstance(backbone_out, tuple):
                features_bchw = backbone_out[0]
            else:
                features_bchw = backbone_out

        output_size_hw = self._normalize_img_size(output_size)
        return (
            self.jafar(pixel_values_bchw, features_bchw, output_size_hw)
            .permute(0, 2, 3, 1)  # (b,c,h,w) => (b,h,w,c)
            .contiguous()
        )
