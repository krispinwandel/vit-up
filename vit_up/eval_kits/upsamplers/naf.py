from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional
import sys

import torch

from vit_up.layers.backbones.dino_vit_base import DinoViTBackboneBase

from .anyup import _contruct_backbone
from .base import UpsamplerBase


def _clear_top_level_src_modules() -> dict[str, object]:
    old_src_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == "src" or module_name.startswith("src.")
    }
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            sys.modules.pop(module_name, None)
    return old_src_modules


@contextmanager
def _naf_src_import_context(repo_dir: str) -> Iterator[None]:
    repo_dir = str(Path(repo_dir).expanduser())
    old_path = list(sys.path)
    old_src_modules = _clear_top_level_src_modules()
    sys.path[:] = [
        path
        for path in old_path
        if not (Path(path or ".").resolve() / "src" / "__init__.py").is_file()
    ]
    sys.path.insert(0, repo_dir)
    try:
        yield
    finally:
        sys.path[:] = old_path
        _clear_top_level_src_modules()
        sys.modules.update(old_src_modules)


def _load_naf_hub_model(
    pretrained: bool,
    local_repo: str,
    force_local: bool,
) -> torch.nn.Module:
    if not force_local:
        repo_dir = Path(torch.hub.get_dir()) / "valeoai_NAF_main"
        if repo_dir.is_dir():
            with _naf_src_import_context(str(repo_dir)):
                return torch.hub.load(
                    str(repo_dir),
                    "naf",
                    pretrained=pretrained,
                    device="cpu",
                    source="local",
                ).eval()
        _clear_top_level_src_modules()
        return torch.hub.load(
            "valeoai/NAF",
            "naf",
            pretrained=pretrained,
            device="cpu",
        ).eval()

    with _naf_src_import_context(local_repo):
        return torch.hub.load(
            local_repo,
            "naf",
            pretrained=pretrained,
            device="cpu",
            source="local",
        ).eval()


def _set_naf_kernel_size(naf: torch.nn.Module, kernel_size: Optional[int]) -> None:
    if kernel_size is None:
        return
    if not hasattr(naf, "upsampler") or not hasattr(naf.upsampler, "kernel_size"):
        raise ValueError("NAF model does not expose upsampler.kernel_size.")
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("NAF kernel size must be a positive odd integer.")
    naf.upsampler.kernel_size = (kernel_size, kernel_size)


class NAFUpsampler(UpsamplerBase):
    def __init__(
        self,
        backbone: Optional[str] = None,
        name: str = "naf",
        pretrained: bool = True,
        local_repo: str = "~/.cache/torch/hub/valeoai_NAF_main",
        force_local: bool = False,
        window_size: int = 0,
        **kwargs,
    ):
        super().__init__(name=name)
        del kwargs

        self.pretrained = bool(pretrained)
        self.local_repo = str(Path(local_repo).expanduser())
        self.force_local = bool(force_local)
        self.window_size = int(window_size)
        self._device: Optional[torch.device] = None

        self.backbone_name = backbone
        self.backbone = None
        if backbone is not None:
            self.backbone = _contruct_backbone(backbone)
            self.backbone.eval()
            self.backbone.requires_grad_(False)
            self.backbone.compile()

        self.upsampler = _load_naf_hub_model(
            pretrained=self.pretrained,
            local_repo=self.local_repo,
            force_local=self.force_local,
        )

    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int | tuple[int, int],
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
    ) -> torch.Tensor:
        del layer_hidden_states_bhwc, cache_data

        out_h, out_w = self._normalize_img_size(output_size)
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output_size must be positive.")
        if self.backbone is None:
            raise ValueError("NAFUpsampler requires a backbone to compute features.")

        device = pixel_values_bchw.device
        if self._device != device:
            self.upsampler.to(device)
            self.backbone.to(device)
            self._device = device

        with torch.no_grad():
            pixel_values_bchw = self._maybe_resize_pixel_values(
                pixel_values_bchw, input_size
            )
            layer_hidden_states_bhwc = (
                DinoViTBackboneBase._compute_backbone_hidden_states(
                    backbone=self.backbone,
                    pixel_values=pixel_values_bchw,
                    img_size=None,
                    window_size=self.window_size,
                )
            )
            lr_features_bchw = (
                layer_hidden_states_bhwc[-1].permute(0, 3, 1, 2).contiguous()
            )
            hr_features_bchw = self.upsampler(
                pixel_values_bchw,
                lr_features_bchw,
                (out_h, out_w),
            )

        if hr_features_bchw.ndim != 4:
            raise ValueError(
                "NAF upsampler returned unexpected shape: "
                f"{tuple(hr_features_bchw.shape)}."
            )
        return hr_features_bchw.permute(0, 2, 3, 1).contiguous()

    def pre_compute_cache(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int,
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
    ):
        del pixel_values_bchw, output_size, input_size, layer_hidden_states_bhwc
        return None
