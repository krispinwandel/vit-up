from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from .base import UpsamplerBase


class UpLiftUpsampler(UpsamplerBase):
    def __init__(
        self,
        dino_version: int,
        local_repo: str = "~/.cache/torch/hub/mwalmer-umd_UPLiFT_main",
        force_local: bool = False,
        name: str = "uplift",
        iter_values: Optional[List[int]] = None,
        fast: bool = True,
        compile: bool = False,
        **kwargs,
    ):
        super().__init__(name=name)
        self.local_repo = str(Path(local_repo).expanduser())
        self.force_local = force_local
        self._device: Optional[torch.device] = None

        if int(dino_version) == 2:
            uplift_hub_entrypoint = "uplift_dinov2_s14"
        elif int(dino_version) == 3:
            uplift_hub_entrypoint = "uplift_dinov3_splus16"
        else:
            raise ValueError("dino_version must be 2 or 3. " f"Got: {dino_version}.")
        if iter_values is None:
            iter_values = [1, 2, 3, 4] if force_local else [1, 2, 3, 4, 5]

        load_kwargs = {
            "no_transform": True,
            "auto_resize": False,
            "return_base_feat": False,
        }
        if fast:
            load_kwargs["fast"] = True

        self.model_uplift: dict[int, Any] = {}
        if not force_local:
            for num_iter in iter_values:
                self.model_uplift[num_iter] = torch.hub.load(
                    "mwalmer-umd/UPLiFT",
                    uplift_hub_entrypoint,
                    iters=num_iter,
                    **load_kwargs,
                ).eval()
        else:
            if not Path(self.local_repo).exists():
                raise FileNotFoundError(
                    "UPLiFT local repo not found. "
                    f"Got: {self.local_repo}. "
                    "Set model.local_repo to a valid path or disable force_local."
                )
            for num_iter in iter_values:
                self.model_uplift[num_iter] = torch.hub.load(
                    self.local_repo,
                    uplift_hub_entrypoint,
                    iters=num_iter,
                    source="local",
                    **load_kwargs,
                ).eval()

        if compile and not fast:
            for model in self.model_uplift.values():
                if hasattr(model, "compile"):
                    try:
                        model.compile(
                            backend="inductor",
                            mode="default",
                            fullgraph=True,
                            dynamic=True,
                        )
                    except Exception:
                        pass

        self.patch_size = self.model_uplift[
            iter_values[-1]
        ].extractor.model.patch_embed.proj.kernel_size[0]

    def forward(
        self,
        pixel_values_bchw: torch.Tensor,
        output_size: int,
        input_size: Optional[int] = None,
        layer_hidden_states_bhwc: Optional[List[torch.Tensor]] = None,
        cache_data: Optional[Any] = None,
    ) -> torch.Tensor:

        device = pixel_values_bchw.device
        if self._device != device:
            for model in self.model_uplift.values():
                model.to(device)
            self._device = device

        in_h, in_w = (
            self._normalize_img_size(input_size)
            if input_size is not None
            else tuple(pixel_values_bchw.shape[-2:])
        )
        h_embd = in_h / self.patch_size
        out_h, _ = self._normalize_img_size(output_size)
        ratio = out_h / float(h_embd)
        num_iters = int(math.ceil(math.log(ratio, 2))) if ratio > 0 else 0

        available = sorted(self.model_uplift.keys())
        if num_iters not in self.model_uplift:
            raise ValueError(
                f"UPLiFT model for num_iters={num_iters} is not loaded. "
                f"Loaded iters: {available}"
            )
        model_to_use = self.model_uplift[num_iters]

        # print("num_iters:", num_iters)

        with torch.no_grad():
            feats_out = model_to_use(
                self._maybe_resize_pixel_values(pixel_values_bchw, input_size)
            )

        # Uplift may not produce the exact output size,
        # so we do a final interpolation to ensure the output size is correct.
        if feats_out.shape[-2:] != (out_h, out_h):
            print(
                f"Warning: UPLiFT output size {feats_out.shape[-2:]} does not match "
                f"target output size {(out_h, out_h)}. "
                f"Interpolating to target size."
            )
            feats_out = F.interpolate(
                feats_out,
                size=(out_h, out_h),
                mode="bilinear",
                align_corners=False,
            )
        feats_out = feats_out.permute(0, 2, 3, 1).contiguous()  # bchw -> bhwc
        return feats_out
