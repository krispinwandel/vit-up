# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from google-research/lila (probes/linear.py, probes/attention.py):
# the decoder branch is optional (encoder-only baselines), trainable parameters
# are exposed as `trainable_parameters()` instead of overriding `parameters()`,
# and the probe widths are passed explicitly instead of through the LILA config.
"""Two-branch probes of the LILA evaluation protocol.

Every probe sums logits from the low-resolution encoder (backbone) tokens and
from the dense decoder features, each upsampled to the target resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size, mode="bilinear", align_corners=False)


class LinearCat(nn.Module):
    """Sum of a 1x1 conv on encoder tokens and a kxk conv on decoder features."""

    def __init__(self, idim_enc: int, idim_dec: int | None, odim: int, alpha_enc: float = 1.0,
                 alpha_dec: float = 1.0, kernel_size: int = 5, padding: int = 2, with_bn: bool = False):
        super().__init__()
        self.alpha_enc = alpha_enc
        self.alpha_dec = alpha_dec if idim_dec is not None else 0.0
        self.probe_enc = nn.Conv2d(idim_enc, odim, 1)
        if with_bn:
            self.probe_enc = nn.Sequential(nn.BatchNorm2d(idim_enc), self.probe_enc)
        self.probe_dec = (nn.Conv2d(idim_dec, odim, kernel_size, stride=1, padding=padding)
                          if idim_dec is not None else None)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, feats_enc, feats_dec, scale_to):
        logits = self.alpha_enc * _resize(self.probe_enc(feats_enc), scale_to)
        if self.probe_dec is not None and self.alpha_dec != 0:
            logits_dec = self.probe_dec(feats_dec)
            if logits_dec.shape[-2:] != tuple(scale_to):
                logits_dec = _resize(logits_dec, scale_to)
            logits = logits + self.alpha_dec * logits_dec
        return logits
