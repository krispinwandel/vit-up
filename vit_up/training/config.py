from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class ClassParams:
    class_path: str
    init_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackboneFeatureArgs:
    window_size: int = 0
    gt_img_size: Optional[int] = None
    gt_img_sizes: Optional[List[int]] = None
    lr_img_size: Optional[int] = None


@dataclass
class LossMetricArgs:
    l2_weight: float = 1.0
    cos_weight: float = 1.0
    kl_weight: float = 1.0
    enable_kl_last_only: bool = False
    kl_tau: float = 0.5
    eps: float = 1.0e-5



