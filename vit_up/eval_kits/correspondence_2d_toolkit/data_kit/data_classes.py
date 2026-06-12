from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class ImgAnnoNormalized:
    bbox: List[int]
    kp_ids: List[int]
    kp_xy: List[List[int]]
    filename: str
    img_width: int
    img_height: int
    category: str
    supercategory: str
    rel_fp: str


@dataclass
class SrcTrgImgAnno:
    src: ImgAnnoNormalized
    trg: ImgAnnoNormalized
