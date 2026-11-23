"""SPair-71k test pairs and keypoint annotations."""
import glob
import json
import os
from dataclasses import dataclass

CATEGORIES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog", "horse",
              "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor"]


@dataclass
class Annotation:
    bbox: list[int]
    kp_ids: list[int]
    kp_xy: list[list[int]]
    img_width: int
    img_height: int


def load_annotation(image_path: str) -> Annotation:
    with open(image_path.replace("JPEGImages", "ImageAnnotation").replace(".jpg", ".json")) as f:
        annotation = json.load(f)
    kps = annotation["kps"]
    return Annotation(bbox=annotation["bndbox"], kp_ids=[int(k) for k, v in kps.items() if v is not None],
                      kp_xy=[v for v in kps.values() if v is not None], img_width=annotation["image_width"],
                      img_height=annotation["image_height"])


def load_test_pairs(dataset_dir: str, category: str) -> list[str]:
    """Alternating source and target image paths of a category's test pairs."""
    paths = []
    for pair_path in sorted(glob.glob(os.path.join(dataset_dir, "PairAnnotation", "test", f"*:{category}.json"))):
        with open(pair_path) as f:
            pair = json.load(f)
        paths += [os.path.join(dataset_dir, "JPEGImages", category, pair[key]) for key in ("src_imname", "trg_imname")]
    return paths


def pck_threshold(annotation: Annotation) -> float:
    """The longer side of the object bounding box."""
    x0, y0, x1, y1 = annotation.bbox
    return max(x1 - x0, y1 - y0)
