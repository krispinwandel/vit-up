import json
import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm import tqdm

from . import spair
from .geometry import SquareImageGeometry
from .matching import CategoryMatcher
from .metrics import compute_pck

ALPHAS = (0.1, 0.05, 0.01)


@dataclass
class EvalConfig:
    dataset_dir: str
    save_dir: str
    img_size: int
    out_size: int
    feature_cache_device: str = "device"
    matching_query_chunk_size: int = 256
    matching_target_chunk_size: int = 16384


def eval_category(category: str, eval_config: EvalConfig, inference, prepare_inputs, device="cuda") -> dict:
    """PCK on the test pairs of one category. Images are zero-padded to squares before inference."""
    image_pairs = spair.load_test_pairs(eval_config.dataset_dir, category)
    images = sorted(set(image_pairs))
    annotations = {path: spair.load_annotation(path) for path in images}
    matcher = CategoryMatcher(
        device=device, cache_device=device if eval_config.feature_cache_device == "device" else eval_config.feature_cache_device,
        query_chunk_size=eval_config.matching_query_chunk_size,
        target_chunk_size=eval_config.matching_target_chunk_size)
    start = time.perf_counter()
    for path in tqdm(images, desc="Extracting image features"):
        annotation = annotations[path]
        with Image.open(path) as image:
            image = SquareImageGeometry(annotation.img_width, annotation.img_height).pad_image(image)
        features = inference(**prepare_inputs(image))
        matcher.add_image(path, features[0], annotation)
    extracted = time.perf_counter()
    predictions = matcher.predict(image_pairs)
    del matcher
    pck = compute_pck(image_pairs, predictions, annotations, ALPHAS)
    print(f"{category}: {len(images)} images, {len(image_pairs) // 2} pairs, "
          f"features {extracted - start:.1f}s, matching {time.perf_counter() - extracted:.1f}s", flush=True)
    return {f"pck@{alpha}": float(pck[alpha]) for alpha in ALPHAS}


def evaluate_spair(eval_config: EvalConfig, inference_fun_dict: dict, prepare_inputs_dict: dict, device="cuda"):
    """Per-category PCK and its macro-average over the 18 SPair-71k categories."""
    os.makedirs(eval_config.save_dir, exist_ok=True)
    by_category = {}
    for category in spair.CATEGORIES:
        print(f"Evaluating category {category}...", flush=True)
        by_category[category] = {
            model: eval_category(category, eval_config, inference, prepare_inputs_dict[model], device)
            for model, inference in inference_fun_dict.items()}
    means = {model: {key: float(np.mean([scores[model][key] for scores in by_category.values()]))
                     for key in (f"pck@{alpha}" for alpha in ALPHAS)}
             for model in inference_fun_dict}
    results = {"pck_mean": means, "pck_by_category": by_category}
    with open(os.path.join(eval_config.save_dir, "pck_summary.json"), "w") as file:
        json.dump(results, file, indent=2)
    return results
