import os
import json
import torch
import math
from dataclasses import dataclass
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

from .data_kit import data_splits
from . import metrics


from vit_up.utils import pil_img_utils


@dataclass
class EvalConfig:
    eval_id: str
    dataset_name: str
    dataset_dir: str
    cache_dir: str
    save_dir: str
    img_size: int
    out_size: int


def _remove_batch_dim(infer_res: torch.Tensor) -> torch.Tensor:
    if infer_res.ndim != 4:
        raise ValueError(
            "Expected infer_res with 4 dims (BHWC). "
            f"Got shape={tuple(infer_res.shape)}"
        )

    # Expect BHWC tensors only.
    _, h, w, c = infer_res.shape
    if h != w:
        raise ValueError(
            "Expected BHWC infer_res with square spatial dims. "
            f"Got shape={tuple(infer_res.shape)}"
        )
    return infer_res[0]


def build_oracle_fun(img_fts_cache, device="cuda"):
    def oracle_fun(img1_fp, img2_fp, img1_kps, h1, w1, h2, w2):
        img1_fts_hwc = img_fts_cache[img1_fp].to(device)
        img2_fts_hwc = img_fts_cache[img2_fp].to(device)
        ft_size = int(img2_fts_hwc.shape[0])
        img1_px, img1_py = pil_img_utils.get_square_paddings(w1, h1)
        img2_px, img2_py = pil_img_utils.get_square_paddings(w2, h2)
        img1_max_size = max(h1, w1)
        img2_max_size = max(h2, w2)
        preds = []
        cos_sims = []
        for i in range(img1_kps.shape[0]):
            # print("kp_xy", kp_xy)
            kp_xy = img1_kps[i]
            kp_xy_pad = (
                kp_xy[0] + img1_px,
                kp_xy[1] + img1_py,
            )
            kp_xy_pad_normalized = (
                kp_xy_pad[0] / img1_max_size,
                kp_xy_pad[1] / img1_max_size,
            )
            qxy_normalized = torch.tensor(
                kp_xy_pad_normalized, device=img1_fts_hwc.device, dtype=torch.float32
            )
            query_coords_xy = (qxy_normalized * ft_size).long()
            query_ft = img1_fts_hwc[query_coords_xy[1], query_coords_xy[0]]
            cos_sim = F.cosine_similarity(query_ft[None, None, :], img2_fts_hwc, dim=-1)
            cos_sims.append(cos_sim.cpu())
            # get argmax
            flat_idx = torch.argmax(cos_sim)
            best_y = flat_idx // ft_size
            best_x = flat_idx % ft_size
            # Use the center of the matched feature cell when mapping to pixels.
            best_x = int(((best_x + 0.5) / ft_size * img2_max_size)) - img2_px
            best_y = int(((best_y + 0.5) / ft_size * img2_max_size)) - img2_py
            # print(best_x, best_y)
            preds.append((best_x, best_y))
        return np.array(preds), cos_sims

    return oracle_fun


def eval_category(
    category: str,
    eval_config: EvalConfig,
    inference_fun_dict,
    prepare_inputs_dict,
    device="cuda",
    save_results=True,
):
    img_file_splits, img_file_pairs_split = data_splits.load_img_file_splits(
        eval_config.dataset_name,
        eval_config.dataset_dir,
        category,
        eval_config.cache_dir,
    )

    model_keys = inference_fun_dict.keys()
    caches = {}
    for model_key in model_keys:
        caches[model_key] = {}
        prepare_input = prepare_inputs_dict[model_key]
        infer_fun = inference_fun_dict[model_key]
        for img_file in tqdm(img_file_splits["test"]):
            img_pil = Image.open(img_file)
            img_pil_padded = pil_img_utils.pad_image_to_square(img_pil)
            img_inputs = prepare_input(img_pil_padded)
            infer_res = infer_fun(**img_inputs)
            caches[model_key][img_file] = _remove_batch_dim(infer_res).cpu()

    eval_res = {}
    for model_key in model_keys:
        (
            pck_statistics,
            pck_pair_statistics,
            pck_plus_statistics,
            pck_plus_pair_statistics,
            kap_statistics,
            kap_labels_scores,
            debug_data,
        ) = metrics.compute_correspondence_2d_metrics(
            img_file_pairs_split["test"],
            oracle_fun=build_oracle_fun(caches[model_key], device=device),
            alpha_levels=[0.1, 0.05, 0.01],
            compute_mAP=False,
        )
        eval_res[model_key] = {
            "pck_statistics": pck_statistics,
            "pck_pair_statistics": pck_pair_statistics,
            "pck_plus_statistics": pck_plus_statistics,
            "pck_plus_pair_statistics": pck_plus_pair_statistics,
            "kap_statistics": kap_statistics,
            "kap_labels_scores": kap_labels_scores,
            "debug_data": debug_data,
        }
        if save_results:
            out_dir = os.path.join(
                eval_config.save_dir, eval_config.eval_id, category, model_key
            )
            os.makedirs(out_dir, exist_ok=True)
            for eval_key in eval_res[model_key].keys():
                if eval_key in ["debug_data"]:
                    continue
                out_fn = os.path.join(out_dir, f"{eval_key}.json")
                with open(out_fn, "w") as f:
                    json.dump(eval_res[model_key][eval_key], f)
    return {"eval_res": eval_res, "caches": caches}
