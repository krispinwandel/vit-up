import json
import os
from dataclasses import asdict
from typing import Any, Dict

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from vit_up.eval_kits.correspondence_2d_toolkit import eval_utils
from vit_up.eval_kits.correspondence_2d_toolkit.data_kit import spair_dataset_utils
from vit_up.utils import img_transforms

ALPHA_LEVELS = [0.1, 0.05, 0.01]


def _build_eval_config(cfg: DictConfig, run_dir: str) -> eval_utils.EvalConfig:
    return eval_utils.EvalConfig(
        eval_id="hydra_run",
        dataset_name="spair-71k",
        dataset_dir=cfg.dataset_dir,
        cache_dir=cfg.cache_dir,
        save_dir=run_dir,
        img_size=int(cfg.img_size),
        out_size=int(cfg.out_size),
    )


def _stringify_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_keys(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize_category_pck(category_result: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    for model_key, result in category_result["eval_res"].items():
        pck_statistics = result["pck_statistics"]
        summary[model_key] = {
            f"pck@{alpha}": float(pck_statistics[alpha]) for alpha in ALPHA_LEVELS
        }
    return summary


def _mean_pck(category_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    model_keys = sorted(
        {
            model_key
            for category_summary in category_summaries.values()
            for model_key in category_summary.keys()
        }
    )
    means = {}
    for model_key in model_keys:
        means[model_key] = {}
        for alpha in ALPHA_LEVELS:
            metric_key = f"pck@{alpha}"
            values = [
                category_summary[model_key][metric_key]
                for category_summary in category_summaries.values()
                if model_key in category_summary
            ]
            means[model_key][metric_key] = float(np.mean(values))
    return means


def _write_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(_stringify_keys(data), f, indent=2)


def _build_prepare_inputs_fn(device: str):
    def build_transform(img_size: int):
        return img_transforms.build_image_transform(img_size, img_size)

    def prepare_inputs(img_square, img_size: int):
        img_to_pixel_values = build_transform(img_size)
        return {"pixel_values": img_to_pixel_values(img_square).to(device).unsqueeze(0)}

    return prepare_inputs


def _build_inference_fn(model, cfg: DictConfig):
    img_size = int(cfg.img_size)
    out_size = int(cfg.out_size)

    def inference(pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out_bhwc = model(
                pixel_values_bchw=pixel_values,
                output_size=out_size,
                input_size=img_size,
            )
        return out_bhwc

    return inference


@hydra.main(
    config_path="../config/correspondence_2d",
    config_name="correspondence_2d",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().run.dir
    os.makedirs(run_dir, exist_ok=True)
    device = "cuda"
    eval_config = _build_eval_config(cfg, run_dir)

    _write_json(os.path.join(run_dir, "eval_config.json"), asdict(eval_config))

    model_key = str(cfg.model.get("name", "model"))
    model = instantiate(cfg.model).eval().to(device)
    inference_fun_dict = {model_key: _build_inference_fn(model, cfg)}
    prepare_inputs_fn = _build_prepare_inputs_fn(device)
    img_size = int(cfg.img_size)
    prepare_inputs_dict = {
        model_key: lambda img_square: prepare_inputs_fn(img_square, img_size)
    }

    category_summaries: Dict[str, Dict[str, Any]] = {}
    for category in spair_dataset_utils.SPAIR_SORTED_CATEGORIES:
        print(f"Evaluating category {category}...")
        category_result = eval_utils.eval_category(
            category=category,
            eval_config=eval_config,
            inference_fun_dict=inference_fun_dict,
            prepare_inputs_dict=prepare_inputs_dict,
            device=device,
            save_results=False,
        )
        category_summaries[category] = _summarize_category_pck(category_result)

    results = {
        "pck_mean": _mean_pck(category_summaries),
        "pck_by_category": category_summaries,
    }
    _write_json(os.path.join(run_dir, "pck_summary.json"), results)
    print(json.dumps(_stringify_keys(results["pck_mean"]), indent=2))


if __name__ == "__main__":
    main()
