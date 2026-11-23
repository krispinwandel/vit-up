"""SPair-71k keypoint transfer (PCK) by nearest-neighbor matching of dense features."""
import json
import os
from dataclasses import asdict

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from vit_up.data.transforms import image_transform
from vit_up.eval_kits.correspondence_2d_toolkit import eval_utils


@hydra.main(config_path="../config/correspondence_2d", config_name="correspondence_2d", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().run.dir
    device = "cuda"
    eval_config = eval_utils.EvalConfig(
        dataset_dir=cfg.dataset_dir, save_dir=run_dir, img_size=cfg.img_size, out_size=cfg.out_size, feature_cache_device=cfg.feature_cache_device,
        matching_query_chunk_size=cfg.matching_query_chunk_size,
        matching_target_chunk_size=cfg.matching_target_chunk_size)
    with open(os.path.join(run_dir, "eval_config.json"), "w") as file:
        json.dump(asdict(eval_config), file, indent=2)

    model = instantiate(cfg.model).eval().to(device)
    transform = image_transform(cfg.img_size)

    def prepare_inputs(image):
        return {"images": transform(image).to(device).unsqueeze(0)}

    @torch.no_grad()
    def inference(images):
        return model(images, output_size=cfg.out_size, input_size=cfg.img_size)

    results = eval_utils.evaluate_spair(eval_config, {cfg.model.name: inference}, {cfg.model.name: prepare_inputs},
                                        device)
    print(json.dumps(results["pck_mean"], indent=2))


if __name__ == "__main__":
    main()
