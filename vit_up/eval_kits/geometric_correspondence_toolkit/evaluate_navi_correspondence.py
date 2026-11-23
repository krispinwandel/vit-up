"""NAVI geometric correspondence (Probe3D protocol), adapted from Mohamed El Banani's MIT-licensed code."""
import json
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from vit_up.eval_kits.geometric_correspondence_toolkit.evals.datasets.builder import build_loader
from vit_up.eval_kits.geometric_correspondence_toolkit.evals.utils.correspondence import (
    compute_binned_performance, estimate_correspondence_xyz, project_3dto2d)
from vit_up.eval_kits.geometric_correspondence_toolkit.evals.utils.transformations import (
    so3_rotation_angle, transform_points_Rt)


def correspondence_errors(xyz0, xyz1, Rt_01, intrinsics_1):
    """3D (m) and reprojected 2D (px) errors of correspondences, given the relative pose."""
    xyz0_in_1 = transform_points_Rt(xyz0, Rt_01)
    error_3d = (xyz0_in_1 - xyz1).norm(p=2, dim=1)
    error_2d = (project_3dto2d(xyz0_in_1, intrinsics_1) - project_3dto2d(xyz1, intrinsics_1)).norm(p=2, dim=1)
    return error_3d, error_2d


def pad_errors(errors, num_corr):
    """Pairs with fewer valid correspondences count the missing ones as failures."""
    return errors[:num_corr] if errors.numel() >= num_corr else F.pad(errors, (0, num_corr - errors.numel()),
                                                                       value=float("inf"))


def summarize(errors_3d, errors_2d, Rt):
    results = {f"recall_{t:.2f}m": float(100 * (errors_3d < t).float().mean()) for t in (0.01, 0.02, 0.05)}
    results |= {f"recall_{t}px": float(100 * (errors_2d < t).float().mean()) for t in (5, 25, 50)}
    angles = so3_rotation_angle(Rt[:, :3, :3]) * 180.0 / np.pi
    bins = [0, 30, 60, 90, 120]
    for i, recall in enumerate(compute_binned_performance((errors_3d < 0.02).float().mean(dim=1), angles, bins)):
        results[f"recall_2cm_angle_{bins[i]}_{bins[i + 1]}"] = float(recall * 100)
    return results


@hydra.main(config_path="../config/geometric_correspondence", config_name="navi_correspondence", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    device = cfg.device
    model = instantiate(cfg.model).eval().to(device)
    loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1, pair_dataset=True)

    errors_3d, errors_2d, poses = [], [], []
    for batch in tqdm(loader):
        with torch.inference_mode():
            features = [model(batch[f"image_{view}"].to(device), output_size=cfg.output_size,
                              input_size=cfg.input_size).permute(0, 3, 1, 2) for view in (0, 1)]
        Rt = batch["Rt_01"].float()[:, :3, :4].to(device)
        poses.append(Rt.cpu())
        intrinsics = batch["intrinsics_1"].float().to(device)
        xyz = [F.interpolate(batch[f"xyz_grid_{view}"].to(device), scale_factor=cfg.scale_factor, mode="nearest")
               for view in (0, 1)]
        for i in range(len(Rt)):
            with torch.inference_mode():
                xyz0, xyz1, _, _, _ = estimate_correspondence_xyz(features[0][i], features[1][i], xyz[0][i], xyz[1][i],
                                                                  cfg.num_corr)
            error_3d, error_2d = correspondence_errors(xyz0, xyz1, Rt[i], intrinsics[i])
            errors_3d.append(pad_errors(error_3d.cpu(), cfg.num_corr))
            errors_2d.append(pad_errors(error_2d.cpu(), cfg.num_corr))

    results = summarize(torch.stack(errors_3d).float(), torch.stack(errors_2d).float(), torch.cat(poses))
    print(json.dumps(results, indent=2))
    summary = {"model": cfg.model.name, "num_pairs": len(loader.dataset), "results": results}
    with open(os.path.join(HydraConfig.get().run.dir, "navi_correspondence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
