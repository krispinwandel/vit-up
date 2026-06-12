"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from datetime import datetime
import json
import os
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from vit_up.eval_kits.geometric_correspondence_toolkit.evals.datasets.builder import (
    build_loader,
)
from vit_up.eval_kits.geometric_correspondence_toolkit.evals.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_visible_diamond_xyz,
    estimate_correspondence_xyz,
    project_3dto2d,
)
from vit_up.eval_kits.geometric_correspondence_toolkit.evals.utils.transformations import (
    so3_rotation_angle,
    transform_points_Rt,
)


def _dense_features_to_bchw(features: Any) -> torch.Tensor:
    if isinstance(features, (list, tuple)):
        features = torch.cat(features, dim=-1)
    if not torch.is_tensor(features):
        raise TypeError(f"Expected dense feature tensor, got {type(features)!r}.")
    if features.ndim != 4:
        raise ValueError(f"Expected dense features as 4D tensor, got {features.shape}.")

    # nf_dino wrappers return B,H,W,C. Probe3D matching expects B,C,H,W.
    # if (
    #     features.shape[-1] > features.shape[1]
    #     and features.shape[-1] > features.shape[2]
    # ):
    #     return features.permute(0, 3, 1, 2).contiguous()
    # if features.shape[1] > features.shape[2] and features.shape[1] > features.shape[3]:
    #     return features.contiguous()
    # raise ValueError(
    #     "Could not infer dense feature layout. Expected channel-last B,H,W,C from "
    #     f"nf_dino wrappers or channel-first B,C,H,W, got {tuple(features.shape)}."
    # )
    return features.permute(0, 3, 1, 2).contiguous()


def _extract_dense_features(
    model: torch.nn.Module,
    image: torch.Tensor,
    input_size: int | tuple[int, int],
    output_size: int | tuple[int, int],
) -> torch.Tensor:
    with torch.inference_mode():
        features = model(
            pixel_values_bchw=image,
            input_size=input_size,
            output_size=output_size,
        )
    return _dense_features_to_bchw(features)


def _stringify_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_keys(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(_stringify_keys(data), f, indent=2)


def _compute_error_tensors(
    c_xyz0: torch.Tensor,
    c_xyz1: torch.Tensor,
    c_uv0: torch.Tensor,
    c_uv1: torch.Tensor,
    Rt_01: torch.Tensor,
    intrinsics_1: torch.Tensor,
    scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    c_uv0 = c_uv0 / scale_factor
    c_uv1 = c_uv1 / scale_factor

    c_xyz0in1 = transform_points_Rt(c_xyz0, Rt_01)
    c_err3d = (c_xyz0in1 - c_xyz1).norm(p=2, dim=1)

    c_xyz1in1_uv = project_3dto2d(c_xyz1, intrinsics_1)
    c_xyz0in1_uv = project_3dto2d(c_xyz0in1, intrinsics_1)
    c_err2d = (c_xyz0in1_uv - c_xyz1in1_uv).norm(p=2, dim=1)
    return c_err3d, c_err2d


def _pad_errors(err: torch.Tensor, num_corr: int) -> torch.Tensor:
    if err.numel() >= num_corr:
        return err[:num_corr]
    return nn_F.pad(err, (0, num_corr - err.numel()), value=float("inf"))


def _summarize_navi_errors(
    results: dict[str, float],
    err_3d: torch.Tensor,
    err_2d: torch.Tensor,
    Rt_gt: torch.Tensor,
    prefix: str = "",
) -> None:
    metric_thresh = [0.01, 0.02, 0.05]
    for _th in metric_thresh:
        recall_i = 100 * (err_3d < _th).float().mean()
        label = f"{prefix}recall_{_th:.2f}m"
        print(f"{label}:  {recall_i:.2f}")
        results[label] = float(recall_i)

    px_thresh = [5, 25, 50]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        label = f"{prefix}recall_{_th}px"
        print(f"{label}:  {recall_i:.2f}")
        results[label] = float(recall_i)

    rel_ang = so3_rotation_angle(Rt_gt[:, :3, :3])
    rel_ang = rel_ang * 180.0 / np.pi
    rec_2cm = (err_3d < 0.02).float().mean(dim=1)
    angle_bins = [0, 30, 60, 90, 120]
    bin_rec = compute_binned_performance(rec_2cm, rel_ang, angle_bins)
    for idx, bin_acc in enumerate(bin_rec):
        results[
            f"{prefix}recall_2cm_angle_{angle_bins[idx]}_{angle_bins[idx + 1]}"
        ] = float(bin_acc * 100)


@hydra.main(
    config_path="../config/geometric_correspondence",
    config_name="navi_correspondence",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    torch.manual_seed(int(cfg.random_seed))
    np.random.seed(int(cfg.random_seed))

    # ===== Get model and dataset ====
    device = str(cfg.get("device", "cuda"))
    run_dir = HydraConfig.get().run.dir
    model = instantiate(cfg.model).eval().to(device)
    loader = build_loader(
        cfg.dataset,
        "test",
        int(cfg.batch_size),
        1,
        pair_dataset=True,
    )
    _ = loader.dataset.__getitem__(0)

    # extract features
    feats_0 = []
    feats_1 = []
    xyz_grid_0 = []
    xyz_grid_1 = []
    masks_0 = []
    masks_1 = []
    Rt_gt = []
    intrinsics = []

    for batch in tqdm(loader):
        image_0 = batch["image_0"].to(device, non_blocking=True)
        image_1 = batch["image_1"].to(device, non_blocking=True)
        feat_0 = _extract_dense_features(
            model,
            image_0,
            input_size=cfg.input_size,
            output_size=cfg.output_size,
        )
        feat_1 = _extract_dense_features(
            model,
            image_1,
            input_size=cfg.input_size,
            output_size=cfg.output_size,
        )
        feats_0.append(feat_0.detach().cpu())
        feats_1.append(feat_1.detach().cpu())
        Rt_gt.append(batch["Rt_01"])
        intrinsics.append(batch["intrinsics_1"])

        # scale down to avoid a huge matching problem
        xyz_grid_0_i = nn_F.interpolate(
            batch["xyz_grid_0"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_1_i = nn_F.interpolate(
            batch["xyz_grid_1"], scale_factor=cfg.scale_factor, mode="nearest"
        )
        xyz_grid_0.append(xyz_grid_0_i)
        xyz_grid_1.append(xyz_grid_1_i)
        masks_0.append(
            nn_F.interpolate(
                batch["mask_0"].float(), scale_factor=cfg.scale_factor, mode="nearest"
            )
        )
        masks_1.append(
            nn_F.interpolate(
                batch["mask_1"].float(), scale_factor=cfg.scale_factor, mode="nearest"
            )
        )

    feats_0 = torch.cat(feats_0, dim=0)
    feats_1 = torch.cat(feats_1, dim=0)
    xyz_grid_0 = torch.cat(xyz_grid_0, dim=0)
    xyz_grid_1 = torch.cat(xyz_grid_1, dim=0)
    masks_0 = torch.cat(masks_0, dim=0)
    masks_1 = torch.cat(masks_1, dim=0)
    Rt_gt = torch.cat(Rt_gt, dim=0).float()[:, :3, :4]
    intrinsics = torch.cat(intrinsics, dim=0).float()

    num_instances = len(loader.dataset)
    err_3d = []
    err_2d = []
    err_3d_visible_diamond = []
    err_2d_visible_diamond = []
    for i in tqdm(range(num_instances)):
        with torch.inference_mode():
            c_xyz0, c_xyz1, c_dist, c_uv0, c_uv1 = estimate_correspondence_xyz(
                feats_0[i].to(device),
                feats_1[i].to(device),
                xyz_grid_0[i].to(device),
                xyz_grid_1[i].to(device),
                cfg.num_corr,
            )
            (
                vd_xyz0,
                vd_xyz1,
                vd_dist,
                vd_uv0,
                vd_uv1,
            ) = estimate_correspondence_visible_diamond_xyz(
                feats_0[i].to(device),
                feats_1[i].to(device),
                xyz_grid_0[i].to(device),
                xyz_grid_1[i].to(device),
                masks_0[i].to(device),
                masks_1[i].to(device),
                Rt_gt[i].float().to(device),
                intrinsics[i].to(device),
                scale_factor=float(cfg.scale_factor),
                num_corr=int(cfg.num_corr),
                diamond_steps=list(cfg.visible_diamond_metric.diamond_steps),
                visibility_depth_tolerance_m=float(
                    cfg.visible_diamond_metric.visibility_depth_tolerance_m
                ),
            )

        c_err3d, c_err2d = _compute_error_tensors(
            c_xyz0,
            c_xyz1,
            c_uv0,
            c_uv1,
            Rt_gt[i].float().to(device),
            intrinsics[i].to(device),
            float(cfg.scale_factor),
        )

        err_3d.append(_pad_errors(c_err3d.detach().cpu(), int(cfg.num_corr)))
        err_2d.append(_pad_errors(c_err2d.detach().cpu(), int(cfg.num_corr)))

        vd_err3d, vd_err2d = _compute_error_tensors(
            vd_xyz0,
            vd_xyz1,
            vd_uv0,
            vd_uv1,
            Rt_gt[i].float().to(device),
            intrinsics[i].to(device),
            float(cfg.scale_factor),
        )
        err_3d_visible_diamond.append(
            _pad_errors(vd_err3d.detach().cpu(), int(cfg.num_corr))
        )
        err_2d_visible_diamond.append(
            _pad_errors(vd_err2d.detach().cpu(), int(cfg.num_corr))
        )

    err_3d = torch.stack(err_3d, dim=0).float()
    err_2d = torch.stack(err_2d, dim=0).float()
    err_3d_visible_diamond = torch.stack(err_3d_visible_diamond, dim=0).float()
    err_2d_visible_diamond = torch.stack(err_2d_visible_diamond, dim=0).float()
    results = {}

    _summarize_navi_errors(results, err_3d, err_2d, Rt_gt)
    _summarize_navi_errors(
        results,
        err_3d_visible_diamond,
        err_2d_visible_diamond,
        Rt_gt,
        prefix="visible_diamond_",
    )

    # # result summary
    time = datetime.now().strftime("%d%m%Y-%H%M")
    model_name = str(cfg.model.get("name", model.__class__.__name__))
    dset = loader.dataset.name
    result_values = ", ".join(f"{value:5.02f}" for value in results.values())
    exp_info = ", ".join(
        [
            f"{model_name:30s}",
            str(cfg.input_size),
            str(cfg.output_size),
            str(cfg.num_corr),
            str(cfg.scale_factor),
        ]
    )
    log = f"{time}, {exp_info}, {dset}, {result_values} \n"
    with open(os.path.join(run_dir, "navi_correspondence.log"), "a") as f:
        f.write(log)
    _write_json(
        os.path.join(run_dir, "navi_correspondence_summary.json"),
        {
            "dataset": dset,
            "model": model_name,
            "num_instances": num_instances,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
