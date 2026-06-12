"""Evaluation utilities for 3D semantic alignment."""

import numpy as np
import os
from typing import Dict, Callable, List
from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score

from .data_kit import raw_data_utils
from .utils import image_processing


def compute_label_score(
    sim: torch.Tensor,  # shape: (n_img1_kps, h2, w2)
    img1_kp_ids: List[int],
    img2_kp_xy: torch.Tensor,
    img2_kp_ids: List[int],
    threshold: float,
    alpha_levels: List[float] = [0.1, 0.05, 0.01],
):
    # https://github.com/VICO-UoE/SphericalMaps/blob/main/pck_spair_pascal_sphere.py#L198
    #  => compute_score
    n_img1_kps, h2, w2 = sim.shape
    img_coords = image_processing.create_coordinate_tensor(
        h2, w2, device=sim.device
    ).float()  # (h2*w2, 2)

    img_coords_flat = img_coords.view(-1, 2)
    img2_kp_dists = torch.cdist(
        img2_kp_xy.float(), img_coords_flat, p=2
    )  # (n_img2_kps, h2*w2)

    labels_scores = {alpha: {"labels": [], "scores": []} for alpha in alpha_levels}
    kp_id_2_index = {kp_id: i for i, kp_id in enumerate(img2_kp_ids)}

    sim_hat = (sim - sim.min()) / (sim.max() - sim.min())
    for alpha in alpha_levels:
        for i in range(n_img1_kps):
            img1_kp_idx = img1_kp_ids[i]
            s_map = sim_hat[i].view(-1)
            if img1_kp_idx in kp_id_2_index:
                j = kp_id_2_index[img1_kp_idx]
                thresh_alpha = alpha * threshold
                in_radius = img2_kp_dists[j] < thresh_alpha
                pos_pred = s_map[in_radius].max()
                neg_pred = s_map[~in_radius].max()
                labels_scores[alpha]["scores"] += [
                    pos_pred.cpu().item(),
                    neg_pred.cpu().item(),
                ]
                labels_scores[alpha]["labels"] += [1, 0]
            else:
                neg_pred = s_map.max()
                labels_scores[alpha]["labels"].append(neg_pred.item())
                labels_scores[alpha]["scores"].append(0)

    return labels_scores


def numpy_cdist(X, Y):
    # X: shape (n_samples_X, n_features)
    # Y: shape (n_samples_Y, n_features)
    # Returns: shape (n_samples_X, n_samples_Y)
    differences = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return distances


def compute_pck(
    img2_kps: np.array,
    img2_kps_pred: np.array,
    alphas: List[float],
    threshold: float,
):
    n_query_pts = img2_kps.shape[0]
    alpha_is_match = {alpha: np.zeros(n_query_pts, dtype=bool) for alpha in alphas}
    pair_correct_counts = {alpha: 0 for alpha in alphas}
    alpha_is_match_plus = {alpha: np.zeros(n_query_pts, dtype=bool) for alpha in alphas}
    pair_correct_counts_plus = {alpha: 0 for alpha in alphas}
    # adaption to PCK+
    dists = numpy_cdist(img2_kps, img2_kps_pred)
    dists_to_gt = (
        dists.diagonal()
    )  # dists[i, i] is the distance between the i-th keypoint in img2_kps and its prediction
    argmin_dists = dists.argmin(axis=1)  # n_query_pts
    min_is_identity = argmin_dists == np.arange(n_query_pts)
    pair_statistics = {}
    pair_statistics_plus = {}
    for alpha in alphas:
        below_threshold = dists_to_gt <= (alpha * threshold)
        pair_correct_counts[alpha] = np.sum(below_threshold)
        alpha_is_match[alpha] = dists_to_gt <= (alpha * threshold)
        pair_statistics[f"acc_{alpha}"] = pair_correct_counts[alpha] / n_query_pts

        pair_correct_counts_plus[alpha] = np.sum(below_threshold & min_is_identity)
        alpha_is_match_plus[alpha] = below_threshold & min_is_identity
        pair_statistics_plus[f"acc_{alpha}"] = (
            pair_correct_counts_plus[alpha] / n_query_pts
        )

    # for j in range(n_query_pts):
    #     x_trg, y_trg = img2_kps[j]
    #     x_pred, y_pred = img2_kps_pred[j]
    #     dist = ((y_trg - y_pred) ** 2 + (x_trg - x_pred) ** 2) ** 0.5
    #     for alpha in alphas:
    #         if dist <= (alpha * threshold):
    #             pair_correct_counts[alpha] += 1
    #             alpha_is_match[alpha][j] = True

    # for alpha in alphas:
    #     # print(alpha, pair_correct_counts[alpha] / img1_kps.shape[0])
    #     pair_statistics[f'acc_{alpha}'] = pair_correct_counts[alpha] / n_query_pts
    #     pair_statistics_plus[f'acc_{alpha}'] = pair_correct_counts_plus[alpha] / n_query_pts
    return (
        pair_correct_counts,
        alpha_is_match,
        pair_statistics,
        pair_correct_counts_plus,
        alpha_is_match_plus,
        pair_statistics_plus,
    )


def get_kp_intersection(img1_kp_xy, img2_kp_xy, img1_kp_labels, img2_kp_labels):
    img1_kps, img2_kps = [], []
    kpt_labels = []
    inter1_idx = []
    inter2_idx = []
    for i, kp_id in enumerate(img1_kp_labels):
        for j, kp_id2 in enumerate(img2_kp_labels):
            if kp_id == kp_id2:
                inter1_idx.append(i)
                inter2_idx.append(j)
                img1_kps.append(img1_kp_xy[i])
                img2_kps.append(img2_kp_xy[j])
                kpt_labels.append(kp_id)
                break
    kpt_labels = np.array(kpt_labels)
    img1_kps = np.array(img1_kps)
    img2_kps = np.array(img2_kps)

    return img1_kps, img2_kps, inter1_idx, inter2_idx, kpt_labels


def compute_2d_correspondence_metrics_for_pair(
    pair_idx: int,
    alphas: List[float],
    img_files: List[str],
    oracle_fun: Callable,
    compute_mAP: bool = True,
):
    # Load image annotations and extract keypoints and thresholds
    img1_fp = img_files[2 * pair_idx]
    img2_fp = img_files[2 * pair_idx + 1]
    img1_anno = raw_data_utils.load_normalized_img_anno_from_img_fp(img1_fp)
    img2_anno = raw_data_utils.load_normalized_img_anno_from_img_fp(img2_fp)
    h1, w1 = img1_anno.img_height, img1_anno.img_width
    h2, w2 = img2_anno.img_height, img2_anno.img_width
    threshold = raw_data_utils.get_threshold_from_annotation(img2_anno)

    # sim_map has shape (img1_kps.shape[0], h2, w2)
    img1_kps = np.array(img1_anno.kp_xy)
    img2_kps_pred, sim = oracle_fun(img1_fp, img2_fp, img1_kps, h1, w1, h2, w2)
    if isinstance(img2_kps_pred, torch.Tensor):
        img2_kps_pred = img2_kps_pred.cpu().numpy()

    # PCK
    img1_kps_inter, img2_kps_inter, inter1_idx, inter2_idx, kpt_labels = (
        get_kp_intersection(
            img1_anno.kp_xy, img2_anno.kp_xy, img1_anno.kp_ids, img2_anno.kp_ids
        )
    )
    (
        pair_correct_counts,
        alpha_is_match,
        pair_statistics,
        pair_correct_counts_plus,
        alpha_is_match_plus,
        pair_statistics_plus,
    ) = compute_pck(
        img2_kps=img2_kps_inter,
        img2_kps_pred=img2_kps_pred[inter1_idx],
        alphas=alphas,
        threshold=threshold,
    )

    # mAP
    labels_scores = None
    if sim is not None and compute_mAP:
        labels_scores = compute_label_score(
            sim=sim,
            img1_kp_ids=img1_anno.kp_ids,
            img2_kp_ids=img2_anno.kp_ids,
            img2_kp_xy=torch.tensor(img2_anno.kp_xy, device=sim.device),
            threshold=threshold,
            alpha_levels=alphas,
        )

    other_data = {
        "img1_fp": img1_fp,
        "img2_fp": img2_fp,
        "img1_kps": img1_kps_inter,
        "img2_kps": img2_kps_inter,
        "img2_kps_pred": img2_kps_pred,
        "threshold": threshold,
        "kpt_labels": kpt_labels,
        "alpha_is_match": alpha_is_match,
        "alpha_is_match_plus": alpha_is_match_plus,
    }

    return (
        img1_kps_inter.shape[0],
        pair_correct_counts,
        pair_statistics,
        pair_correct_counts_plus,
        pair_statistics_plus,
        labels_scores,
        other_data,
    )


def compute_correspondence_2d_metrics(
    img_files: List[str],
    oracle_fun: Callable,
    alpha_levels=[0.1, 0.05, 0.01],
    compute_mAP: bool = True,
):
    """Compute evaluation metrics
    Args:
        img_files: list of image files of size 2*N, where N is the number of image pairs
            and (2*i, 2*i+1) are the images of the i-th pair
        oracle_fun: function that takes (img1_fp, img2_fp, img1_kps, h1, w1, h2, w2) and returns img2_kps_pred and sim_map (optional, can be None if not computing mAP)
        alpha_levels: list of alpha levels
        compute_mAP: whether to compute mAP (or KAP)
    """
    # TODO instead of loading each image json file for each image we should use the pair annotation
    N = len(img_files) // 2
    total_num_kpts = 0

    # PCK
    pck_statistics: Dict[float, float] = {alpha: 0.0 for alpha in alpha_levels}
    pck_pair_statistics = {f"acc_{alpha}": [] for alpha in alpha_levels}
    # PCK+
    pck_plus_statistics: Dict[float, float] = {alpha: 0.0 for alpha in alpha_levels}
    pck_plus_pair_statistics = {f"acc_{alpha}": [] for alpha in alpha_levels}
    # mAP (or KAP)
    kap_statistics: Dict[float, float] = {alpha: 0.0 for alpha in alpha_levels}
    kap_labels_scores = {alpha: {"labels": [], "scores": []} for alpha in alpha_levels}

    debug_data = []
    for pair_idx in tqdm(range(N)):
        # Load image annotations and extract keypoints and thresholds
        (
            n_query_pts,
            pair_correct_counts,
            pair_statistics_i,
            pair_correct_counts_plus_i,
            pair_statistics_plus_i,
            labels_scores_i,
            other_data,
        ) = compute_2d_correspondence_metrics_for_pair(
            pair_idx=pair_idx,
            alphas=alpha_levels,
            img_files=img_files,
            oracle_fun=oracle_fun,
            compute_mAP=compute_mAP,
        )
        debug_data.append(other_data)
        total_num_kpts += n_query_pts
        for alpha in alpha_levels:
            pck_statistics[alpha] += pair_correct_counts[alpha]
            pck_pair_statistics[f"acc_{alpha}"].append(
                pair_statistics_i[f"acc_{alpha}"]
            )
            pck_plus_statistics[alpha] += pair_correct_counts_plus_i[alpha]
            pck_plus_pair_statistics[f"acc_{alpha}"].append(
                pair_statistics_plus_i[f"acc_{alpha}"]
            )
            if compute_mAP:
                kap_labels_scores[alpha]["labels"] += labels_scores_i[alpha]["labels"]
                kap_labels_scores[alpha]["scores"] += labels_scores_i[alpha]["scores"]

    for alpha in alpha_levels:
        pck_statistics[alpha] /= total_num_kpts
        pck_pair_statistics[f"acc_{alpha}_mean"] = [
            np.mean(pck_pair_statistics[f"acc_{alpha}"]) * 100
        ]
        pck_plus_statistics[alpha] /= total_num_kpts
        pck_plus_pair_statistics[f"acc_{alpha}_mean"] = [
            np.mean(pck_plus_pair_statistics[f"acc_{alpha}"]) * 100
        ]
        if compute_mAP:
            kap_statistics[alpha] = average_precision_score(
                np.array(kap_labels_scores[alpha]["labels"]).astype(bool),
                kap_labels_scores[alpha]["scores"],
                average="weighted",
            )

    return (
        pck_statistics,
        pck_pair_statistics,
        pck_plus_statistics,
        pck_plus_pair_statistics,
        kap_statistics,
        kap_labels_scores,
        debug_data,
    )
