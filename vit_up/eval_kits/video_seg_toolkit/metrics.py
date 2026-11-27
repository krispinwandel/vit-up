# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from google-research/lila (metrics.py, utils/davis_eval.py): only the
# metrics used by the evaluations are kept, the confusion matrix is accumulated
# with np.bincount, and results are returned as plain dicts. The definitions
# (DAVIS J&F) are unchanged.
"""Metrics of the LILA video segmentation protocol."""
from __future__ import annotations

import warnings

import cv2
import numpy as np


# --------------------------------------------------------------------------- DAVIS J&F

class DavisMetric:
    """DAVIS-2017 region (J) and boundary (F) similarity; first and last frames are excluded."""

    def __init__(self):
        self.results = {"J": {"M": [], "R": [], "D": []}, "F": {"M": [], "R": [], "D": []}}

    def add_sequence(self, gt_masks: np.ndarray, pred_masks: np.ndarray) -> None:
        gt = _object_masks(gt_masks.copy())[:, 1:-1]
        # As in the DAVIS code, the number of predicted objects comes from the first predicted
        # frame; objects missing there are scored as empty for the whole sequence.
        pred = _object_masks(pred_masks.copy())[:, 1:-1]
        if pred.shape[0] > gt.shape[0]:
            raise ValueError("Prediction contains more objects than the ground truth.")
        if pred.shape[0] < gt.shape[0]:
            pred = np.concatenate([pred, np.zeros((gt.shape[0] - pred.shape[0], *pred.shape[1:]), bool)])
        for obj in range(gt.shape[0]):
            j = _db_eval_iou(gt[obj], pred[obj])
            f = np.array([_f_measure(p, g) for p, g in zip(pred[obj], gt[obj])])
            for key, values in (("J", j), ("F", f)):
                for stat, value in zip("MRD", _db_statistics(values)):
                    self.results[key][stat].append(value)

    def compute(self) -> dict:
        J, F = self.results["J"], self.results["F"]
        return {"JF": float((np.mean(J["M"]) + np.mean(F["M"])) / 2),
                "J": float(np.mean(J["M"])), "J_recall": float(np.mean(J["R"])), "J_decay": float(np.mean(J["D"])),
                "F": float(np.mean(F["M"])), "F_recall": float(np.mean(F["R"])), "F_decay": float(np.mean(F["D"]))}


def _object_masks(masks: np.ndarray) -> np.ndarray:
    """(T, H, W) label maps to (objects in frame 0, T, H, W) binary masks; void (255) is background."""
    masks[masks == 255] = 0
    ids = np.arange(1, int(np.max(masks[0])) + 1)[:, None, None, None]
    return masks[None] == ids


def _db_statistics(values: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values)
        recall = np.nanmean(values > 0.5)
    ids = (np.round(np.linspace(1, len(values), 5) + 1e-10) - 1).astype(np.uint8)
    bins = [values[ids[i]:ids[i + 1] + 1] for i in range(4)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        decay = np.nanmean(bins[0]) - np.nanmean(bins[3])
    return mean, recall, decay


def _db_eval_iou(annotation: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    annotation, segmentation = annotation.astype(bool), segmentation.astype(bool)
    inters = np.sum(segmentation & annotation, axis=(-2, -1))
    union = np.sum(segmentation | annotation, axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        j = inters / union
    j[np.isclose(union, 0)] = 1
    return j


def _f_measure(foreground_mask: np.ndarray, gt_mask: np.ndarray, bound_th: float = 0.008) -> float:
    from skimage.morphology import disk

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)
    kernel = disk(bound_pix).astype(np.uint8)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), kernel)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), kernel)
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil
    n_fg, n_gt = np.sum(fg_boundary), np.sum(gt_boundary)
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1, 0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0, 1
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1, 1
    else:
        precision, recall = np.sum(fg_match) / float(n_fg), np.sum(gt_match) / float(n_gt)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Binary boundary map with 1 pixel wide boundaries (D. Martin, 2003)."""
    seg = seg.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b
