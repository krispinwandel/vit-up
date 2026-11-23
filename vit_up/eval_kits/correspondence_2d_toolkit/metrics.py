"""Percentage of correct keypoints (PCK) for SPair-71k keypoint transfer."""
import numpy as np

from .spair import Annotation, pck_threshold


def shared_keypoints(source: Annotation, target: Annotation) -> tuple[list[int], list[int]]:
    """Indices of the keypoints annotated in both images, in source order."""
    target_index = {}
    for j, kp_id in enumerate(target.kp_ids):
        target_index.setdefault(kp_id, j)
    pairs = [(i, target_index[kp_id]) for i, kp_id in enumerate(source.kp_ids) if kp_id in target_index]
    return [i for i, _ in pairs], [j for _, j in pairs]


def compute_pck(image_pairs: list[str], predictions: dict, annotations: dict,
                alphas=(0.1, 0.05, 0.01)) -> dict[float, float]:
    """PCK over all shared keypoints of alternating (source, target) image paths.

    `predictions[source, target]` holds the predicted target-image (x, y) of every
    source keypoint. A transfer is correct within alpha times the longer side of
    the target object's bounding box.
    """
    correct = {alpha: 0 for alpha in alphas}
    total = 0
    for source_path, target_path in zip(image_pairs[::2], image_pairs[1::2]):
        source, target = annotations[source_path], annotations[target_path]
        source_idx, target_idx = shared_keypoints(source, target)
        if not source_idx:
            continue
        threshold = pck_threshold(target)
        errors = np.linalg.norm(np.array(target.kp_xy)[target_idx] - predictions[source_path, target_path][source_idx],
                                axis=1)
        for alpha in alphas:
            correct[alpha] += np.sum(errors <= alpha * threshold)
        total += len(source_idx)
    return {alpha: correct[alpha] / total for alpha in alphas}
