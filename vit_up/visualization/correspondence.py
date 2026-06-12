from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from vit_up.eval_kits.correspondence_2d_toolkit import metrics
from vit_up.eval_kits.correspondence_2d_toolkit.data_kit import raw_data_utils
from vit_up.utils import img_transforms, pil_img_utils


GREEN = (22, 170, 72)
RED = (220, 48, 48)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@dataclass(frozen=True)
class SpairPair:
    img1_fp: str
    img2_fp: str
    img1: Image.Image
    img2: Image.Image
    img1_kps: np.ndarray
    img2_kps: np.ndarray
    img1_size: tuple[int, int]
    img2_size: tuple[int, int]
    img1_bbox: tuple[float, float, float, float]
    img2_bbox: tuple[float, float, float, float]
    threshold: float
    kpt_labels: np.ndarray
    category: str
    pair_anno_fp: str


@dataclass(frozen=True)
class MatchResult:
    method_key: str
    title: str
    img2_kps_pred: np.ndarray
    is_correct: np.ndarray
    accuracy: float
    alpha: float = 0.1


def find_spair_dataset_dir(root: str | os.PathLike[str]) -> str:
    root = Path(root)
    candidates = [root, root / "SPair-71k", root / "spair71k"]
    for candidate in candidates:
        if (candidate / "JPEGImages").is_dir() and (candidate / "PairAnnotation").is_dir():
            return str(candidate)
    raise FileNotFoundError(f"Could not find SPair-71k under {root}")


def load_spair_pair(
    dataset_dir: str | os.PathLike[str],
    category: str = "dog",
    split: str = "test",
    pair_idx: int = 0,
    min_shared_keypoints: int = 6,
) -> SpairPair:
    dataset_dir = find_spair_dataset_dir(dataset_dir)
    pair_fps = sorted(
        glob.glob(os.path.join(dataset_dir, "PairAnnotation", split, f"*:{category}.json"))
    )
    if not pair_fps:
        raise FileNotFoundError(
            f"No SPair-71k pair annotations for category={category!r}, split={split!r}"
        )

    start_idx = pair_idx % len(pair_fps)
    ordered = pair_fps[start_idx:] + pair_fps[:start_idx]
    last_pair = None
    for pair_anno_fp in ordered:
        src_trg = raw_data_utils.load_normalized_src_tgt_img_anno(
            pair_anno_fp, "spair-71k"
        )
        img1_kps, img2_kps, _, _, kpt_labels = metrics.get_kp_intersection(
            src_trg.src.kp_xy,
            src_trg.trg.kp_xy,
            src_trg.src.kp_ids,
            src_trg.trg.kp_ids,
        )
        last_pair = (pair_anno_fp, src_trg, img1_kps, img2_kps, kpt_labels)
        if img1_kps.shape[0] >= min_shared_keypoints:
            break

    pair_anno_fp, src_trg, img1_kps, img2_kps, kpt_labels = last_pair
    img1_fp = os.path.join(dataset_dir, src_trg.src.rel_fp)
    img2_fp = os.path.join(dataset_dir, src_trg.trg.rel_fp)
    img1 = Image.open(img1_fp).convert("RGB")
    img2 = Image.open(img2_fp).convert("RGB")
    return SpairPair(
        img1_fp=img1_fp,
        img2_fp=img2_fp,
        img1=img1,
        img2=img2,
        img1_kps=img1_kps.astype(np.float32),
        img2_kps=img2_kps.astype(np.float32),
        img1_size=(src_trg.src.img_width, src_trg.src.img_height),
        img2_size=(src_trg.trg.img_width, src_trg.trg.img_height),
        img1_bbox=tuple(float(x) for x in src_trg.src.bbox),
        img2_bbox=tuple(float(x) for x in src_trg.trg.bbox),
        threshold=raw_data_utils.get_threshold_from_annotation(src_trg.trg),
        kpt_labels=kpt_labels,
        category=category,
        pair_anno_fp=pair_anno_fp,
    )


def _fit_centered_pca_data(centered_tokens_nc: torch.Tensor, k: int = 3) -> dict:
    if centered_tokens_nc.ndim != 2:
        raise ValueError(
            f"Expected centered PCA tokens with shape (N, C), got {tuple(centered_tokens_nc.shape)}"
        )
    if centered_tokens_nc.shape[0] < k:
        raise ValueError(f"Need at least {k} tokens to fit PCA, got {centered_tokens_nc.shape[0]}.")

    centered = centered_tokens_nc.float()
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:k].T

    projected = centered @ components
    color_min = projected.amin(dim=0)
    color_max = projected.amax(dim=0)
    flat = torch.isclose(color_max, color_min)
    color_max = torch.where(flat, color_min + 1.0, color_max)

    return {
        "pca_singular_values": singular_values[:k],
        "pca_eig": components,
        "pca_color_min": color_min,
        "pca_color_max": color_max,
        "pca_fit_projected_min": color_min,
        "pca_fit_projected_max": color_max,
        "pca_fit_token_count": torch.tensor(centered.shape[0], device=centered.device),
        "pca_fit_feature_dim": torch.tensor(centered.shape[1], device=centered.device),
    }


def _fit_pca_data(tokens_nc: torch.Tensor, k: int = 3) -> dict:
    if tokens_nc.ndim != 2:
        raise ValueError(f"Expected PCA tokens with shape (N, C), got {tuple(tokens_nc.shape)}")
    if tokens_nc.shape[0] < k:
        raise ValueError(f"Need at least {k} tokens to fit PCA, got {tokens_nc.shape[0]}.")

    # tokens = F.normalize(tokens_nc.float(), dim=-1)
    tokens = tokens_nc.float()
    mean = tokens.mean(dim=0)
    pca_data = _fit_centered_pca_data(tokens - mean, k=k)
    pca_data["pca_mean"] = mean
    return pca_data


def _apply_pca_rgb(
    feats_hwc: torch.Tensor,
    pca_data: Mapping,
    mean_key: str = "pca_mean",
) -> torch.Tensor:
    if feats_hwc.ndim != 3:
        raise ValueError(f"Expected HWC features, got {tuple(feats_hwc.shape)}")
    h, w, c = feats_hwc.shape
    tokens = feats_hwc.float().reshape(-1, c)
    # tokens = F.normalize(tokens, dim=-1)
    mean = pca_data[mean_key].to(device=tokens.device, dtype=tokens.dtype)
    components = pca_data["pca_eig"].to(device=tokens.device, dtype=tokens.dtype)
    color_min = pca_data["pca_color_min"].to(device=tokens.device, dtype=tokens.dtype)
    color_max = pca_data["pca_color_max"].to(device=tokens.device, dtype=tokens.dtype)

    # projected = (F.normalize(tokens, dim=-1) - mean) @ components
    projected = (tokens - mean) @ components
    rgb = (projected - color_min.view(1, -1)) / (color_max - color_min).view(1, -1).add(1e-8)
    rgb = rgb.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    return rgb.reshape(h, w, 3)


def _pca_projection_stats(
    feats_hwc: torch.Tensor,
    pca_data: Mapping,
    mean_key: str = "pca_mean",
) -> dict:
    h, w, c = feats_hwc.shape
    tokens = feats_hwc.float().reshape(-1, c)
    mean = pca_data[mean_key].to(device=tokens.device, dtype=tokens.dtype)
    components = pca_data["pca_eig"].to(device=tokens.device, dtype=tokens.dtype)
    projected = (tokens - mean) @ components
    color_min = pca_data["pca_color_min"].to(device=tokens.device, dtype=tokens.dtype)
    color_max = pca_data["pca_color_max"].to(device=tokens.device, dtype=tokens.dtype)
    below = (projected < color_min.view(1, -1)).float().mean(dim=0)
    above = (projected > color_max.view(1, -1)).float().mean(dim=0)
    return {
        "shape": tuple(feats_hwc.shape),
        "proj_min": tuple(projected.amin(dim=0).detach().cpu().tolist()),
        "proj_max": tuple(projected.amax(dim=0).detach().cpu().tolist()),
        "clip_below": tuple(below.detach().cpu().tolist()),
        "clip_above": tuple(above.detach().cpu().tolist()),
    }


def prepare_spair_image(
    img: Image.Image,
    img_size: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    transform = img_transforms.build_image_transform(img_size, img_size)
    img_square = pil_img_utils.pad_image_to_square(img)
    return transform(img_square).unsqueeze(0).to(device)


def prepare_spair_pair_images(
    pair: SpairPair,
    img_size: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    return torch.cat(
        [
            prepare_spair_image(pair.img1, img_size=img_size, device=device),
            prepare_spair_image(pair.img2, img_size=img_size, device=device),
        ],
        dim=0,
    )


class HuggingFaceSamPredictor:
    def __init__(
        self,
        model_id: str = "facebook/sam3.1",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.backend = "sam3" if "sam3" in model_id.lower() else "transformers"
        self.image: Image.Image | None = None
        self.sam3_state = None

        if self.backend == "sam3":
            try:
                from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
                from sam3.model.sam3_image_processor import Sam3Processor
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "facebook/sam3.1 requires the official SAM3 package, not `transformers`. "
                    "Install the Meta repo in the notebook environment, e.g. "
                    "`pip install git+https://github.com/facebookresearch/sam3.git`, "
                    "and make sure you have accepted the Hugging Face model terms and run `hf auth login`."
                ) from exc

            version = "sam3.1" if "3.1" in model_id else "sam3"
            checkpoint_path = download_ckpt_from_hf(version=version)
            self.model = build_sam3_image_model(
                device=str(self.device),
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
            self.processor = Sam3Processor(self.model, device=str(self.device))
            return

        try:
            if "sam2" in model_id.lower():
                from transformers import Sam2Model as SamModelCls
                from transformers import Sam2Processor as SamProcessorCls
            else:
                from transformers import SamModel as SamModelCls
                from transformers import SamProcessor as SamProcessorCls
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "HuggingFaceSamPredictor requires `transformers` with SAM/SAM2 support. "
                "Install or upgrade it in the notebook environment, e.g. `pip install -U transformers`."
            ) from exc
        except ImportError as exc:
            raise ImportError(
                f"The installed `transformers` package does not expose the classes needed for {model_id!r}. "
                "Upgrade it in the notebook environment, e.g. `pip install -U transformers`."
            ) from exc

        self.processor = SamProcessorCls.from_pretrained(model_id)
        kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
        self.model = SamModelCls.from_pretrained(model_id, **kwargs).to(self.device).eval()

    def set_image(self, image: np.ndarray | Image.Image) -> None:
        if isinstance(image, Image.Image):
            self.image = image.convert("RGB")
        else:
            self.image = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        if self.backend == "sam3":
            self.sam3_state = self.processor.set_image(self.image)

    @staticmethod
    def _xyxy_to_normalized_cxcywh(box_xyxy: np.ndarray, image_size: tuple[int, int]) -> list[float]:
        width, height = image_size
        x0, y0, x1, y1 = box_xyxy.astype(np.float32).tolist()
        x0 = min(max(x0, 0.0), float(width))
        x1 = min(max(x1, 0.0), float(width))
        y0 = min(max(y0, 0.0), float(height))
        y1 = min(max(y1, 0.0), float(height))
        return [
            ((x0 + x1) * 0.5) / max(1, width),
            ((y0 + y1) * 0.5) / max(1, height),
            max(1.0, x1 - x0) / max(1, width),
            max(1.0, y1 - y0) / max(1, height),
        ]

    @staticmethod
    def _point_box(points_xy: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        width, height = image_size
        x0, y0 = points_xy.min(axis=0)
        x1, y1 = points_xy.max(axis=0)
        pad = 0.08 * max(width, height)
        return np.asarray([x0 - pad, y0 - pad, x1 + pad, y1 + pad], dtype=np.float32)

    def _predict_sam3(
        self,
        point_coords: np.ndarray,
        box: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        if self.image is None or self.sam3_state is None:
            raise RuntimeError("Call set_image before predict.")
        if box is None:
            box = self._point_box(point_coords, self.image.size)
        normalized_box = self._xyxy_to_normalized_cxcywh(np.asarray(box), self.image.size)
        self.processor.reset_all_prompts(self.sam3_state)
        state = self.processor.add_geometric_prompt(
            box=normalized_box,
            label=True,
            state=self.sam3_state,
        )
        masks = state["masks"].detach().float().cpu()
        scores = state["scores"].detach().float().cpu()
        if masks.ndim == 4:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        return masks.numpy().astype(bool), scores.reshape(-1).numpy(), None

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        if self.image is None:
            raise RuntimeError("Call set_image before predict.")
        point_coords = np.asarray(point_coords, dtype=np.float32)
        point_labels = np.asarray(point_labels, dtype=np.int64)

        if self.backend == "sam3":
            return self._predict_sam3(point_coords=point_coords, box=box)

        processor_kwargs = {
            "images": self.image,
            "input_points": [[point_coords.tolist()]],
            "input_labels": [[point_labels.tolist()]],
            "return_tensors": "pt",
        }
        if box is not None:
            processor_kwargs["input_boxes"] = [[np.asarray(box, dtype=np.float32).tolist()]]

        inputs = self.processor(**processor_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
        with torch.no_grad():
            try:
                outputs = self.model(**inputs, multimask_output=multimask_output)
            except TypeError:
                outputs = self.model(**inputs)

        pred_masks = outputs.pred_masks.detach().cpu()
        original_sizes = inputs["original_sizes"].detach().cpu()
        if hasattr(self.processor, "post_process_masks"):
            masks = self.processor.post_process_masks(pred_masks, original_sizes)[0]
        else:
            masks = self.processor.image_processor.post_process_masks(
                pred_masks,
                original_sizes,
                inputs["reshaped_input_sizes"].detach().cpu(),
            )[0]
        scores = outputs.iou_scores.detach().cpu()

        masks = masks.squeeze(0)
        scores = scores.squeeze(0).squeeze(0)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim == 4:
            masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        scores = scores.reshape(-1)
        return masks.numpy().astype(bool), scores.numpy(), None


def _select_sam_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"Expected SAM masks with shape (M, H, W), got {masks.shape}")
    if masks.shape[0] == 1 or points_xy.size == 0:
        return masks[int(np.argmax(scores))].astype(bool)

    point_hits = []
    point_coverage = []
    for mask in masks:
        h, w = mask.shape
        xs = np.clip(np.round(points_xy[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(points_xy[:, 1]).astype(int), 0, h - 1)
        hit_count = int(mask[ys, xs].sum())
        point_hits.append(hit_count)
        point_coverage.append(hit_count / max(1, int(mask.sum())))
    point_hits = np.asarray(point_hits)
    best_hits = point_hits.max()
    if best_hits == 0:
        return masks[int(np.argmax(scores))].astype(bool)
    candidates = np.flatnonzero(point_hits == best_hits)
    if len(candidates) > 1:
        coverages = np.asarray(point_coverage)
        best_coverage = coverages[candidates].max()
        candidates = candidates[coverages[candidates] == best_coverage]
    best = candidates[int(np.argmax(scores[candidates]))]
    return masks[best].astype(bool)


def _predict_sam_mask(
    sam_predictor,
    img: Image.Image,
    points_xy: np.ndarray,
    bbox_xyxy: Sequence[float] | None = None,
) -> np.ndarray:
    if points_xy.size == 0:
        raise ValueError("SAM point prompting requires at least one point.")
    image_np = np.asarray(img.convert("RGB"))
    sam_predictor.set_image(image_np)
    point_coords = points_xy.astype(np.float32)
    point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)
    box = None if bbox_xyxy is None else np.asarray(bbox_xyxy, dtype=np.float32)
    masks, scores, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    return _select_sam_mask(masks, np.asarray(scores), point_coords)


def compute_spair_sam_masks(pair: SpairPair, sam_predictor) -> tuple[np.ndarray, np.ndarray]:
    img1_mask = _predict_sam_mask(
        sam_predictor, pair.img1, pair.img1_kps, bbox_xyxy=pair.img1_bbox
    )
    img2_mask = _predict_sam_mask(
        sam_predictor, pair.img2, pair.img2_kps, bbox_xyxy=pair.img2_bbox
    )
    return img1_mask, img2_mask


def _image_mask_to_feature_mask(
    mask_hw: np.ndarray | torch.Tensor,
    image_size: tuple[int, int],
    feature_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    img_w, img_h = image_size
    max_size = max(img_w, img_h)
    px, py = pil_img_utils.get_square_paddings(img_w, img_h)
    mask = torch.as_tensor(mask_hw, dtype=torch.float32, device=device)
    if tuple(mask.shape) != (img_h, img_w):
        raise ValueError(
            f"Expected mask shape {(img_h, img_w)} for image_size={image_size}, got {tuple(mask.shape)}"
        )
    padded = torch.zeros((max_size, max_size), dtype=torch.float32, device=device)
    padded[py : py + img_h, px : px + img_w] = mask
    resized = F.interpolate(
        padded[None, None], size=feature_hw, mode="nearest"
    )[0, 0]
    feature_mask = resized > 0.5
    if not bool(feature_mask.any()):
        feature_mask = torch.ones(feature_hw, dtype=torch.bool, device=device)
    return feature_mask


def compute_spair_pair_pca_data(
    backbone,
    pair: SpairPair,
    img_size: int,
    device: str | torch.device = "cuda",
    pca_hidden_states_sizes: Sequence[int] | None = None,
) -> dict:
    pixel_values = prepare_spair_pair_images(pair, img_size=img_size, device=device)
    patch_size = backbone.get_patch_size()
    if pca_hidden_states_sizes is None:
        hidden_states_grid_size = img_size // patch_size
        pca_hidden_states_sizes = [16, hidden_states_grid_size, 64]

    h_size_to_feats_bhwc = {}
    fit_tokens = []
    with torch.no_grad():
        for h_size in pca_hidden_states_sizes:
            input_size = h_size * patch_size
            pixel_values_in = F.interpolate(
                pixel_values,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            feats_bhwc = backbone._compute_gt_features(
                backbone,
                pixel_values=pixel_values_in,
                layer_indices=[12],
                flatten_hw_to_seq=False,
            )[0].float()
            if feats_bhwc.ndim != 4:
                raise ValueError(f"Expected backbone features as BHWC, got {tuple(feats_bhwc.shape)}")
            h_size_to_feats_bhwc[h_size] = feats_bhwc
            print("feats_bhwc.shape", feats_bhwc.shape)
            fit_tokens.append(feats_bhwc.reshape(-1, feats_bhwc.shape[-1]))

    pca_data = _fit_pca_data(torch.cat(fit_tokens, dim=0), k=3)
    pca_data["h_size_to_feats_bhwc"] = h_size_to_feats_bhwc
    return pca_data


def compute_spair_pair_masked_pca_data(
    backbone,
    pair: SpairPair,
    img_size: int,
    device: str | torch.device = "cuda",
    pca_hidden_states_sizes: Sequence[int] | None = None,
    sam_predictor=None,
) -> dict:
    pixel_values = prepare_spair_pair_images(pair, img_size=img_size, device=device)
    patch_size = backbone.get_patch_size()
    if pca_hidden_states_sizes is None:
        hidden_states_grid_size = img_size // patch_size
        pca_hidden_states_sizes = [16, hidden_states_grid_size, 64]

    sam_masks = compute_spair_sam_masks(pair, sam_predictor) if sam_predictor is not None else None
    h_size_to_feats_bhwc = {}
    fit_tokens = []
    source_token_count = 0
    target_token_count = 0
    with torch.no_grad():
        for h_size in pca_hidden_states_sizes:
            input_size = h_size * patch_size
            pixel_values_in = F.interpolate(
                pixel_values,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            feats_bhwc = backbone._compute_gt_features(
                backbone,
                pixel_values=pixel_values_in,
                layer_indices=[12],
                flatten_hw_to_seq=False,
            )[0].float()
            if feats_bhwc.ndim != 4:
                raise ValueError(f"Expected backbone features as BHWC, got {tuple(feats_bhwc.shape)}")
            if feats_bhwc.shape[0] < 2:
                raise ValueError("Masked SPair PCA expects source and target in the batch.")
            h_size_to_feats_bhwc[h_size] = feats_bhwc

            if sam_masks is not None:
                img1_mask = _image_mask_to_feature_mask(
                    sam_masks[0], pair.img1_size, feats_bhwc.shape[1:3], feats_bhwc.device
                )
                img2_mask = _image_mask_to_feature_mask(
                    sam_masks[1], pair.img2_size, feats_bhwc.shape[1:3], feats_bhwc.device
                )
            else:
                img1_mask = _feature_bbox_mask(
                    feats_bhwc.shape[1:3], pair.img1_size, pair.img1_bbox, feats_bhwc.device
                )
                img2_mask = _feature_bbox_mask(
                    feats_bhwc.shape[1:3], pair.img2_size, pair.img2_bbox, feats_bhwc.device
                )
            img1_tokens = feats_bhwc[0][img1_mask]
            img2_tokens = feats_bhwc[1][img2_mask]
            fit_tokens.extend([img1_tokens, img2_tokens])
            source_token_count += img1_tokens.shape[0]
            target_token_count += img2_tokens.shape[0]

    img1_fit_tokens = torch.cat(fit_tokens[0::2], dim=0)
    img2_fit_tokens = torch.cat(fit_tokens[1::2], dim=0)
    # img1_fit_tokens = F.normalize(img1_fit_tokens, dim=-1)
    # img2_fit_tokens = F.normalize(img2_fit_tokens, dim=-1)
    img1_mean = img1_fit_tokens.mean(dim=0)
    img2_mean = img2_fit_tokens.mean(dim=0)
    centered_fit_tokens = torch.cat(
        [img1_fit_tokens - img1_mean, img2_fit_tokens - img2_mean],
        dim=0,
    )
    pca_data = _fit_centered_pca_data(centered_fit_tokens, k=3)
    pca_data["pca_mean"] = 0.5 * (img1_mean + img2_mean)
    pca_data["pca_source_mean"] = img1_mean
    pca_data["pca_target_mean"] = img2_mean
    pca_data["h_size_to_feats_bhwc"] = h_size_to_feats_bhwc
    if sam_masks is not None:
        pca_data["pca_image_masks_hw"] = sam_masks
    pca_data["pca_fit_image_count"] = torch.tensor(2, device=fit_tokens[0].device)
    pca_data["pca_fit_source_token_count"] = torch.tensor(source_token_count, device=fit_tokens[0].device)
    pca_data["pca_fit_target_token_count"] = torch.tensor(target_token_count, device=fit_tokens[0].device)
    pca_data["pca_fit_scope"] = "source_target_sam_masked" if sam_masks is not None else "source_target_bbox_masked"
    return pca_data


def infer_feature_map(
    model: Callable,
    img: Image.Image,
    img_size: int,
    out_size: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    pixel_values = prepare_spair_image(img, img_size=img_size, device=device)
    with torch.no_grad():
        feats = model(
            pixel_values_bchw=pixel_values,
            output_size=out_size,
            input_size=img_size,
        )
    if feats.ndim != 4:
        raise ValueError(f"Expected BHWC feature map, got shape={tuple(feats.shape)}")
    return feats[0].float()


def infer_feature_maps_for_pair(
    pair: SpairPair,
    models: Mapping[str, Callable],
    img_size: int,
    out_size: int,
    device: str | torch.device = "cuda",
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    feature_maps = {}
    for method_key, model in models.items():
        if hasattr(model, "eval"):
            model.eval()
        img1_fts = infer_feature_map(model, pair.img1, img_size, out_size, device)
        img2_fts = infer_feature_map(model, pair.img2, img_size, out_size, device)
        feature_maps[method_key] = (img1_fts, img2_fts)
    return feature_maps


def predict_matches_from_features(
    pair: SpairPair,
    img1_fts_hwc: torch.Tensor,
    img2_fts_hwc: torch.Tensor,
) -> np.ndarray:
    if img1_fts_hwc.ndim != 3 or img2_fts_hwc.ndim != 3:
        raise ValueError("Feature maps must be HWC tensors.")
    if img1_fts_hwc.shape[:2] != img2_fts_hwc.shape[:2]:
        raise ValueError("Source and target feature maps must have matching H,W.")
    if img1_fts_hwc.shape[0] != img1_fts_hwc.shape[1]:
        raise ValueError("Expected square feature maps.")

    ft_size = int(img1_fts_hwc.shape[0])
    w1, h1 = pair.img1_size
    w2, h2 = pair.img2_size
    img1_px, img1_py = pil_img_utils.get_square_paddings(w1, h1)
    img2_px, img2_py = pil_img_utils.get_square_paddings(w2, h2)
    img1_max_size = max(h1, w1)
    img2_max_size = max(h2, w2)

    img1_fts = F.normalize(img1_fts_hwc, dim=-1)
    img2_fts = F.normalize(img2_fts_hwc, dim=-1)
    # img1_fts = img1_fts_hwc
    # img2_fts = img2_fts_hwc

    preds = []
    for kp_xy in pair.img1_kps:
        kp_x = (float(kp_xy[0]) + img1_px) / img1_max_size
        kp_y = (float(kp_xy[1]) + img1_py) / img1_max_size
        query_x = min(ft_size - 1, max(0, int(kp_x * ft_size)))
        query_y = min(ft_size - 1, max(0, int(kp_y * ft_size)))
        query_ft = img1_fts[query_y, query_x]
        cos_sim = torch.einsum("c,hwc->hw", query_ft, img2_fts)
        flat_idx = torch.argmax(cos_sim)
        best_y = int(flat_idx // ft_size)
        best_x = int(flat_idx % ft_size)
        pred_x = ((best_x + 0.5) / ft_size * img2_max_size) - img2_px
        pred_y = ((best_y + 0.5) / ft_size * img2_max_size) - img2_py
        preds.append((pred_x, pred_y))
    return np.asarray(preds, dtype=np.float32)


def build_match_results(
    pair: SpairPair,
    feature_maps: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    titles: Mapping[str, str] | None = None,
    alpha: float = 0.1,
) -> list[MatchResult]:
    results = []
    for method_key, (img1_fts, img2_fts) in feature_maps.items():
        preds = predict_matches_from_features(pair, img1_fts, img2_fts)
        _, is_correct_by_alpha, _, _, _, _ = metrics.compute_pck(
            img2_kps=pair.img2_kps,
            img2_kps_pred=preds,
            alphas=[alpha],
            threshold=pair.threshold,
        )
        is_correct = is_correct_by_alpha[alpha]
        results.append(
            MatchResult(
                method_key=method_key,
                title=(titles or {}).get(method_key, method_key),
                img2_kps_pred=preds,
                is_correct=is_correct,
                accuracy=float(np.mean(is_correct)) if len(is_correct) else 0.0,
                alpha=alpha,
            )
        )
    return results


def _resize_point(xy: Sequence[float], src_size: tuple[int, int], dst_size: tuple[int, int]):
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return float(xy[0]) * sx, float(xy[1]) * sy


def _match_panel_sizes(
    pair: SpairPair,
    match_img_width: int = 2 * 448,
    gap: int = 2,
) -> tuple[int, int, int]:
    if gap < 0:
        raise ValueError("gap must be non-negative.")
    if match_img_width <= gap + 1:
        raise ValueError("match_img_width must leave room for both images and the gap.")

    r1 = pair.img1_size[0] / pair.img1_size[1]
    r2 = pair.img2_size[0] / pair.img2_size[1]
    content_width = match_img_width - gap
    image_height = max(1, int(round(content_width / (r1 + r2))))
    img1_w = int(round(content_width * r1 / (r1 + r2)))
    img1_w = min(max(1, img1_w), content_width - 1)
    img2_w = content_width - img1_w
    return image_height, img1_w, img2_w


def draw_match_item(
    pair: SpairPair,
    result: MatchResult,
    # image_height: int = 260,
    match_img_width: int = 2*448,
    gap: int = 2,
    line_width: int = 6,
    point_radius: int = 8,
    max_matches: int | None = 12,
    show_accuracy: bool = True,
    font_size: int = 18,
) -> Image.Image:
    image_height, img1_w, img2_w = _match_panel_sizes(pair, match_img_width, gap)
    # img1_w = int(round(pair.img1.width * image_height / pair.img1.height))
    # img2_w = int(round(pair.img2.width * image_height / pair.img2.height))
    img1 = pair.img1.resize((img1_w, image_height), Image.Resampling.LANCZOS)
    img2 = pair.img2.resize((img2_w, image_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (img1_w + gap + img2_w, image_height), WHITE)
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1_w + gap, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    match_indices = np.arange(pair.img1_kps.shape[0])
    if max_matches is not None and len(match_indices) > max_matches:
        correct_idx = match_indices[result.is_correct]
        wrong_idx = match_indices[~result.is_correct]
        n_wrong = min(len(wrong_idx), max_matches // 2)
        n_correct = min(len(correct_idx), max_matches - n_wrong)
        selected = np.concatenate([wrong_idx[:n_wrong], correct_idx[:n_correct]])
        if selected.size < max_matches:
            remainder = np.setdiff1d(match_indices, selected, assume_unique=False)
            selected = np.concatenate([selected, remainder[: max_matches - selected.size]])
        match_indices = selected

    for idx in match_indices:
        src_xy = _resize_point(pair.img1_kps[idx], pair.img1.size, img1.size)
        pred_xy = _resize_point(result.img2_kps_pred[idx], pair.img2.size, img2.size)
        pred_xy = (pred_xy[0] + img1_w + gap, pred_xy[1])
        color = GREEN if result.is_correct[idx] else RED
        draw.line([src_xy, pred_xy], fill=(*color, 210), width=line_width)
        draw.ellipse(
            [
                src_xy[0] - point_radius,
                src_xy[1] - point_radius,
                src_xy[0] + point_radius,
                src_xy[1] + point_radius,
            ],
            fill=(*color, 235),
            outline=(*BLACK, 220),
            width=1,
        )
        draw.ellipse(
            [
                pred_xy[0] - point_radius,
                pred_xy[1] - point_radius,
                pred_xy[0] + point_radius,
                pred_xy[1] + point_radius,
            ],
            fill=(*color, 235),
            outline=(*BLACK, 220),
            width=1,
        )

    title = result.title
    if show_accuracy:
        title = f"{title}  PCK@{result.alpha:g} {100.0 * result.accuracy:.0f}%"
    # return pil_img_utils.add_description_to_image(
    #     canvas,
    #     title,
    #     font_size=font_size,
    #     text_align="center",
    #     textbox_height=max(32, font_size + 16),
    #     overlay=False,
    #     bg_color="white",
    #     font_color="black",
    # )
    return canvas


def _crop_feature_square_to_image_aspect(
    img: Image.Image,
    original_size: tuple[int, int],
) -> Image.Image:
    width, height = original_size
    max_size = max(width, height)
    px, py = pil_img_utils.get_square_paddings(width, height)
    scale_x = img.width / max_size
    scale_y = img.height / max_size
    left = int(round(px * scale_x))
    top = int(round(py * scale_y))
    right = int(round((px + width) * scale_x))
    bottom = int(round((py + height) * scale_y))
    return img.crop((left, top, right, bottom))


def _compute_feature_pca_data(feats_hwc: torch.Tensor) -> dict:
    tokens = feats_hwc.float().reshape(-1, feats_hwc.shape[-1])
    return _fit_pca_data(tokens, k=3)


def _feature_bbox_mask(
    feature_hw: tuple[int, int],
    image_size: tuple[int, int],
    bbox_xyxy: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    h_ft, w_ft = feature_hw
    img_w, img_h = image_size
    max_size = max(img_w, img_h)
    px, py = pil_img_utils.get_square_paddings(img_w, img_h)
    x0, y0, x1, y1 = bbox_xyxy

    ys = (torch.arange(h_ft, device=device, dtype=torch.float32) + 0.5) / h_ft
    xs = (torch.arange(w_ft, device=device, dtype=torch.float32) + 0.5) / w_ft
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    img_x = xx * max_size - px
    img_y = yy * max_size - py
    mask = (img_x >= x0) & (img_x <= x1) & (img_y >= y0) & (img_y <= y1)
    if not bool(mask.any()):
        mask = torch.ones((h_ft, w_ft), device=device, dtype=torch.bool)
    return mask


def _pca_feature_image(
    feats_hwc: torch.Tensor,
    pca_data: Mapping,
    original_size: tuple[int, int],
    display_size: tuple[int, int],
    mask_hw: torch.Tensor | None = None,
    mean_key: str = "pca_mean",
) -> Image.Image:
    pca_color = _apply_pca_rgb(feats_hwc, pca_data, mean_key=mean_key)
    if mask_hw is not None:
        if tuple(mask_hw.shape) != tuple(feats_hwc.shape[:2]):
            raise ValueError(
                f"Expected PCA display mask shape {tuple(feats_hwc.shape[:2])}, got {tuple(mask_hw.shape)}"
            )
        pca_color = pca_color.clone()
        pca_color[~mask_hw.to(device=pca_color.device, dtype=torch.bool)] = 0
    pca_np = pca_color.detach().cpu().numpy().astype(np.uint8)
    pca_img = Image.fromarray(pca_np, mode="RGB")
    pca_img = _crop_feature_square_to_image_aspect(pca_img, original_size)
    return pca_img.resize(display_size, Image.Resampling.NEAREST)


def _pca_display_masks_for_features(
    pair: SpairPair,
    pca_data: Mapping,
    img1_feature_hw: tuple[int, int],
    img2_feature_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_masks = pca_data.get("pca_image_masks_hw")
    if image_masks is not None:
        img1_mask = _image_mask_to_feature_mask(image_masks[0], pair.img1_size, img1_feature_hw, device)
        img2_mask = _image_mask_to_feature_mask(image_masks[1], pair.img2_size, img2_feature_hw, device)
    else:
        img1_mask = _feature_bbox_mask(img1_feature_hw, pair.img1_size, pair.img1_bbox, device)
        img2_mask = _feature_bbox_mask(img2_feature_hw, pair.img2_size, pair.img2_bbox, device)
    return img1_mask, img2_mask


def _pca_data_display_feature_pair(pca_data: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    h_size_to_feats_bhwc = pca_data.get("h_size_to_feats_bhwc")
    if not h_size_to_feats_bhwc:
        raise ValueError("pca_data does not include cached backbone features for PCA display.")
    h_size = max(h_size_to_feats_bhwc)
    feats_bhwc = h_size_to_feats_bhwc[h_size]
    if feats_bhwc.ndim != 4 or feats_bhwc.shape[0] < 2:
        raise ValueError(f"Expected cached PCA features as B,H,W,C with B>=2, got {tuple(feats_bhwc.shape)}")
    return feats_bhwc[0], feats_bhwc[1]


def draw_pca_item(
    pair: SpairPair,
    img1_fts_hwc: torch.Tensor,
    img2_fts_hwc: torch.Tensor,
    pca_data: Mapping | None = None,
    match_img_width: int = 2 * 448,
    gap: int = 2,
    pca_scope: str = "shared",
) -> Image.Image:
    if pca_scope not in {"shared", "per_image", "shared_masked"}:
        raise ValueError(f"Unsupported pca_scope={pca_scope!r}.")
    if pca_scope in {"shared", "shared_masked"} and pca_data is None:
        raise ValueError(f"pca_data is required when pca_scope={pca_scope!r}.")

    image_height, img1_w, img2_w = _match_panel_sizes(pair, match_img_width, gap)
    if pca_scope == "per_image":
        img1_pca_data = _compute_feature_pca_data(img1_fts_hwc)
        img2_pca_data = _compute_feature_pca_data(img2_fts_hwc)
        img1_mask = None
        img2_mask = None
        img1_mean_key = "pca_mean"
        img2_mean_key = "pca_mean"
    else:
        img1_pca_data = pca_data
        img2_pca_data = pca_data
        img1_mean_key = "pca_source_mean" if "pca_source_mean" in pca_data else "pca_mean"
        img2_mean_key = "pca_target_mean" if "pca_target_mean" in pca_data else "pca_mean"
        if pca_scope == "shared_masked":
            img1_mask, img2_mask = _pca_display_masks_for_features(
                pair,
                pca_data,
                img1_fts_hwc.shape[:2],
                img2_fts_hwc.shape[:2],
                img1_fts_hwc.device,
            )
            img2_mask = img2_mask.to(img2_fts_hwc.device)
        else:
            img1_mask = None
            img2_mask = None
    img1 = _pca_feature_image(
        img1_fts_hwc,
        img1_pca_data,
        pair.img1_size,
        (img1_w, image_height),
        # mask_hw=img1_mask,
        mean_key=img1_mean_key,
    )
    img2 = _pca_feature_image(
        img2_fts_hwc,
        img2_pca_data,
        pair.img2_size,
        (img2_w, image_height),
        # mask_hw=img2_mask,
        mean_key=img2_mean_key,
    )

    canvas = Image.new("RGB", (img1_w + gap + img2_w, image_height), WHITE)
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1_w + gap, 0))
    return canvas


def make_pca_row_image(
    pair: SpairPair,
    feature_maps: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    pca_data: Mapping | None = None,
    method_order: Sequence[str] | None = None,
    match_img_width: int = 2 * 448,
    pad: int = 8,
    pca_scope: str = "shared",
    pca_feature_source: str = "model",
) -> Image.Image:
    if pca_feature_source not in {"model", "pca_data"}:
        raise ValueError(f"Unsupported pca_feature_source={pca_feature_source!r}.")
    keys = list(method_order) if method_order is not None else list(feature_maps.keys())
    if pca_feature_source == "pca_data":
        if pca_data is None:
            raise ValueError("pca_data is required when pca_feature_source='pca_data'.")
        img1_fts_hwc, img2_fts_hwc = _pca_data_display_feature_pair(pca_data)
        feature_pair_by_key = {key: (img1_fts_hwc, img2_fts_hwc) for key in keys}
    else:
        feature_pair_by_key = feature_maps
    items = [
        draw_pca_item(
            pair,
            feature_pair_by_key[key][0],
            feature_pair_by_key[key][1],
            pca_data,
            match_img_width=match_img_width,
            pca_scope=pca_scope,
        )
        for key in keys
        if key in feature_pair_by_key
    ]
    return pil_img_utils.concat_images(items, mode="row", pad=pad, pad_color=WHITE)


def make_match_row_image(
    pair: SpairPair,
    results: Sequence[MatchResult],
    method_order: Sequence[str] | None = None,
    match_img_width: int = 2*448,
    pad: int = 8,
    max_matches: int | None = 12,
) -> Image.Image:
    if method_order is not None:
        by_key = {result.method_key: result for result in results}
        ordered = [by_key[key] for key in method_order if key in by_key]
    else:
        ordered = list(results)
    items = [
        draw_match_item(
            pair,
            result,
            match_img_width=match_img_width,
            max_matches=max_matches,
        )
        for result in ordered
    ]
    return pil_img_utils.concat_images(items, mode="row", pad=pad, pad_color=WHITE)


def visualize_spair_methods(
    dataset_dir: str | os.PathLike[str],
    models: Mapping[str, Callable],
    titles: Mapping[str, str] | None = None,
    category: str = "dog",
    split: str = "test",
    pair_idx: int = 0,
    img_size: int = 448,
    out_size: int = 112,
    device: str | torch.device = "cuda",
    match_img_width: int = 2*448,
    max_matches: int | None = 12,
    method_order: Sequence[str] | None = None,
    alpha: float = 0.1,
    show_pca_row: bool = False,
    pca_data: Mapping | None = None,
    pca_backbone=None,
    pca_hidden_states_sizes: Sequence[int] | None = None,
    sam_predictor=None,
    pca_scope: str = "shared",
    pca_feature_source: str = "model",
    row_gap: int = 2,
    return_pca_data: bool = False,
) -> tuple:
    pair = load_spair_pair(
        dataset_dir=dataset_dir,
        category=category,
        split=split,
        pair_idx=pair_idx,
    )
    feature_maps = infer_feature_maps_for_pair(
        pair=pair,
        models=models,
        img_size=img_size,
        out_size=out_size,
        device=device,
    )
    results = build_match_results(pair, feature_maps, titles=titles, alpha=alpha)
    match_row_img = make_match_row_image(
        pair=pair,
        results=results,
        method_order=method_order,
        match_img_width=match_img_width,
        max_matches=max_matches,
    )
    if show_pca_row:
        if pca_scope not in {"shared", "per_image", "shared_masked"}:
            raise ValueError(f"Unsupported pca_scope={pca_scope!r}.")
        if pca_scope == "shared_masked" and pca_data is None:
            if pca_backbone is None:
                raise ValueError(
                    "pca_backbone is required when show_pca_row=True, "
                    "pca_scope='shared_masked', and pca_data=None."
                )
            pca_data = compute_spair_pair_masked_pca_data(
                pca_backbone,
                pair,
                img_size=img_size,
                device=device,
                pca_hidden_states_sizes=pca_hidden_states_sizes,
                sam_predictor=sam_predictor,
            )
        if pca_scope == "shared" and pca_data is None:
            if pca_backbone is None:
                raise ValueError(
                    "pca_backbone is required when show_pca_row=True, "
                    "pca_scope='shared', and pca_data=None."
                )
            pca_data = compute_spair_pair_pca_data(
                pca_backbone,
                pair,
                img_size=img_size,
                device=device,
                pca_hidden_states_sizes=pca_hidden_states_sizes,
            )
        pca_row_img = make_pca_row_image(
            pair=pair,
            feature_maps=feature_maps,
            pca_data=pca_data,
            method_order=method_order,
            match_img_width=match_img_width,
            pca_scope=pca_scope,
            pca_feature_source=pca_feature_source,
        )
        row_img = pil_img_utils.concat_images(
            [pca_row_img, match_row_img],
            mode="col",
            pad=row_gap,
            pad_color=WHITE,
        )
    else:
        row_img = match_row_img
    if return_pca_data:
        return row_img, pair, results, pca_data
    return row_img, pair, results
