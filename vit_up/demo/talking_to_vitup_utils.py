from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel
from transformers.models.auto.auto_factory import get_class_from_dynamic_module

from vit_up.inference.vit_up_wrapper import ViTUpWrapper
from vit_up.utils import pil_img_utils


ThresholdMode = Literal["talk2dino_bg", "otsu", "percentile", "manual"]


@dataclass
class DenseImageFeatures:
    image: Image.Image
    image_path: Path
    dense_features: torch.Tensor
    query_size: tuple[int, int]
    pca_rgb: Image.Image


@dataclass
class TextSimilarityResult:
    image: Image.Image
    image_path: Path
    prompt: str
    similarity: np.ndarray
    sigmoid_score: np.ndarray
    cosine_display_01: np.ndarray
    cosine_overlay: Image.Image
    query_size: tuple[int, int]
    pca_rgb: Image.Image
    pamr_score: np.ndarray | None = None


def get_repo_root() -> Path:
    repo_root = Path.cwd().resolve()
    if repo_root.name == "notebooks":
        repo_root = repo_root.parent
    return repo_root


def list_asset_images(assets_dir: Path) -> list[Path]:
    image_paths = sorted(
        p for p in assets_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {assets_dir}")
    return image_paths


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_vit_up_model(device: str) -> ViTUpWrapper:
    return ViTUpWrapper(
        "vit_up_dinov3_base",
        device=device,
        use_bfloat16=(device == "cuda"),
        hidden_layer_img_size=448,
        query_chunk_size=8192,
    ).eval()


def load_talk2dino_model(device: str, model_id: str = "lorebianchi98/Talk2DINOv3-ViTB"):
    # Talk2DINO's custom class currently skips PreTrainedModel.__init__, which
    # leaves this Transformers 5 metadata field unset during from_pretrained().
    talk2dino_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    talk2dino_class = get_class_from_dynamic_module(
        talk2dino_config.auto_map["AutoModel"],
        model_id,
    )
    talk2dino_class.all_tied_weights_keys = {}

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        warnings.filterwarnings(
            "ignore",
            message=r"for .*: copying from a non-meta parameter in the checkpoint to a meta parameter.*",
            category=UserWarning,
        )
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(device).eval()

    # The remote CLIP blocks can keep meta attention masks with recent Transformers.
    # Rebuild them as real tensors so encode_text works reliably.
    base_attn_mask = model.clip_model.build_attention_mask()
    for block in model.clip_model.transformer.resblocks:
        block.attn_mask = base_attn_mask.clone()

    meta_params = [name for name, param in model.named_parameters() if param.is_meta]
    if meta_params:
        raise RuntimeError(f"Talk2DINO still has meta parameters after loading: {meta_params[:5]}")

    return model


def make_query_grid(width: int, height: int, max_side: int, device: str) -> tuple[torch.Tensor, int, int]:
    scale = max_side / max(width, height)
    query_w = max(1, int(round(width * scale)))
    query_h = max(1, int(round(height * scale)))

    xs = (torch.arange(query_w, device=device, dtype=torch.float32) + 0.5) / query_w
    ys = (torch.arange(query_h, device=device, dtype=torch.float32) + 0.5) / query_h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    query_coords = torch.stack([xx, yy], dim=-1).reshape(1, query_h * query_w, 2)
    return query_coords, query_h, query_w


def minmax_for_display(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max <= value_min:
        return np.zeros_like(values, dtype=np.float32)
    return (values - value_min) / (value_max - value_min)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    values = np.asarray(values, dtype=np.float32)
    hist, bin_edges = np.histogram(values.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * centers)
    sum_total = sum_bg[-1]
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return 0.5
    mean_bg = np.zeros_like(centers)
    mean_fg = np.zeros_like(centers)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1
    return float(centers[int(np.argmax(between))])


def resolve_threshold(
    score_01: np.ndarray,
    mode: ThresholdMode,
    manual_threshold: float,
    percentile: float,
) -> float:
    if mode == "talk2dino_bg":
        return 0.55
    if mode == "otsu":
        return otsu_threshold(score_01)
    if mode == "percentile":
        return float(np.percentile(score_01, percentile))
    if mode == "manual":
        return float(manual_threshold)
    raise ValueError(f"Unknown threshold mode: {mode}")


def overlay_score(image: Image.Image, score_01: np.ndarray, alpha: float = 0.55) -> Image.Image:
    heatmap = pil_img_utils.heatmap_to_rgb(score_01, colormap="magma", heat_min=0.0, heat_max=1.0)
    heatmap = heatmap.resize(image.size, resample=Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha=alpha)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.resize(size, resample=Image.Resampling.NEAREST)
    return np.array(mask_img) > 0


def pca_rgb_from_features(
    dense_features: torch.Tensor,
    image_size: tuple[int, int],
    max_samples: int = 100_000,
) -> Image.Image:
    h, w, d = dense_features.shape
    features = dense_features.reshape(h * w, d).float()
    sample_step = max(1, features.shape[0] // max_samples)
    sample = features[::sample_step]

    mean = sample.mean(dim=0, keepdim=True)
    std = sample.std(dim=0, keepdim=True).clamp_min(1e-6)
    sample = (sample - mean) / std
    covariance = sample.T @ sample / max(sample.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -3:].flip(dims=(1,))

    projected = ((features - mean) / std) @ components
    lo = torch.quantile(projected, 0.01, dim=0)
    hi = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)

    rgb = (projected.reshape(h, w, 3).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize(image_size, resample=Image.Resampling.BILINEAR)


def extract_dense_image_features(
    vit_up_model: ViTUpWrapper,
    image_path: str | Path,
    max_side: int,
    device: str,
) -> DenseImageFeatures:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    query_coords, query_h, query_w = make_query_grid(*image.size, max_side=max_side, device=device)

    with torch.inference_mode():
        dense_features = vit_up_model(image, query_coords=query_coords)
        dense_features = dense_features.reshape(query_h, query_w, -1).float()
        dense_features = F.normalize(dense_features, dim=-1)
        pca_rgb = pca_rgb_from_features(dense_features, image.size)

    return DenseImageFeatures(
        image=image,
        image_path=image_path,
        dense_features=dense_features,
        query_size=(query_w, query_h),
        pca_rgb=pca_rgb,
    )


def encode_text(talk2dino_model, prompt: str) -> torch.Tensor:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    with torch.inference_mode():
        text_features = talk2dino_model.encode_text(prompt).float()
        text_features = text_features.reshape(-1, text_features.shape[-1])[:1]
        return F.normalize(text_features, dim=-1)


def compute_similarity(
    image_features: DenseImageFeatures,
    text_features: torch.Tensor,
    prompt: str,
) -> TextSimilarityResult:
    with torch.inference_mode():
        similarity = torch.einsum("hwd,nd->hwn", image_features.dense_features, text_features).squeeze(-1)

    similarity_np = similarity.detach().cpu().numpy()
    sigmoid_score = (1.0 / (1.0 + np.exp(-similarity_np))).astype(np.float32)
    cosine_display_01 = minmax_for_display(similarity_np)

    return TextSimilarityResult(
        image=image_features.image,
        image_path=image_features.image_path,
        prompt=prompt.strip(),
        similarity=similarity_np,
        sigmoid_score=sigmoid_score,
        cosine_display_01=cosine_display_01,
        cosine_overlay=overlay_score(image_features.image, cosine_display_01),
        query_size=image_features.query_size,
        pca_rgb=image_features.pca_rgb,
    )


def pamr_refine(image: Image.Image, score_01: np.ndarray, num_iter: int = 10) -> np.ndarray:
    # Single-class port of Talk2DINO's PAMR settings: 10 iterations and
    # dilations [1, 2, 4, 8, 12, 24] with image-guided local affinities.
    dilations = [1, 2, 4, 8, 12, 24]
    rgb = np.asarray(image.resize(score_01.shape[::-1], Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    mask = torch.from_numpy(score_01.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    def affinity_kernel(copy: bool, include_center: bool = False) -> torch.Tensor:
        points = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
        if include_center:
            points = [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            ]
        weight = torch.zeros(len(points), 1, 3, 3)
        for i, (y, x_coord) in enumerate(points):
            weight[i, 0, y, x_coord] = 1.0 if copy else -1.0
            if not copy:
                weight[i, 0, 1, 1] = 1.0
        return weight

    diff_weight = affinity_kernel(copy=False)
    copy_weight = affinity_kernel(copy=True)
    std_weight = affinity_kernel(copy=True, include_center=True)

    def local_affinity(values: torch.Tensor, weight: torch.Tensor, dilations: list[int]) -> torch.Tensor:
        b, k, h, w = values.shape
        values = values.reshape(b * k, 1, h, w)
        outputs = []
        for dilation in dilations:
            padded = F.pad(values, [dilation] * 4, mode="replicate")
            outputs.append(F.conv2d(padded, weight, dilation=dilation))
        return torch.cat(outputs, dim=1).reshape(b, k, -1, h, w)

    image_std = local_affinity(x, std_weight, dilations).std(dim=2, keepdim=True)
    image_affinity = -local_affinity(x, diff_weight, dilations).abs() / (1e-8 + 0.1 * image_std)
    image_affinity = image_affinity.mean(dim=1, keepdim=True).softmax(dim=2)

    for _ in range(num_iter):
        mask_affinity = local_affinity(mask, copy_weight, dilations)
        mask = (mask_affinity * image_affinity).sum(dim=2)

    return mask.squeeze(0).squeeze(0).clamp(0.0, 1.0).numpy()


def render_result(
    result: TextSimilarityResult,
    threshold_mode: ThresholdMode,
    threshold: float,
    percentile: float,
    refine: bool,
) -> None:
    image = result.image
    if refine:
        if result.pamr_score is None:
            result.pamr_score = pamr_refine(image, result.sigmoid_score)
        score = result.pamr_score
    else:
        score = result.sigmoid_score
    resolved_threshold = resolve_threshold(score, threshold_mode, threshold, percentile)
    mask_lowres = score >= resolved_threshold
    mask = resize_mask(mask_lowres, image.size)
    segmented = pil_img_utils.apply_mask_black(image, mask)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    cosine_heatmap = pil_img_utils.heatmap_to_rgb(
        result.cosine_display_01,
        colormap="magma",
        heat_min=0.0,
        heat_max=1.0,
    ).resize(image.size, resample=Image.Resampling.BILINEAR)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    axes[0].imshow(image)
    axes[0].set_title(f"Image: {result.image_path.name}")
    axes[1].imshow(result.pca_rgb)
    axes[1].set_title(f"ViT-Up PCA: {result.query_size[0]}x{result.query_size[1]}")
    axes[2].imshow(result.cosine_overlay)
    axes[2].set_title(f"Cosine heatmap, min-max display: {result.prompt}")
    axes[3].imshow(cosine_heatmap)
    axes[3].set_title("Cosine heatmap")
    axes[4].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
    axes[4].set_title(f"Score mask >= {resolved_threshold:.2f}")
    axes[5].imshow(segmented)
    axes[5].set_title("Segmented image")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


class TalkingToVitUpRunner:
    def __init__(self, vit_up_model: ViTUpWrapper, talk2dino_model, device: str):
        self.vit_up_model = vit_up_model
        self.talk2dino_model = talk2dino_model
        self.device = device
        self._image_feature_cache: dict[tuple[str, int], DenseImageFeatures] = {}
        self._text_cache: dict[str, torch.Tensor] = {}
        self._similarity_cache: dict[tuple[str, int, str], TextSimilarityResult] = {}

    def get_image_features(self, image_path: str | Path, max_side: int) -> DenseImageFeatures:
        key = (str(Path(image_path).resolve()), int(max_side))
        if key not in self._image_feature_cache:
            self._image_feature_cache[key] = extract_dense_image_features(
                self.vit_up_model,
                image_path=image_path,
                max_side=max_side,
                device=self.device,
            )
        return self._image_feature_cache[key]

    def get_text_features(self, prompt: str) -> torch.Tensor:
        key = prompt.strip()
        if key not in self._text_cache:
            self._text_cache[key] = encode_text(self.talk2dino_model, key)
        return self._text_cache[key]

    def run(self, image_path: str | Path, prompt: str, max_side: int) -> TextSimilarityResult:
        prompt = prompt.strip()
        key = (str(Path(image_path).resolve()), int(max_side), prompt)
        if key not in self._similarity_cache:
            image_features = self.get_image_features(image_path, max_side)
            text_features = self.get_text_features(prompt)
            self._similarity_cache[key] = compute_similarity(image_features, text_features, prompt)
        return self._similarity_cache[key]
