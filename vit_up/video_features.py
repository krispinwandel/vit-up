"""Lazy readers for dense video feature arrays."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np


PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class VideoFeatureFrame:
    """Aligned data for one encoded video frame."""

    features: np.ndarray
    mask: np.ndarray | None = None
    rgb: np.ndarray | None = None
    pca_rgb: np.ndarray | None = None


class LazyVideoFeatureReader:
    """Read frames from encoder-produced video feature artifacts."""

    def __init__(
        self,
        features_path: PathLike,
        *,
        metadata_path: PathLike | None = None,
        video_path: PathLike | None = None,
        seg_video_path: PathLike | None = None,
        pca_features_path: PathLike | None = None,
        pca_path: PathLike | None = None,
    ) -> None:
        self.features_path = Path(features_path)
        if not self.features_path.is_file():
            raise FileNotFoundError(f"Feature file not found: {self.features_path}")

        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else self.features_path.with_suffix(".json")
        )
        self.metadata = self._read_metadata(self.metadata_path)

        features = np.load(self.features_path, mmap_mode="r")
        if features.ndim != 4:
            raise ValueError(
                "Expected video features with shape (T, H, W, D), got "
                f"{features.shape}."
            )
        if not np.issubdtype(features.dtype, np.floating):
            raise ValueError(
                "Expected floating-point video features, got "
                f"dtype={features.dtype}."
            )

        self._features = features
        self.video_path = self._resolve_video_path(video_path)
        self.seg_video_path = self._resolve_seg_video_path(seg_video_path)
        self.seg_threshold = self._resolve_seg_threshold()
        inferred_stem = self._infer_encoder_stem()
        self.pca_features_path = (
            Path(pca_features_path)
            if pca_features_path is not None
            else self.features_path.with_name(f"{inferred_stem}_pca_features.npy")
        )
        self.pca_path = (
            Path(pca_path)
            if pca_path is not None
            else self.features_path.with_name(f"{inferred_stem}_pca.npz")
        )
        self._pca_features = self._load_pca_features()
        self._pca_rgb_min: np.ndarray | None = None
        self._pca_rgb_range: np.ndarray | None = None
        self._load_pca_rgb_stats()

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(int(dim) for dim in self._features.shape)

    @property
    def frame_count(self) -> int:
        return int(self._features.shape[0])

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self._features.shape[1:])

    def read_features(self, frame_idx: int) -> np.ndarray:
        """Return one ``(H, W, D)`` feature frame without loading the full file."""
        return self._features[self._normalize_frame_idx(frame_idx)]

    def read_mask_frame(self, frame_idx: int) -> np.ndarray | None:
        """Return one ``(H, W)`` mask frame if a segmentation video is configured."""
        frame_idx = self._normalize_frame_idx(frame_idx)
        if self.seg_video_path is None:
            return None

        frame_bgr = self._read_video_frame(self.seg_video_path, frame_idx)
        grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        output_height, output_width = self.shape[1:3]
        resized = cv2.resize(
            grayscale,
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized > self.seg_threshold

    def read_rgb_frame(self, frame_idx: int) -> np.ndarray | None:
        """Return one RGB video frame resized to the feature frame resolution."""
        frame_idx = self._normalize_frame_idx(frame_idx)
        if self.video_path is None:
            return None

        frame_bgr = self._read_video_frame(self.video_path, frame_idx)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        output_height, output_width = self.shape[1:3]
        if frame_rgb.shape[:2] != (output_height, output_width):
            frame_rgb = cv2.resize(
                frame_rgb,
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame_rgb

    def read_pca_rgb_frame(
        self, frame_idx: int, mask: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Return one PCA RGB frame using the encoder's saved normalization."""
        frame_idx = self._normalize_frame_idx(frame_idx)
        if (
            self._pca_features is None
            or self._pca_rgb_min is None
            or self._pca_rgb_range is None
        ):
            return None

        frame_rgb = np.clip(
            (self._pca_features[frame_idx] - self._pca_rgb_min)
            / self._pca_rgb_range
            * 255.0,
            0,
            255,
        ).astype(np.uint8)
        if mask is None:
            mask = self.read_mask_frame(frame_idx)
        if mask is not None:
            frame_rgb[~mask] = 0
        return frame_rgb

    def read_ft_frame(self, frame_idx: int) -> VideoFeatureFrame:
        """Return features plus available mask, RGB, and PCA RGB for one frame."""
        frame_idx = self._normalize_frame_idx(frame_idx)
        mask = self.read_mask_frame(frame_idx)
        return VideoFeatureFrame(
            features=self._features[frame_idx],
            mask=mask,
            rgb=self.read_rgb_frame(frame_idx),
            pca_rgb=self.read_pca_rgb_frame(frame_idx, mask),
        )

    def _normalize_frame_idx(self, frame_idx: int) -> int:
        if not isinstance(frame_idx, (int, np.integer)):
            raise TypeError(f"frame_idx must be an int, got {type(frame_idx)!r}.")
        if not -self.frame_count <= int(frame_idx) < self.frame_count:
            raise IndexError(
                f"frame_idx {frame_idx} is out of bounds for "
                f"{self.frame_count} frames."
            )
        return int(frame_idx) % self.frame_count

    def _read_metadata(self, metadata_path: Path) -> dict[str, Any]:
        if not metadata_path.is_file():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read feature metadata {metadata_path}: {exc}"
            ) from exc

    def _infer_encoder_stem(self) -> str:
        stem = self.features_path.stem
        return stem[: -len("_features")] if stem.endswith("_features") else stem

    def _resolve_video_path(self, video_path: PathLike | None) -> Path | None:
        if video_path is not None:
            path = Path(video_path)
        else:
            metadata_video_path = self.metadata.get("video_path")
            path = Path(metadata_video_path) if metadata_video_path else None
        return path if path is not None and path.is_file() else None

    def _resolve_seg_video_path(self, seg_video_path: PathLike | None) -> Path | None:
        if seg_video_path is not None:
            path = Path(seg_video_path)
        else:
            segmentation = self.metadata.get("segmentation")
            metadata_seg_path = (
                segmentation.get("path") if isinstance(segmentation, dict) else None
            )
            path = Path(metadata_seg_path) if metadata_seg_path else None
        return path if path is not None and path.is_file() else None

    def _resolve_seg_threshold(self) -> int:
        segmentation = self.metadata.get("segmentation")
        if isinstance(segmentation, dict):
            return int(segmentation.get("threshold", 127))
        return 127

    def _load_pca_features(self) -> np.ndarray | None:
        if not self.pca_features_path.is_file():
            return None
        pca_features = np.load(self.pca_features_path, mmap_mode="r")
        expected_shape = (*self.shape[:3], 3)
        if pca_features.shape != expected_shape:
            raise ValueError(
                "Expected PCA features with shape "
                f"{expected_shape}, got {pca_features.shape}."
            )
        return pca_features

    def _load_pca_rgb_stats(self) -> None:
        if not self.pca_path.is_file():
            return
        with np.load(self.pca_path) as pca_data:
            rgb_min = np.asarray(pca_data["rgb_min"], dtype=np.float32)
            rgb_max = np.asarray(pca_data["rgb_max"], dtype=np.float32)
        self._pca_rgb_min = rgb_min
        self._pca_rgb_range = np.maximum(rgb_max - rgb_min, 1e-8)

    def _read_video_frame(self, path: Path, frame_idx: int) -> np.ndarray:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_idx} from {path}")
        return frame


def make_read_ft_frame(
    features_path: PathLike, **reader_kwargs: Any
) -> Callable[[int], VideoFeatureFrame]:
    """Open ``features_path`` lazily and return ``read_ft_frame(frame_idx)``."""
    reader = LazyVideoFeatureReader(features_path, **reader_kwargs)
    return reader.read_ft_frame
