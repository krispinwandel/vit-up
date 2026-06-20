#!/usr/bin/env python3
"""Extract dense ViT-Up video features and render a global PCA visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vit_up.inference.vit_up_wrapper import MODEL_SPECS, ViTUpWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ViT-Up features at every output pixel, fit a global PCA, "
            "and write an RGB feature video."
        )
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--seg-video",
        "--segmentation-video",
        type=Path,
        help="Optional segmentation-mask video; only foreground points are evaluated.",
    )
    parser.add_argument(
        "--seg-threshold",
        type=int,
        default=127,
        help="Foreground threshold applied to grayscale segmentation frames.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--width", type=int, help="Output feature/video width.")
    parser.add_argument("--height", type=int, help="Output feature/video height.")
    parser.add_argument(
        "--model",
        default="vit_up_dinov3_splus",
        choices=sorted(MODEL_SPECS),
        help="Pretrained ViT-Up variant.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--pca-batch-size", type=int, default=16384)
    parser.add_argument(
        "--pca-nframes",
        "--nframes",
        dest="pca_nframes",
        type=int,
        default=1,
        help="Number of evenly spaced video frames used to fit PCA (default: 1).",
    )
    parser.add_argument("--hidden-layer-img-size", type=int, default=448)
    parser.add_argument(
        "--no-bfloat16", action="store_true", help="Use float32 model inference."
    )
    return parser.parse_args()


def validate_positive(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return capture


def inspect_video(path: Path) -> tuple[int, int, float, int]:
    capture = open_video(path)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {source_width}x{source_height}")
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("The input video does not report a valid frame rate.")

    capture = open_video(path)
    frame_count = 0
    with tqdm(desc="Counting frames", unit="frame") as progress:
        while capture.grab():
            frame_count += 1
            progress.update()
    capture.release()
    if frame_count == 0:
        raise RuntimeError("The input video contains no decodable frames.")
    return source_width, source_height, fps, frame_count


def resolve_output_size(
    source_width: int,
    source_height: int,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    validate_positive(width, "width")
    validate_positive(height, "height")
    aspect_ratio = source_width / source_height
    if width is None and height is None:
        return source_width, source_height
    if width is None:
        return max(1, round(height * aspect_ratio)), height  # type: ignore[operator]
    if height is None:
        return width, max(1, round(width / aspect_ratio))
    return width, height


def build_video_query_grid(
    source_width: int, source_height: int, output_width: int, output_height: int
) -> torch.Tensor:
    """Map output pixel centers into normalized coordinates of the padded frame."""
    square_size = max(source_width, source_height)
    left = (square_size - source_width) // 2
    top = (square_size - source_height) // 2
    x = (left + (torch.arange(output_width) + 0.5) * source_width / output_width)
    y = (top + (torch.arange(output_height) + 0.5) * source_height / output_height)
    grid_y, grid_x = torch.meshgrid(y / square_size, x / square_size, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)


def pad_rgb_frame(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    square_size = max(width, height)
    padded = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    top = (square_size - height) // 2
    left = (square_size - width) // 2
    padded[top : top + height, left : left + width] = frame_rgb
    return Image.fromarray(padded)


def read_segmentation_masks(
    path: Path, frame_count: int, output_width: int, output_height: int, threshold: int
) -> np.ndarray:
    masks = np.empty((frame_count, output_height, output_width), dtype=np.bool_)
    capture = open_video(path)
    for frame_index in tqdm(
        range(frame_count), desc="Reading segmentation", unit="frame"
    ):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(
                f"Segmentation video ended before frame {frame_index}."
            )
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            grayscale,
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        )
        masks[frame_index] = resized > threshold
    has_extra_frame = capture.grab()
    capture.release()
    if has_extra_frame:
        raise RuntimeError(
            "Segmentation video has more frames than the input video."
        )
    return masks


def batches(array: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield batches while ensuring IncrementalPCA's final batch is large enough."""
    sample_count = len(array)
    start = 0
    while start < sample_count:
        end = min(start + batch_size, sample_count)
        if sample_count - end < 3:
            end = sample_count
        yield array[start:end]
        start = end


def selected_frame_batches(
    features: np.ndarray,
    frame_indices: np.ndarray,
    batch_size: int,
    masks: np.ndarray | None = None,
) -> Iterator[np.ndarray]:
    points_per_frame = int(np.prod(features.shape[1:3]))
    frame_point_counts = [
        points_per_frame if masks is None else int(masks[index].sum())
        for index in frame_indices
    ]
    selected_sample_count = sum(frame_point_counts)
    array_index = 0
    array_offset = 0
    current_array: np.ndarray | None = None
    emitted = 0
    while emitted < selected_sample_count:
        target_size = min(batch_size, selected_sample_count - emitted)
        if selected_sample_count - emitted - target_size < 3:
            target_size = selected_sample_count - emitted
        pieces = []
        collected = 0
        while collected < target_size:
            while current_array is None:
                frame_index = frame_indices[array_index]
                frame_features = features[frame_index].reshape(
                    points_per_frame, features.shape[-1]
                )
                if masks is not None:
                    frame_features = frame_features[masks[frame_index].reshape(-1)]
                if len(frame_features):
                    current_array = frame_features
                else:
                    array_index += 1
            take = min(
                target_size - collected, len(current_array) - array_offset
            )
            pieces.append(current_array[array_offset : array_offset + take])
            collected += take
            array_offset += take
            if array_offset == len(current_array):
                array_index += 1
                array_offset = 0
                current_array = None
        yield pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        emitted += target_size


def compute_feature_stats(
    features: np.ndarray,
    frame_indices: np.ndarray,
    batch_size: int,
    masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute population mean/std without materializing the full memmap."""
    feature_dim = features.shape[-1]
    selected_sample_count = (
        len(frame_indices) * int(np.prod(features.shape[1:3]))
        if masks is None
        else int(masks[frame_indices].sum())
    )
    count = 0
    mean = np.zeros(feature_dim, dtype=np.float64)
    squared_deviations = np.zeros(feature_dim, dtype=np.float64)

    with tqdm(
        total=selected_sample_count, desc="Computing statistics", unit="point"
    ) as progress:
        for batch in selected_frame_batches(
            features, frame_indices, batch_size, masks
        ):
            batch_float64 = np.asarray(batch, dtype=np.float64)
            batch_count = len(batch_float64)
            batch_mean = batch_float64.mean(axis=0)
            batch_squared_deviations = np.square(
                batch_float64 - batch_mean
            ).sum(axis=0)

            combined_count = count + batch_count
            delta = batch_mean - mean
            squared_deviations += batch_squared_deviations
            if count:
                squared_deviations += (
                    np.square(delta) * count * batch_count / combined_count
                )
            mean += delta * batch_count / combined_count
            count = combined_count
            progress.update(batch_count)

    std = np.sqrt(squared_deviations / count)
    std[std < 1e-8] = 1.0
    return mean, std


def build_cache_metadata(
    args: argparse.Namespace,
    source_width: int,
    source_height: int,
    frame_count: int,
    output_width: int,
    output_height: int,
) -> dict[str, Any]:
    video_stat = args.video.stat()
    metadata = {
        "version": 1,
        "video_path": str(args.video.resolve()),
        "video_size_bytes": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "source_size": [source_height, source_width],
        "frame_count": frame_count,
        "output_size": [output_height, output_width],
        "model": args.model,
        "use_bfloat16": not args.no_bfloat16,
        "hidden_layer_img_size": args.hidden_layer_img_size,
    }
    if args.seg_video is None:
        metadata["segmentation"] = None
    else:
        seg_stat = args.seg_video.stat()
        metadata["segmentation"] = {
            "path": str(args.seg_video.resolve()),
            "size_bytes": seg_stat.st_size,
            "mtime_ns": seg_stat.st_mtime_ns,
            "threshold": args.seg_threshold,
        }
    return metadata


def load_cached_features(
    features_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
) -> np.ndarray | None:
    if not features_path.is_file():
        return None

    try:
        features = np.load(features_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        print(f"Ignoring unreadable feature cache {features_path}: {exc}")
        return None

    expected_shape = tuple(expected_metadata["output_size"])
    expected_shape = (expected_metadata["frame_count"], *expected_shape)
    shape_matches = features.ndim == 4 and features.shape[:3] == expected_shape
    dtype_matches = np.issubdtype(features.dtype, np.floating)
    if not shape_matches or not dtype_matches or features.shape[-1] < 3:
        print(
            f"Ignoring incompatible feature cache {features_path} with shape "
            f"{features.shape} and dtype {features.dtype}."
        )
        return None

    if metadata_path.is_file():
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Ignoring feature cache with unreadable metadata: {exc}")
            return None
        cached_metadata.setdefault("segmentation", None)
        if cached_metadata != expected_metadata:
            print("Feature cache metadata does not match this run; regenerating.")
            return None
    else:
        if expected_metadata.get("segmentation") is not None:
            print(
                "Feature cache predates segmentation metadata; regenerating masked "
                "features."
            )
            return None
        print(
            "Reusing legacy feature cache based on matching shape and dtype; "
            "future runs will also validate its metadata."
        )
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Reusing cached features: {features_path}")
    return features


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.seg_video is not None and not args.seg_video.is_file():
        raise FileNotFoundError(f"Segmentation video not found: {args.seg_video}")
    if not 0 <= args.seg_threshold <= 255:
        raise ValueError("Segmentation threshold must be between 0 and 255.")
    validate_positive(args.query_chunk_size, "query chunk size")
    validate_positive(args.pca_batch_size, "PCA batch size")
    validate_positive(args.pca_nframes, "PCA frame count")
    if args.pca_batch_size < 3:
        raise ValueError("PCA batch size must be at least 3.")

    source_width, source_height, fps, frame_count = inspect_video(args.video)
    output_width, output_height = resolve_output_size(
        source_width, source_height, args.width, args.height
    )
    sample_count = frame_count * output_width * output_height
    if sample_count < 3:
        raise ValueError("At least three total feature points are required for PCA.")
    masks = None
    if args.seg_video is not None:
        masks = read_segmentation_masks(
            args.seg_video,
            frame_count,
            output_width,
            output_height,
            args.seg_threshold,
        )
        foreground_count = int(masks.sum())
        if foreground_count == 0:
            raise ValueError("The segmentation video contains no foreground points.")
        print(
            f"Segmentation foreground: {foreground_count}/{sample_count} points "
            f"({foreground_count / sample_count:.2%})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    features_path = args.output_dir / f"{stem}_features.npy"
    features_metadata_path = args.output_dir / f"{stem}_features.json"
    pca_features_path = args.output_dir / f"{stem}_pca_features.npy"
    pca_path = args.output_dir / f"{stem}_pca.npz"
    output_video_path = args.output_dir / f"{stem}_pca_rgb.mp4"

    print(f"Input: {source_width}x{source_height}, {frame_count} frames at {fps:g} FPS")
    print(f"Dense feature resolution: {output_width}x{output_height}")
    cache_metadata = build_cache_metadata(
        args,
        source_width,
        source_height,
        frame_count,
        output_width,
        output_height,
    )
    features = load_cached_features(
        features_path, features_metadata_path, cache_metadata
    )

    if features is None:
        model = ViTUpWrapper(
            model_name=args.model,
            device=args.device,
            use_bfloat16=not args.no_bfloat16,
            hidden_layer_img_size=args.hidden_layer_img_size,
            query_chunk_size=args.query_chunk_size,
        ).eval()
        query_coords = build_video_query_grid(
            source_width, source_height, output_width, output_height
        )
        capture = open_video(args.video)
        extracted_features: np.memmap | None = None
        temporary_features_path = features_path.with_suffix(".tmp.npy")
        temporary_features_path.unlink(missing_ok=True)
        feature_dim = 0
        frame_progress = tqdm(
            range(frame_count), desc="Extracting features", unit="frame"
        )
        for frame_index in frame_progress:
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(
                    f"Failed to decode frame {frame_index} during extraction."
                )
            frame_mask = (
                np.ones((output_height, output_width), dtype=np.bool_)
                if masks is None
                else masks[frame_index]
            )
            if not frame_mask.any():
                continue
            query_mask = torch.from_numpy(frame_mask.reshape(-1))
            masked_query_coords = query_coords[:, query_mask, :]
            frame_features = model(
                images=pad_rgb_frame(frame), query_coords=masked_query_coords
            ).squeeze(0)
            frame_features_np = frame_features.float().cpu().numpy()
            if extracted_features is None:
                feature_dim = int(frame_features_np.shape[-1])
                extracted_features = np.lib.format.open_memmap(
                    temporary_features_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(
                        frame_count,
                        output_height,
                        output_width,
                        feature_dim,
                    ),
                )
            extracted_features[frame_index].reshape(-1, feature_dim)[
                frame_mask.reshape(-1)
            ] = frame_features_np
        capture.release()
        assert extracted_features is not None
        extracted_features.flush()
        temporary_features_path.replace(features_path)
        features_metadata_path.write_text(
            json.dumps(cache_metadata, indent=2) + "\n", encoding="utf-8"
        )
        features = extracted_features

    feature_dim = int(features.shape[-1])

    if feature_dim < 3:
        raise ValueError(
            "PCA needs at least three feature dimensions, but the model returned "
            f"{feature_dim}."
        )
    pca_frame_count = min(args.pca_nframes, frame_count)
    if pca_frame_count == 1:
        pca_frame_indices = np.asarray([frame_count // 2], dtype=np.int64)
    else:
        pca_frame_indices = np.linspace(
            0, frame_count - 1, num=pca_frame_count, dtype=np.int64
        )
    pca_sample_count = (
        pca_frame_count * output_height * output_width
        if masks is None
        else int(masks[pca_frame_indices].sum())
    )
    if pca_sample_count < 3:
        raise ValueError(
            "The selected PCA frames contain fewer than three feature points. "
            "Increase --pca-nframes or the output resolution."
        )
    print(
        f"Fitting PCA on {pca_frame_count} evenly spaced frame(s): "
        f"{pca_frame_indices.tolist()}"
    )
    mean, std = compute_feature_stats(
        features, pca_frame_indices, args.pca_batch_size, masks
    )
    pca = IncrementalPCA(n_components=3, batch_size=args.pca_batch_size)
    with tqdm(total=pca_sample_count, desc="Fitting PCA", unit="point") as progress:
        for batch in selected_frame_batches(
            features, pca_frame_indices, args.pca_batch_size, masks
        ):
            pca.partial_fit((batch - mean) / std)
            progress.update(len(batch))

    pca_features = np.lib.format.open_memmap(
        pca_features_path,
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, output_height, output_width, 3),
    )
    pca_features[:] = 0
    apply_sample_count = sample_count if masks is None else int(masks.sum())
    rgb_min = np.full(3, np.inf, dtype=np.float32)
    rgb_max = np.full(3, -np.inf, dtype=np.float32)
    with tqdm(
        total=apply_sample_count, desc="Applying PCA", unit="point"
    ) as progress:
        for frame_index in range(frame_count):
            frame_mask = (
                np.ones((output_height, output_width), dtype=np.bool_)
                if masks is None
                else masks[frame_index]
            )
            masked_features = features[frame_index].reshape(-1, feature_dim)[
                frame_mask.reshape(-1)
            ]
            masked_pca_features = pca_features[frame_index].reshape(-1, 3)[
                frame_mask.reshape(-1)
            ]
            offset = 0
            for batch in batches(masked_features, args.pca_batch_size):
                transformed = pca.transform((batch - mean) / std).astype(np.float32)
                masked_pca_features[offset : offset + len(batch)] = transformed
                rgb_min = np.minimum(rgb_min, transformed.min(axis=0))
                rgb_max = np.maximum(rgb_max, transformed.max(axis=0))
                offset += len(batch)
                progress.update(len(batch))
            pca_features[frame_index].reshape(-1, 3)[
                frame_mask.reshape(-1)
            ] = masked_pca_features
    pca_features.flush()
    np.savez(
        pca_path,
        mean=mean,
        std=std,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        rgb_min=rgb_min,
        rgb_max=rgb_max,
        fit_frame_indices=pca_frame_indices,
        source_size=np.asarray([source_height, source_width]),
        output_size=np.asarray([output_height, output_width]),
        fps=np.asarray(fps),
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")
    rgb_range = np.maximum(rgb_max - rgb_min, 1e-8)
    for frame_index in tqdm(range(frame_count), desc="Writing RGB video", unit="frame"):
        frame_rgb = np.clip(
            (pca_features[frame_index] - rgb_min) / rgb_range * 255.0, 0, 255
        ).astype(np.uint8)
        if masks is not None:
            frame_rgb[~masks[frame_index]] = 0
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"Features: {features_path}")
    print(f"PCA features: {pca_features_path}")
    print(f"PCA parameters: {pca_path}")
    print(f"RGB video: {output_video_path}")


if __name__ == "__main__":
    main()
