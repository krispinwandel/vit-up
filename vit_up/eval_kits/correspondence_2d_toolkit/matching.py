"""Category-local feature caching and batched nearest-neighbor correspondence."""

from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .spair import Annotation
from .geometry import SquareImageGeometry


class CategoryMatcher:
    def __init__(self, device="cuda", cache_device="cuda", query_chunk_size=256, target_chunk_size=16384):
        self.device = torch.device(device)
        self.cache_device = torch.device(cache_device)
        self.query_chunk_size = query_chunk_size
        self.target_chunk_size = target_chunk_size
        self.features = {}
        self.queries = {}
        self.annotations = {}
        self.grid_sizes = {}

    @torch.inference_mode()
    def add_image(self, path: str, features: torch.Tensor, annotation: Annotation):
        """Cache L2-normalized features and the features at the annotated keypoints' cells."""
        if features.ndim != 3 or features.shape[0] != features.shape[1]:
            raise ValueError("Expected square HWC feature maps.")
        grid_size = features.shape[0]
        features = features.detach().to(device=self.device, dtype=torch.float32)
        geometry = SquareImageGeometry(annotation.img_width, annotation.img_height)
        points = torch.as_tensor(annotation.kp_xy, device=self.device).reshape(-1, 2)
        with torch.autocast(device_type=self.device.type, enabled=False):
            normalized = F.normalize(features, dim=-1, eps=1e-8)
            indices = geometry.keypoint_indices(points, grid_size)
            queries = normalized[indices[:, 1], indices[:, 0]]
        self.queries[path] = queries
        self.features[path] = normalized.reshape(-1, normalized.shape[-1]).to(self.cache_device)
        self.annotations[path] = annotation
        self.grid_sizes[path] = grid_size

    @torch.inference_mode()
    def _argmax(self, queries: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bound score memory; preserve first-index tie breaking across chunks."""
        # Keep one batched pass. FP32 reduction order can change near-tied rankings.
        indices = []
        for query_batch in queries.split(self.query_chunk_size):
            best_scores = query_batch.new_full((len(query_batch),), -torch.inf)
            best_indices = torch.zeros(len(query_batch), device=self.device, dtype=torch.long)
            for start in range(0, len(target), self.target_chunk_size):
                scores = query_batch @ target[start:start + self.target_chunk_size].T
                values, local_indices = scores.max(dim=1)
                better = values > best_scores
                best_indices = torch.where(better, local_indices + start, best_indices)
                best_scores = torch.maximum(best_scores, values)
            indices.append(best_indices)
        return torch.cat(indices) if indices else torch.empty(0, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def predict(self, image_pairs: list[str], show_progress=True) -> dict:
        """Predicted target-image (x, y) of every source keypoint, keyed by (source, target) path."""
        sources_by_target = defaultdict(dict)
        for source, target in zip(image_pairs[::2], image_pairs[1::2]):
            sources_by_target[target][source] = None

        predictions = {}
        previous_precision = torch.get_float32_matmul_precision()
        try:
            # Avoid TF32/autocast changing nearest-neighbor rankings.
            torch.set_float32_matmul_precision("highest")
            with torch.autocast(device_type=self.device.type, enabled=False):
                for target_path, sources in tqdm(sources_by_target.items(), desc="Matching target images",
                                                 disable=not show_progress):
                    target = self.features[target_path].to(self.device)
                    queries = torch.cat([self.queries[source] for source in sources])
                    matches = self._argmax(queries, target)
                    grid_size = self.grid_sizes[target_path]
                    annotation = self.annotations[target_path]
                    geometry = SquareImageGeometry(annotation.img_width, annotation.img_height)
                    xy = torch.stack((matches % grid_size, matches // grid_size), dim=-1)
                    xy = geometry.grid_to_image(xy, grid_size).cpu().numpy()
                    offset = 0
                    for source in sources:
                        count = len(self.queries[source])
                        predictions[source, target_path] = xy[offset:offset + count]
                        offset += count
        finally:
            torch.set_float32_matmul_precision(previous_precision)
        return predictions
