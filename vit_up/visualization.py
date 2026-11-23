import torch


class PCA:
    """Three-component PCA of features, mapped to RGB with robust per-component ranges."""

    def __init__(self, features: torch.Tensor, quantile: float = 0.01):
        features = features.reshape(-1, features.shape[-1]).float()
        self.mean = features.mean(dim=0)
        _, _, v = torch.pca_lowrank(features - self.mean, q=3, center=False)
        self.components = v[:, :3]
        projected = self.project(features)
        self.low = projected.quantile(quantile, dim=0)
        self.high = projected.quantile(1 - quantile, dim=0)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        return (features.float() - self.mean) @ self.components

    def rgb(self, features: torch.Tensor) -> torch.Tensor:
        """(..., C) features to (..., 3) colors in [0, 1]."""
        return ((self.project(features) - self.low) / (self.high - self.low)).clamp(0, 1)
