import torch


def regular_grid(size: int, device=None) -> torch.Tensor:
    """Normalized (x, y) centers of a size x size grid of cells, shape (size, size, 2)."""
    centers = (torch.arange(size, dtype=torch.float32, device=device) + 0.5) * (1.0 / size)
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack((x, y), dim=-1)
