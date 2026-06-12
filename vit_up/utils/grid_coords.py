import torch


def build_regular_grid_centers_xy(n_cells: int):
    """
    Builds a regular grid of cell centers for a query grid of given side length.

    Args:
        n_cells: The number of cells along one side of the query grid.
    Returns:
        grid_centers: (n_cells, n_cells, 2) tensor of (x, y) coordinates of the cell centers, normalized to [0, 1].
    """
    cell_size = 1.0 / n_cells
    centers_1d = (torch.arange(n_cells, dtype=torch.float32) + 0.5) * cell_size
    yy, xx = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
    grid_centers = torch.stack((xx, yy), dim=-1)
    return grid_centers


def build_token_center_grid(
    h_tokens: int,
    w_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Use cell-space coordinates so a one-cell shift has distance 1 for RoPE.
    y_coords = torch.arange(h_tokens, device=device, dtype=dtype) + 0.5
    x_coords = torch.arange(w_tokens, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def compute_rel_position(
    q_xy: torch.Tensor,
    closest_idx: torch.Tensor,
    h_tokens: int,
    w_tokens: int,
) -> torch.Tensor:
    rows = torch.div(closest_idx, w_tokens, rounding_mode="floor")
    cols = torch.remainder(closest_idx, w_tokens)

    center_x = (cols.float() + 0.5) / float(w_tokens)
    center_y = (rows.float() + 0.5) / float(h_tokens)

    token_size_x = 1.0 / float(w_tokens)
    token_size_y = 1.0 / float(h_tokens)

    rel_x = (q_xy[..., 0] - center_x) / token_size_x
    rel_y = (q_xy[..., 1] - center_y) / token_size_y
    return torch.stack((rel_x, rel_y), dim=-1)
