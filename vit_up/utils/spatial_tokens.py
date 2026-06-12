import torch


def strip_prefix_to_spatial_tokens(
    layer: torch.Tensor,
    target_side: int,
) -> torch.Tensor:
    expected_tokens = int(target_side) * int(target_side)
    n_prefix = layer.shape[1] - expected_tokens
    if n_prefix < 0:
        raise ValueError(
            "Layer does not contain enough tokens for requested spatial side. "
            f"Got seq_len={layer.shape[1]}, target_side={target_side}."
        )
    return layer[:, n_prefix:]


def mean_pool_spatial_tokens(
    layer: torch.Tensor,
    source_side: int,
    target_side: int,
) -> torch.Tensor:
    if source_side == target_side:
        return layer
    if source_side % target_side != 0:
        raise ValueError(
            "source_side must be divisible by target_side for mean pooling. "
            f"Got source_side={source_side}, target_side={target_side}."
        )

    bsz, n_tokens, dim = layer.shape
    expected_tokens = source_side * source_side
    if n_tokens != expected_tokens:
        raise ValueError(
            "Predicted layer token count does not match source grid side. "
            f"Got n_tokens={n_tokens}, expected={expected_tokens}."
        )

    factor = source_side // target_side
    layer = layer.view(bsz, source_side, source_side, dim)
    layer = layer.view(bsz, target_side, factor, target_side, factor, dim)
    layer = layer.mean(dim=(2, 4))
    return layer.reshape(bsz, target_side * target_side, dim)


def spatial_tokens_to_hwc(
    hidden_state: torch.Tensor,
    n_prefix_tokens: int = 5,
    check_square_grid: bool = False,
) -> torch.Tensor:
    """Convert hidden state from (B, T, C) to spatial (B, H, W, C) after dropping prefix."""
    spatial_tokens = hidden_state[:, n_prefix_tokens:, :]
    n_spatial_tokens = spatial_tokens.shape[1]
    side = int(math.isqrt(n_spatial_tokens))
    if check_square_grid and side * side != n_spatial_tokens:
        raise ValueError(
            "Hidden state spatial token count must form a square grid. "
            f"Got {n_spatial_tokens} tokens after removing n_prefix={n_prefix_tokens}."
        )
    return spatial_tokens.reshape(
        spatial_tokens.shape[0], side, side, spatial_tokens.shape[-1]
    )


def fetch_closest_token(
    q_xy: torch.Tensor,
    hidden_state_hwc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    bsz, h_tokens, w_tokens, d_model = hidden_state_hwc.shape
    hidden_state_flat = hidden_state_hwc.reshape(bsz, h_tokens * w_tokens, d_model)

    grid_x = (q_xy[..., 0] * w_tokens) - 0.5
    grid_y = (q_xy[..., 1] * h_tokens) - 0.5
    closest_col = torch.round(grid_x).long().clamp(0, w_tokens - 1)
    closest_row = torch.round(grid_y).long().clamp(0, h_tokens - 1)
    closest_idx = closest_row * w_tokens + closest_col

    batch_idx = torch.arange(bsz, device=hidden_state_hwc.device)[:, None]
    closest_token = hidden_state_flat[batch_idx, closest_idx, :]
    return closest_token, closest_idx, h_tokens, w_tokens


def tlbr_indices_to_rel_pos(
    tlbr_indices: torch.Tensor,
    q_xy_normalized: torch.Tensor,
    h_tokens: int,
    w_tokens: int,
) -> torch.Tensor:
    """
    Args:
        tlbr_indices: (B, N_q, 4, 2) integer indices of the tlbr tokens in (col, row) format.
        q_xy_normalized: (B, N_q, 2) normalized to [0, 1] relative to image size.
    Returns:
        rel_pos: (B, N_q, 4, 2) relative position of each query point to the tlbr tokens, in units of token size (so a value of 1.0 means the query point is at the center of the corresponding token).
    """
    tlbr_pos_normalized = tlbr_indices.float() / torch.tensor(
        [w_tokens, h_tokens], device=tlbr_indices.device
    )
    rel_pos = (q_xy_normalized[:, :, None, :] - tlbr_pos_normalized) * torch.tensor(
        [w_tokens, h_tokens], device=tlbr_indices.device
    )
    return rel_pos


def build_tlbr_indices(
    h_tokens: int,
    w_tokens: int,
    q_xy_normalized: torch.Tensor,
):
    grid_x = (q_xy_normalized[..., 0] * w_tokens) - 0.5
    grid_y = (q_xy_normalized[..., 1] * h_tokens) - 0.5

    left_col = torch.floor(grid_x).long().clamp(0, w_tokens - 1)
    right_col = torch.ceil(grid_x).long().clamp(0, w_tokens - 1)
    top_row = torch.floor(grid_y).long().clamp(0, h_tokens - 1)
    bottom_row = torch.ceil(grid_y).long().clamp(0, h_tokens - 1)

    tlbr_indices = torch.stack(
        [
            torch.stack((left_col, top_row), dim=-1),
            torch.stack((right_col, top_row), dim=-1),
            torch.stack((left_col, bottom_row), dim=-1),
            torch.stack((right_col, bottom_row), dim=-1),
        ],
        dim=-2,
    )  # (B, N_q, 4, 2)

    return tlbr_indices


def fetch_tlbr_tokens_from_tlbr_indices(
    tlbr_indices: torch.Tensor,
    hidden_state_hwc: torch.Tensor,
) -> torch.Tensor:
    bsz, h_tokens, w_tokens, d_model = hidden_state_hwc.shape
    batch_idx = torch.arange(bsz, device=hidden_state_hwc.device)[:, None, None]
    tlbr_tokens = hidden_state_hwc[
        batch_idx,
        tlbr_indices[..., 1],  # row index
        tlbr_indices[..., 0],  # col index
        :,
    ]  # (B, N_q, 4, C)
    return tlbr_tokens


def fetch_tlbr_tokens(
    q_xy_normalized: torch.Tensor,
    hidden_state_hwc: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        q_xy_normalized: (B, N_q, 2) normalized to [0, 1] relative to image size.
        hidden_state_hwc: (B, H, W, C) spatial tokens in HWC format.
    Returns:
        tlbr_tokens: (B, N_q, 4, C) tokens of the 2x2 grid cell containing each query point,
            ordered as top-left, top-right, bottom-left, bottom-right.
        tlbr_indices: (B, N_q, 4, 2) integer indices of the tlbr tokens in (col, row) format.
    """
    bsz, h_tokens, w_tokens, d_model = hidden_state_hwc.shape
    grid_x = (q_xy_normalized[..., 0] * w_tokens) - 0.5
    grid_y = (q_xy_normalized[..., 1] * h_tokens) - 0.5

    left_col = torch.floor(grid_x).long().clamp(0, w_tokens - 1)
    right_col = torch.ceil(grid_x).long().clamp(0, w_tokens - 1)
    top_row = torch.floor(grid_y).long().clamp(0, h_tokens - 1)
    bottom_row = torch.ceil(grid_y).long().clamp(0, h_tokens - 1)

    batch_idx = torch.arange(bsz, device=hidden_state_hwc.device)[:, None]
    tlbr_tokens = hidden_state_hwc[
        batch_idx,
        [top_row, top_row, bottom_row, bottom_row],
        [left_col, right_col, left_col, right_col],
        :,
    ]
    tlbr_tokens.permute(0, 2, 1, 3)  # (B, N_q, 4, C)

    tlbr_indices = torch.stack(
        [
            torch.stack((left_col, top_row), dim=-1),
            torch.stack((right_col, top_row), dim=-1),
            torch.stack((left_col, bottom_row), dim=-1),
            torch.stack((right_col, bottom_row), dim=-1),
        ],
        dim=-2,
    )  # (B, N_q, 4, 2)

    return tlbr_tokens, tlbr_indices
