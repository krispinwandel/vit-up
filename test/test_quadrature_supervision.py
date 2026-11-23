import torch

from vit_up.supervision.strategies.quadrature import CellIntegralQuadratureSupervision


def _cell_average_monomial(
    power: int, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    """Average of x**power on [start, end]."""
    return (end ** (power + 1) - start ** (power + 1)) / ((power + 1) * (end - start))


def test_quadrature_blocks_align_with_cells():
    quadrature_n = 3
    n_cells = 4
    strategy = CellIntegralQuadratureSupervision(
        quadrature_n=quadrature_n,
        embd_size=n_cells,
        patch_size=16,
        block_chunk_size=2,
    )

    bboxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    q_xy_blocks = strategy.get_transformed_q_xy_blocks(bboxes)[0]
    hidden_ij = strategy.get_hidden_ij_blocks()[0].squeeze(1).squeeze(1)

    assert q_xy_blocks.shape == (n_cells * n_cells, quadrature_n, quadrature_n, 2)
    assert hidden_ij.shape == (n_cells * n_cells, 2)

    cell_size = 1.0 / n_cells
    i_idx = hidden_ij[:, 0].to(torch.float32)
    j_idx = hidden_ij[:, 1].to(torch.float32)

    x_min = j_idx * cell_size
    x_max = (j_idx + 1.0) * cell_size
    y_min = i_idx * cell_size
    y_max = (i_idx + 1.0) * cell_size

    # Every quadrature point in each block must lie in its corresponding hidden cell.
    assert torch.all(q_xy_blocks[..., 0] >= x_min[:, None, None] - 1e-7)
    assert torch.all(q_xy_blocks[..., 0] <= x_max[:, None, None] + 1e-7)
    assert torch.all(q_xy_blocks[..., 1] >= y_min[:, None, None] - 1e-7)
    assert torch.all(q_xy_blocks[..., 1] <= y_max[:, None, None] + 1e-7)


def test_quadrature_exact_for_selected_polynomials():
    # quadrature_n=2 is exact for degree <= 3 per axis with Gauss-Legendre tensor product.
    quadrature_n = 2
    n_cells = 5
    strategy = CellIntegralQuadratureSupervision(
        quadrature_n=quadrature_n,
        embd_size=n_cells,
        patch_size=16,
        block_chunk_size=3,
    )

    bboxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    q_xy_blocks = strategy.get_transformed_q_xy_blocks(bboxes)[0]
    hidden_ij = strategy.get_hidden_ij_blocks()[0].squeeze(1).squeeze(1)

    i_idx = hidden_ij[:, 0].to(torch.float32)
    j_idx = hidden_ij[:, 1].to(torch.float32)
    cell_size = 1.0 / n_cells

    x0 = j_idx * cell_size
    x1 = (j_idx + 1.0) * cell_size
    y0 = i_idx * cell_size
    y1 = (i_idx + 1.0) * cell_size

    # (px, py) with px, py <= 3 should integrate exactly with quadrature_n=2.
    polynomial_powers = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (0, 2),
        (1, 1),
        (3, 0),
        (0, 3),
        (2, 3),
    ]

    for px, py in polynomial_powers:
        x = q_xy_blocks[..., 0]
        y = q_xy_blocks[..., 1]
        q_vals = (x**px) * (y**py)

        q_ft_chunk = q_vals.unsqueeze(0).unsqueeze(-1)
        integrated = strategy.query_features_to_hidden(q_ft_chunk)[0]
        integrated = integrated.squeeze(0).squeeze(-1).squeeze(-1).squeeze(-1)

        expected = _cell_average_monomial(px, x0, x1) * _cell_average_monomial(
            py, y0, y1
        )

        assert torch.allclose(
            integrated, expected, atol=1e-6, rtol=1e-6
        ), f"Quadrature mismatch for polynomial x^{px} y^{py}"
