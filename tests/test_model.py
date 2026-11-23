import torch

from vit_up.model.vit_up import closest_token_index, relative_position
from vit_up.utils import regular_grid


def test_output_shapes(vit_up, backbone):
    images = torch.randn(2, 3, 64, 64)
    cache = vit_up.encode(images, backbone)
    outputs = vit_up(torch.rand(2, 10, 2), cache)
    assert len(outputs) == 7
    assert all(out.shape == (2, 10, 32) for out in outputs)


def test_queries_are_independent(vit_up, backbone):
    cache = vit_up.encode(torch.randn(1, 3, 64, 64), backbone)
    queries = torch.rand(1, 50, 2)
    together = vit_up(queries, cache)[-1]
    separately = torch.cat([vit_up(q, cache)[-1] for q in queries.split(7, dim=1)], dim=1)
    torch.testing.assert_close(together, separately)


def test_coordinates_receive_gradients(vit_up, backbone):
    with torch.no_grad():
        cache = vit_up.encode(torch.randn(1, 3, 64, 64), backbone)
    queries = torch.rand(1, 20, 2, requires_grad=True)
    vit_up(queries, cache)[-1].sum().backward()
    assert queries.grad is not None and queries.grad.abs().sum() > 0


def test_relative_position_of_token_centers():
    centers = regular_grid(4).reshape(1, -1, 2)
    index = closest_token_index(centers, 4, 4)
    assert torch.equal(index[0], torch.arange(16))
    torch.testing.assert_close(relative_position(centers, index, 4, 4), torch.zeros(1, 16, 2))


def test_regular_grid_is_xy_row_major():
    grid = regular_grid(2)
    torch.testing.assert_close(grid[0, 1], torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(grid[1, 0], torch.tensor([0.25, 0.75]))
