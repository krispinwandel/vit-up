import torch

from vit_up.supervision.strategies.point_wise import PointwiseTokenSupervision


def test_pointwise_blocks_and_hidden_indices_shapes():
    embd_size = 3
    strategy = PointwiseTokenSupervision(
        embd_size=embd_size,
        patch_size=16,
        block_chunk_size=8,
    )

    q_xy_blocks = strategy._blocks["q_xy_blocks"]
    hidden_ij_blocks = strategy.get_hidden_ij_blocks()[0]

    # block_size_query=1, so there is one query token per block.
    assert q_xy_blocks.shape == (embd_size * embd_size, 1, 1, 2)
    assert hidden_ij_blocks.shape == (embd_size * embd_size, 1, 1, 2)

    # Hidden indices should enumerate every (i, j) exactly once.
    ij = hidden_ij_blocks[:, 0, 0, :]
    expected = torch.tensor(
        [[i, j] for i in range(embd_size) for j in range(embd_size)],
        dtype=ij.dtype,
        device=ij.device,
    )
    assert torch.equal(ij, expected)


def test_pointwise_transformed_grid_and_identity_mapping():
    strategy = PointwiseTokenSupervision(
        embd_size=2,
        patch_size=16,
        block_chunk_size=8,
    )

    bbox = torch.tensor([[0.2, 0.3, 0.8, 0.9]], dtype=torch.float32)
    q_xy_blocks = strategy.get_transformed_q_xy_blocks(bbox)

    # 2x2 centers in unit square are 0.25/0.75, then affine transformed by bbox.
    expected_x = torch.tensor([0.35, 0.65, 0.35, 0.65], dtype=torch.float32)
    expected_y = torch.tensor([0.45, 0.45, 0.75, 0.75], dtype=torch.float32)

    got = q_xy_blocks[0, :, 0, 0, :]
    assert torch.allclose(got[:, 0], expected_x, atol=1e-7, rtol=0.0)
    assert torch.allclose(got[:, 1], expected_y, atol=1e-7, rtol=0.0)

    q_ft_chunk = torch.randn(2, 4, 1, 1, 5)
    mapped = strategy.query_features_to_hidden(q_ft_chunk)

    assert set(mapped.keys()) == {0}
    assert mapped[0].shape == q_ft_chunk.shape
    assert torch.equal(mapped[0], q_ft_chunk)
