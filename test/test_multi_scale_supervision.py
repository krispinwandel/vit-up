import torch

from vit_up.supervision.strategies.multi_scale import (
    MultiScaleSupervision,
)


def test_multi_scale_blocks_and_shapes_are_consistent():
    strategy = MultiScaleSupervision(
        min_embd_size=2,
        n_iters=3,
        patch_size=16,
        scale=2,
        block_chunk_size=8,
    )

    block_size_q = strategy.block_size_query()
    assert block_size_q == 4

    q_xy_blocks = strategy._blocks["q_xy_blocks"]
    hidden_ij_blocks = strategy.get_hidden_ij_blocks()

    # n_q = min_embd_size * block_size_q = 8 -> n_blocks = (8 / 4)^2 = 4
    assert q_xy_blocks.shape == (4, 4, 4, 2)
    assert set(hidden_ij_blocks.keys()) == {0, 1, 2}
    assert hidden_ij_blocks[0].shape == (4, 1, 1, 2)
    assert hidden_ij_blocks[1].shape == (4, 2, 2, 2)
    assert hidden_ij_blocks[2].shape == (4, 4, 4, 2)


def test_multi_scale_query_features_chunk_invariance():
    strategy = MultiScaleSupervision(
        min_embd_size=2,
        n_iters=3,
        patch_size=16,
        scale=2,
        block_chunk_size=8,
    )

    b = 2
    n_blocks = strategy._blocks["q_xy_blocks"].shape[0]
    block_size_q = strategy.block_size_query()
    c = 3

    # Deterministic feature field with nontrivial spatial/channel variation.
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, block_size_q),
        torch.linspace(0.0, 1.0, block_size_q),
        indexing="ij",
    )
    base = torch.stack((x, y, x + 2.0 * y), dim=-1)  # (q, q, c)
    q_ft_chunk = base[None, None, :, :, :].expand(b, n_blocks, -1, -1, -1).clone()

    full = strategy.query_features_to_hidden(q_ft_chunk)

    split = n_blocks // 2
    out_a = strategy.query_features_to_hidden(q_ft_chunk[:, :split])
    out_b = strategy.query_features_to_hidden(q_ft_chunk[:, split:])
    merged = {k: torch.cat((out_a[k], out_b[k]), dim=1) for k in full}

    for i in range(strategy.n_iters):
        s = strategy.scale**i
        assert full[i].shape == (b, n_blocks, s, s, c)
        assert torch.allclose(merged[i], full[i], atol=0.0, rtol=0.0)


def test_multi_scale_query_features_use_mean_pooling():
    strategy = MultiScaleSupervision(
        min_embd_size=1,
        n_iters=3,
        patch_size=16,
        scale=2,
        block_chunk_size=4,
        mode="mean",
    )

    # block_size_q = 4, so scale 1 (2x2) should be a 2x2 mean pool,
    # and scale 0 (1x1) should be the global mean.
    q = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4, 1)
    out = strategy.query_features_to_hidden(q)

    expected_s1 = torch.tensor([[[[[3.5], [5.5]], [[11.5], [13.5]]]]])
    expected_s0 = torch.tensor([[[[[8.5]]]]])

    assert torch.allclose(out[1], expected_s1, atol=0.0, rtol=0.0)
    assert torch.allclose(out[0], expected_s0, atol=0.0, rtol=0.0)
