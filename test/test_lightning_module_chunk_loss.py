import types

import torch

from vit_up.supervision.strategies.quadrature import CellIntegralQuadratureSupervision
from vit_up.training.lightning_module import ViTUpPL


class _FakeVitUp:
    def __init__(self):
        self.layer_indices = [0]

    def compute_cache_data(self, pixel_values, backbone, hidden_layer_img_size):
        del pixel_values, backbone, hidden_layer_img_size
        return None

    def __call__(self, pixel_values, q_xy_normalized, cache_data):
        del pixel_values, cache_data
        x = q_xy_normalized[..., 0]
        y = q_xy_normalized[..., 1]
        feats = torch.stack([x, y, x + y], dim=-1)
        return [feats]


class _FakeBackboneClass:
    @staticmethod
    def _compute_gt_features(
        backbone,
        pixel_values,
        layer_indices,
        img_size,
        window_size,
        flatten_hw_to_seq=True,
    ):
        del (
            backbone,
            layer_indices,
            window_size,
            flatten_hw_to_seq,
        )
        bsz = pixel_values.shape[0]
        embd_size = img_size

        i_idx, j_idx = torch.meshgrid(
            torch.arange(
                embd_size, dtype=pixel_values.dtype, device=pixel_values.device
            ),
            torch.arange(
                embd_size, dtype=pixel_values.dtype, device=pixel_values.device
            ),
            indexing="ij",
        )
        denom = float(max(embd_size - 1, 1))
        x = j_idx / denom
        y = i_idx / denom
        gt = torch.stack([x, y, x + y], dim=-1).reshape(1, embd_size * embd_size, 3)
        gt = gt.expand(bsz, -1, -1).contiguous()
        return [gt]


class _MSELossDict:
    def __call__(self, gt_layers_by_scale, pred_layers_by_scale):
        total = None
        for scale in gt_layers_by_scale:
            gt_layers = gt_layers_by_scale[scale]
            pred_layers = pred_layers_by_scale[scale]
            for gt, pred in zip(gt_layers, pred_layers):
                val = torch.mean((pred - gt) ** 2)
                total = val if total is None else total + val
        assert total is not None
        return {"loss": total, "mse": total}


def _make_module(block_chunk_size: int) -> ViTUpPL:
    m = ViTUpPL.__new__(ViTUpPL)
    m.supervision_strategy = CellIntegralQuadratureSupervision(
        quadrature_n=2,
        embd_size=2,
        patch_size=1,
        block_chunk_size=block_chunk_size,
    )
    m.backbone_class = _FakeBackboneClass
    m.backbone = object()
    m.backbone_patch_size = 1
    m.vit_up = _FakeVitUp()
    m.vit_up_layer_indices = m.vit_up.layer_indices
    m.window_size = 0
    m.lr_img_size = 2
    m.loss_fn = _MSELossDict()
    return m


def _make_batch(batch_size: int = 2):
    pixel_values = torch.zeros(batch_size, 3, 8, 8, dtype=torch.float32)

    # 2x2 query centers with size 0.5 cover the full [0,1] x [0,1] bbox.
    q_xy = torch.tensor(
        [
            [[0.25, 0.25], [0.75, 0.25]],
            [[0.25, 0.75], [0.75, 0.75]],
        ],
        dtype=torch.float32,
    )
    q_xy = q_xy.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    q_size = torch.full((batch_size, 2, 2), 0.5, dtype=torch.float32)

    return {
        "pixel_values": pixel_values,
        "pixel_values_container": pixel_values,
        "query_xy": q_xy,
        "query_size": q_size,
    }


def test_compute_loss_chunked_matches_no_chunking():
    batch = _make_batch()

    module_no_chunk = _make_module(block_chunk_size=1_000_000)
    module_chunked = _make_module(block_chunk_size=1)

    loss_no_chunk = ViTUpPL._compute_loss(module_no_chunk, batch)
    loss_chunked = ViTUpPL._compute_loss(module_chunked, batch)

    assert torch.allclose(
        loss_chunked["loss"], loss_no_chunk["loss"], atol=1e-7, rtol=0.0
    )
    assert torch.allclose(
        loss_chunked["mse"], loss_no_chunk["mse"], atol=1e-7, rtol=0.0
    )
