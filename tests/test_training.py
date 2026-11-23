import torch

from vit_up.training.losses import CosineDistance, FeatureLoss, KLDistance, L2Distance
from vit_up.training.objectives import GridDistillation, OffsetDistillation, OffsetHead, box_coordinates, to_blocks


class Model(torch.nn.Module):
    def __init__(self, vit_up, backbone):
        super().__init__()
        self.vit_up, self.backbone = vit_up, backbone

    def manual_backward(self, loss, retain_graph=False):
        loss.backward(retain_graph=retain_graph)


def batch(size=64):
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.25, 0.25, 0.75, 0.75]])
    return {"teacher_image": torch.randn(2, 3, size, size), "student_image": torch.randn(2, 3, size, size),
            "teacher_box": boxes}


def loss(layer_weights, scale_weights, distances=None):
    return FeatureLoss(distances or [L2Distance(), CosineDistance()], layer_weights, scale_weights)


def test_to_blocks_row_major():
    x = torch.arange(16).reshape(1, 4, 4, 1)
    blocks = to_blocks(x, 2)
    assert blocks.shape == (1, 4, 2, 2, 1)
    assert blocks[0, 1, :, :, 0].tolist() == [[2, 3], [6, 7]]


def test_box_coordinates():
    points = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]])
    mapped = box_coordinates(points, torch.tensor([[0.2, 0.4, 0.6, 0.8]]))
    torch.testing.assert_close(mapped, torch.tensor([[[0.2, 0.4], [0.6, 0.8], [0.4, 0.6]]]))


def test_feature_loss_gradients_match_autograd():
    target, pred = torch.randn(2, 9, 8), torch.randn(2, 9, 8, requires_grad=True)
    distances = [L2Distance(), CosineDistance(), KLDistance()]
    loss(layer_weights=[1.0], scale_weights=[1.0], distances=distances)({0: [target]}, {0: [pred]})["loss"].backward()
    reference = pred.detach().clone().requires_grad_()
    sum(d(target, reference) for d in distances).backward()
    torch.testing.assert_close(pred.grad, reference.grad)


def test_grid_distillation_backward(vit_up, backbone):
    model = Model(vit_up, backbone)
    objective = GridDistillation(loss([1.0] * 7, [1.0] * 3), patch_size=16, grid_size=2, blocks_per_chunk=3)
    data = batch()
    metrics = objective(model, data, objective.prepare(model, data), image_size=64, backward_weight=1.0)
    assert torch.isfinite(metrics["loss"])
    assert all(p.grad is not None for p in vit_up.decoder_mlp.parameters())


def test_offset_distillation_backward(vit_up, backbone):
    model = Model(vit_up, backbone)
    objective = OffsetDistillation(loss([0.0] * 6 + [1.0], [1.0]), OffsetHead(32, 16), num_queries=20)
    objective.setup(model)
    data = batch()
    metrics = objective(model, data, objective.prepare(model, data), image_size=64, backward_weight=1.0)
    assert torch.isfinite(metrics["loss"])
    assert objective.offset_head.mlp[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in objective.teacher.parameters())


def test_validation_queries_are_fixed(vit_up, backbone):
    model = Model(vit_up, backbone).eval()
    objective = OffsetDistillation(loss([1.0] * 7, [1.0]), OffsetHead(32), num_queries=5)
    data = batch()
    first, second = objective.prepare(model, data)[0], objective.prepare(model, data)[0]
    assert torch.equal(first, second)
