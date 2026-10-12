"""Small CPU smoke tests for public SCD-Det components."""

import torch

from scd_det import CEBA, FREG, UMA, PaperBSCLoss


def test_feature_shapes() -> None:
    with torch.no_grad():
        assert CEBA(64, 64, context_kernel=9)(torch.randn(2, 64, 40, 40)).shape == (2, 64, 40, 40)
        assert FREG(64, 64, k=12, s=10, num_heads=8, ratio=0.25)(
            torch.randn(2, 64, 20, 20)
        ).shape == (2, 64, 20, 20)
        output = UMA([32, 64, 128], out_channels=32)(
            [torch.randn(2, 32, 40, 40), torch.randn(2, 64, 20, 20), torch.randn(2, 128, 10, 10)]
        )
        assert len(output) == 1 and output[0].shape == (2, 32, 40, 40)


def test_bsc_is_finite_and_differentiable() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 3.0, 2.0], [1.0, 1.0, 2.0, 2.0]]],
        requires_grad=True,
    )
    foreground = torch.tensor([[True, True, False]])
    loss = PaperBSCLoss(lambda_cons=0.01)(boxes, foreground)
    assert torch.isfinite(loss)
    loss.backward()
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all()
