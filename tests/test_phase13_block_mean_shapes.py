import torch

from ratchet_gpu.setpoints import block_mean


def test_block_mean_shapes() -> None:
    grid = torch.arange(24 * 24, dtype=torch.float32).reshape(24, 24)
    coarse = block_mean(grid, block=4)
    assert coarse.shape == (6, 6)
    assert torch.isclose(coarse.mean(), grid.mean())

    grid3 = torch.stack([grid, grid + 1.0], dim=0)
    coarse3 = block_mean(grid3, block=4)
    assert coarse3.shape == (2, 6, 6)
