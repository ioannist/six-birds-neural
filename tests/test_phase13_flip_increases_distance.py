import torch

from ratchet_gpu.setpoints import masked_distance


def test_flip_increases_region_distance() -> None:
    grid = torch.ones((4, 4), dtype=torch.float64)
    target = grid.clone()
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[:2, :2] = True
    flipped = grid.clone()
    flipped[mask] = -flipped[mask]
    dist_before = masked_distance(grid, target, mask)
    dist_after = masked_distance(flipped, target, mask)
    assert dist_before == 0.0
    assert dist_after > 0.0
