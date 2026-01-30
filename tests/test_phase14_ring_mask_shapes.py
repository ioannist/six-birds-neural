import numpy as np

from ratchet_gpu.semantics import ring_masks_from_rect


def test_phase14_ring_mask_shapes():
    shape = (10, 10)
    region, ring, outside = ring_masks_from_rect("2:6,3:7", shape, width=1)

    assert region.shape == shape
    assert ring.shape == shape
    assert outside.shape == shape
    assert np.logical_and(region, ring).sum() == 0
    assert np.logical_and(region, outside).sum() == 0
    assert np.logical_and(ring, outside).sum() == 0

    assert region.sum() == 16  # 4x4 region
    assert ring.sum() == 20
    assert outside.sum() == 64
