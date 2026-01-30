import numpy as np

from ratchet_gpu.semantics import alignment_shift_null, hazard_center, pref_axis_map, ring_masks_from_rect


def test_shift_null_p_value_sane():
    rng = np.random.default_rng(0)
    shape = (8, 8)
    rect = "2:6,2:6"
    _region, ring, _outside = ring_masks_from_rect(rect, shape, width=1)
    center = hazard_center(rect, shape)
    pref = pref_axis_map(shape, center)

    axis_bias_by_window = [rng.normal(size=shape) for _ in range(6)]
    pre_idx = [0, 1, 2]
    haz_idx = [3, 4, 5]

    p_val, mean_null, std_null = alignment_shift_null(
        axis_bias_by_window,
        pref,
        ring,
        pre_idx,
        haz_idx,
        shuffle_n=50,
        rng=rng,
    )
    assert 0.0 <= p_val <= 1.0
    assert np.isfinite(mean_null)
    assert np.isfinite(std_null)
