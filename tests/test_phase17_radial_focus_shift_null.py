import numpy as np

from ratchet_gpu.semantics import radial_focus_shift_null


def test_radial_focus_shift_null_range() -> None:
    rng = np.random.default_rng(0)
    ring_mask = np.zeros((4, 4), dtype=bool)
    ring_mask[1:3, 1:3] = True
    radial_by_window = [rng.normal(size=(4, 4)) for _ in range(5)]
    p_val, mean_null, std_null = radial_focus_shift_null(
        radial_by_window, ring_mask, [0, 1], [2, 3], 5, rng
    )
    assert 0.0 <= p_val <= 1.0
    assert isinstance(mean_null, float)
    assert isinstance(std_null, float)
    assert std_null >= 0.0
