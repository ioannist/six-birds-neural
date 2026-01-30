import numpy as np

from ratchet_gpu.semantics import valid_nontrivial_circular_shifts


def test_phase20_shift_null_skips_identical() -> None:
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    shifts = valid_nontrivial_circular_shifts(labels)
    assert 2 not in shifts
    assert 4 not in shifts
    assert 6 not in shifts
    assert 1 in shifts
    assert 3 in shifts


def test_phase20_shift_null_constant_sequence() -> None:
    labels = np.array([1, 1, 1, 1], dtype=np.int64)
    shifts = valid_nontrivial_circular_shifts(labels)
    assert shifts == []
