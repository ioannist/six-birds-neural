import numpy as np

from ratchet_gpu.motifs import transition_jsd


def test_transition_jsd_symmetric() -> None:
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    b = np.array([[4, 3], [2, 1]], dtype=np.int64)
    assert transition_jsd(a, b) == transition_jsd(b, a)


def test_transition_jsd_zero_when_equal() -> None:
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert transition_jsd(a, a) == 0.0


def test_transition_jsd_positive_when_diff() -> None:
    a = np.array([[10, 0], [0, 0]], dtype=np.int64)
    b = np.array([[0, 0], [0, 10]], dtype=np.int64)
    assert transition_jsd(a, b) > 0.0
