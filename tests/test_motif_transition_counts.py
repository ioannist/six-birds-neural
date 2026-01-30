import numpy as np

from ratchet_gpu.motifs import transition_counts


def test_transition_counts_basic() -> None:
    prev = np.array([[0, 1], [1, 0]])
    next_ids = np.array([[1, 1], [0, 0]])
    counts = transition_counts(prev, next_ids, n_motifs=2)

    assert counts.shape == (2, 2)
    assert counts[0, 0] == 1
    assert counts[0, 1] == 1
    assert counts[1, 0] == 1
    assert counts[1, 1] == 1
