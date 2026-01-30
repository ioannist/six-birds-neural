import numpy as np

from ratchet_gpu.semantics import shift_ids_map


def test_semantic_shuffle_shift_preserves_counts():
    ids = np.arange(24, dtype=np.int64).reshape(4, 6) % 5
    counts = np.bincount(ids.ravel(), minlength=5)
    shifted = shift_ids_map(ids, dx=2, dy=-1)
    shifted_counts = np.bincount(shifted.ravel(), minlength=5)
    assert np.array_equal(counts, shifted_counts)
