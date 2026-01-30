import numpy as np

from ratchet_gpu.motifs import jsd


def test_jsd_properties():
    p = np.array([0.5, 0.5], dtype=float)
    q = np.array([0.9, 0.1], dtype=float)
    assert jsd(p, p) == 0.0
    assert abs(jsd(p, q) - jsd(q, p)) < 1e-12
    assert jsd(p, q) >= 0.0
