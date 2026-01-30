import numpy as np


def test_predictive_semantics_shift_null_preserves_marginals():
    rng = np.random.default_rng(0)
    x_tm = rng.normal(size=(32, 4))
    shifted = np.roll(x_tm, shift=7, axis=0)
    assert np.allclose(x_tm.mean(axis=0), shifted.mean(axis=0))
