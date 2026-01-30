import numpy as np

from ratchet_gpu.semantics import shift_null_corr


def test_alignment_shift_null_detects_signal() -> None:
    n = 50
    rng_labels = np.random.default_rng(0)
    labels = rng_labels.choice([-1.0, 1.0], size=n).astype(np.float64)
    rng_scores = np.random.default_rng(1)
    scores = labels + 0.05 * rng_scores.normal(size=n)
    rng_null = np.random.default_rng(2)

    alignment, p_val, _mean_null, _std_null = shift_null_corr(
        scores,
        labels,
        shuffle_n=200,
        rng=rng_null,
    )

    assert alignment > 0.7
    assert p_val < 0.05
