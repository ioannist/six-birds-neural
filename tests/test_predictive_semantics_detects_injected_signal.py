import numpy as np

from ratchet_gpu.semantics import predictive_semantic_pvalue


def test_predictive_semantics_detects_injected_signal():
    rng = np.random.default_rng(123)
    steps = 60
    x_tm = rng.normal(scale=0.05, size=(steps, 3))
    x_tm[:, 0] = np.linspace(0.0, 1.0, steps)
    y_t = -x_tm[:, 0] + rng.normal(scale=0.02, size=steps)

    best_corr, pval, _mean_null, _std_null, best_idx = predictive_semantic_pvalue(
        x_tm,
        y_t,
        shift_n=200,
        rng=np.random.default_rng(7),
        metric="corr",
    )

    assert best_idx == 0
    assert best_corr < -0.5
    assert pval <= 0.05
