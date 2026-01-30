import numpy as np

from ratchet_gpu.semantics import shift_null_p_value_for_accuracy


def test_phase20_decoder_accuracy_detects_signal() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=30)
    preds = labels.copy()
    _obs, p_val, _mean_null, _std_null, _n = shift_null_p_value_for_accuracy(
        preds, labels, shuffle_n=200, rng=np.random.default_rng(1)
    )
    assert p_val < 0.05


def test_phase20_decoder_accuracy_null() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, size=30)
    preds = rng.integers(0, 2, size=30)
    _obs, p_val, _mean_null, _std_null, _n = shift_null_p_value_for_accuracy(
        preds, labels, shuffle_n=200, rng=np.random.default_rng(3)
    )
    assert p_val > 0.1
