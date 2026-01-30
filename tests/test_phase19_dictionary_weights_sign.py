import numpy as np

from ratchet_gpu.motifs import dictionary_score, dictionary_weights


def test_dictionary_weights_sign() -> None:
    p_out = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    p_in = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    weights = dictionary_weights(p_out, p_in, eps=1e-6)
    assert weights[0] > 0
    assert weights[1] < 0
    assert abs(weights[2]) < 1e-6

    score_out = dictionary_score(p_out, weights)
    score_in = dictionary_score(p_in, weights)
    assert score_out > 0
    assert score_in < 0
