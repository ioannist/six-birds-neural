import numpy as np

from ratchet_gpu.motifs import motif_dictionary_eval


def test_dictionary_eval_detects_signal() -> None:
    num_motifs = 3
    total_windows = 10
    in_hist = np.zeros((total_windows, num_motifs), dtype=np.float64)
    out_hist = np.zeros_like(in_hist)
    in_hist[:, 0] = 1.0
    out_hist[:, 1] = 1.0

    rng = np.random.default_rng(0)
    res = motif_dictionary_eval(in_hist, out_hist, shuffle_n=200, rng=rng)
    assert res["dict_delta"] > 0.5
    assert res["dict_p"] <= 0.1
