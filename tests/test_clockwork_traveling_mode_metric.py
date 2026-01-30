import numpy as np

from ratchet_gpu.clockwork import traveling_mode_score


def test_traveling_mode_score_detects_wave() -> None:
    T, H, W = 40, 16, 16
    omega = 0.3
    x = np.arange(W)[None, None, :]
    frames = np.empty((T, H, W), dtype=np.float64)
    for t in range(T):
        frames[t] = np.cos(2 * np.pi * x / W + omega * t)

    result = traveling_mode_score(frames, omega_min=0.1)
    assert result.travel_score > 0.05
    assert result.r2 > 0.5
    assert abs(result.omega) >= 0.1
    assert result.best_k[0] == 0
    assert abs(result.best_k[1]) == 1
