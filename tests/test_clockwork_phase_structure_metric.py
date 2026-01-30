import numpy as np

from ratchet_gpu.clockwork import phase_structure_score


def test_phase_structure_detects_spatial_phase() -> None:
    T, H, W = 36, 16, 16
    omega = 0.25
    y = np.arange(H)[None, :, None]
    frames = np.empty((T, H, W), dtype=np.float64)
    for t in range(T):
        frames[t] = np.cos(omega * t + 2 * np.pi * y / H)

    result = phase_structure_score(frames)
    assert result.phase_score > 0.01
    assert result.phase_grad > 0.01


def test_phase_structure_rejects_global_oscillation() -> None:
    T, H, W = 36, 16, 16
    omega = 0.25
    frames = np.empty((T, H, W), dtype=np.float64)
    for t in range(T):
        frames[t] = np.cos(omega * t)

    result = phase_structure_score(frames)
    assert result.phase_score < 0.005
