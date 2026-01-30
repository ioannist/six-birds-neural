import numpy as np

from ratchet_gpu.motifs import propagation_score


def test_motif_propagation_detects_shift():
    frames = []
    for t in range(4):
        arr = np.zeros((6, 6), dtype=int)
        arr[:, t % 6] = 1
        frames.append(arr)
    score, motif, shift = propagation_score(frames, motifs=[1], shift_max=1)
    assert score > 0.2
    assert motif == 1
    assert shift != (0, 0)
