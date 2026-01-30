import numpy as np

from ratchet_gpu.semantics import semantic_effect


def test_phase14_semantics_effect_detects_signal():
    ring_mask = np.ones((2, 2), dtype=bool)
    motif_ids_by_window = [
        np.ones((2, 2), dtype=np.int64),
        np.ones((2, 2), dtype=np.int64),
    ]
    mismatch_by_window = [
        np.ones((2, 2), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
    ]
    windows = [1, 2]
    result = semantic_effect(
        motif_ids_by_window=motif_ids_by_window,
        mismatch_by_window=mismatch_by_window,
        windows=windows,
        ring_mask=ring_mask,
        num_motifs=2,
        hazard_start=1,
        hazard_duration=1,
        support_min=0.1,
        shuffle_n=0,
        rng=np.random.default_rng(0),
    )

    assert result["semantic_best"] < -0.5
    assert result["semantic_support"] > 0
