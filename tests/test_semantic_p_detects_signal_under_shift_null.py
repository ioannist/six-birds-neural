import numpy as np

from ratchet_gpu.semantics import semantic_effect


def test_semantic_p_detects_signal_under_shift_null():
    H = 6
    W = 6
    ring_mask = np.ones((H, W), dtype=bool)

    ids = np.zeros((H, W), dtype=np.int64)
    ids[0, 0] = 1
    motif_ids_by_window = [ids, ids]

    mismatch0 = np.zeros((H, W), dtype=np.float64)
    mismatch1 = np.ones((H, W), dtype=np.float64)
    mismatch1[0, 0] = -1.0
    mismatch_by_window = [mismatch0, mismatch1]

    result = semantic_effect(
        motif_ids_by_window=motif_ids_by_window,
        mismatch_by_window=mismatch_by_window,
        windows=[1, 2],
        ring_mask=ring_mask,
        num_motifs=2,
        hazard_start=1,
        hazard_duration=1,
        support_min=0.01,
        shuffle_n=200,
        shuffle_mode="shift",
        candidate_top_k=2,
        rng=np.random.default_rng(0),
    )

    assert result["semantic_best"] < -0.5
    assert result["semantic_p"] <= 0.1
