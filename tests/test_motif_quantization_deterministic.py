import numpy as np

from ratchet_gpu.motifs import build_bins, motif_ids, motif_hist


def test_motif_quantization_deterministic():
    data = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=float)
    baseline = {
        "k_axis_bias": [data],
        "k_entropy": [data * 0.5],
    }
    bins = build_bins(baseline, {"k_axis_bias": 2, "k_entropy": 2})
    feats = {"k_axis_bias": data, "k_entropy": data * 0.5}
    ids = motif_ids(feats, bins)
    hist = motif_hist(ids, bins.bins["k_axis_bias"] * bins.bins["k_entropy"])
    assert ids.shape == data.shape
    assert hist.sum() == 1.0
