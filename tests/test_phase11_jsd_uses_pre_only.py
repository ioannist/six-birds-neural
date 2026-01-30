import numpy as np

from ratchet_gpu.motifs import jsd, split_condition_counts


def _normalize(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def test_jsd_uses_pre_only() -> None:
    windows = [1, 2, 3]
    pre = np.array([1.0, 0.0])
    hazard = np.array([0.0, 1.0])
    post = np.array([0.0, 1.0])
    hists = [pre, hazard, post]

    pre_counts, haz_counts, post_counts = split_condition_counts(
        windows, hazard_start=2, hazard_duration=1, histograms=hists
    )
    pre_p = _normalize(pre_counts)
    haz_p = _normalize(haz_counts)
    post_p = _normalize(post_counts)

    jsd_pre_hazard = jsd(pre_p, haz_p)
    mixed = _normalize(pre_p + post_p)
    jsd_mixed_hazard = jsd(mixed, haz_p)

    assert jsd_pre_hazard > jsd_mixed_hazard
    assert jsd_pre_hazard > 0.2
