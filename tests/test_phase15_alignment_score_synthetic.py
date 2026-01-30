import numpy as np

from ratchet_gpu.semantics import alignment_delta, alignment_score, hazard_center, pref_axis_map, ring_masks_from_rect


def test_alignment_score_pref_axis_positive():
    shape = (8, 8)
    rect = "2:6,2:6"
    _region, ring, _outside = ring_masks_from_rect(rect, shape, width=1)
    center = hazard_center(rect, shape)
    pref = pref_axis_map(shape, center)
    axis_bias = pref.copy()
    score = alignment_score(axis_bias, pref, ring)
    assert score > 0.9


def test_alignment_delta_positive():
    scores_pre = [0.1, 0.1, 0.1]
    scores_hazard = [0.3, 0.25, 0.2]
    delta = alignment_delta(scores_pre, scores_hazard)
    assert delta > 0.1
