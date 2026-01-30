from ratchet_gpu.ep import StrobeTracker


def test_strobe_current_metrics_simple_edge():
    tracker = StrobeTracker()
    a = (0,)
    b = (1,)
    tracker.transitions = {a: {b: 10}, b: {a: 2}}
    tracker.total = 12

    expected = abs(10 - 2) / 12
    assert abs(tracker.symgap() - expected) < 1e-12
    assert abs(tracker.current_l2() - expected) < 1e-12

    top = tracker.top_currents(1)
    assert len(top) == 1
    assert abs(top[0]["j"] - expected) < 1e-12
    assert top[0]["count_ab"] == 10
    assert top[0]["count_ba"] == 2
