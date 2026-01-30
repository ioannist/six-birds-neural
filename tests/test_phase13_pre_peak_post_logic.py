from ratchet_gpu.setpoints import pre_peak_post


def test_pre_peak_post_logic() -> None:
    values = [0.1, 0.2, 0.6, 0.4, 0.3]
    pre, peak, post = pre_peak_post(values, injury_window=3, injury_duration=1, last_m=2)
    assert abs(pre - 0.15) < 1e-9
    assert abs(peak - 0.6) < 1e-9
    assert abs(post - 0.35) < 1e-9
