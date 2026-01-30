from ratchet_gpu.diagnostics import ep_windowed


def test_ep_windowed_per_proposal():
    prev = {"ep_total_exact": 0.0, "ep_by_kernel": {"k_local": 0.0}}
    curr = {"ep_total_exact": 10.0, "ep_by_kernel": {"k_local": 10.0}, "window_proposals": 100}
    rates = ep_windowed(prev, curr, 0, 0)
    assert abs(rates["ep_rate_exact_window"] - 0.1) < 1e-9
    assert abs(rates["ep_rate_by_kernel_window"]["k_local"] - 0.1) < 1e-9
