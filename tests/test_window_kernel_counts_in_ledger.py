from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


def test_window_kernel_counts_in_ledger():
    params = Params(
        shape=(2, 2),
        layers=2,
        kernel_weights={"spin_flip_color0": 1.0, "spin_flip_color1": 1.0},
        report_every=1,
    )
    seen = {}

    def cb(state, step, ledger, accepted):
        seen.update(ledger)

    run_sim(params, seed=0, steps=2, report_every=1, device="cpu", report_callback=cb)
    assert "window_proposals_by_kernel" in seen
    assert "window_accepted_by_kernel" in seen
    assert "window_accept_frac_by_kernel" in seen
    total = int(seen["window_proposals"])
    assert total == sum(int(v) for v in seen["window_proposals_by_kernel"].values())
