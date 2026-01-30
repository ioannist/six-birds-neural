from ratchet_gpu.diagnostics import compute_snapshot, ep_totals
from ratchet_gpu.state import State
from ratchet_gpu.params import Params
import torch


def test_snapshot_uses_window_proposals():
    # first snapshot seeds prev_totals
    params = Params(shape=(2, 2), layers=2, B_k=0, radius_k=0, device="cpu")
    dummy = State.initialize(params, seed=0)
    prev_totals = ep_totals(
        {"ep_total_exact": 0.0, "ep_by_kernel": {"k_local": 0.0}, "window_proposals": 100}
    )
    _, diag_state = compute_snapshot(
        state=dummy,
        step=0,
        ep_ledger={"ep_total_exact": 0.0, "ep_by_kernel": {"k_local": 0.0}, "window_proposals": 100},
        accepted_frac=None,
        prev_diag_state=None,
    )
    snapshot, _ = compute_snapshot(
        state=dummy,
        step=100,
        ep_ledger={
            "ep_total_exact": 10.0,
            "ep_by_kernel": {"k_local": 10.0},
            "window_proposals": 100,
        },
        accepted_frac=None,
        prev_diag_state=diag_state,
    )
    assert abs(snapshot["ep_rate_exact_window"] - 0.1) < 1e-9
    assert abs(snapshot["ep_rate_by_kernel_window"]["k_local"] - 0.1) < 1e-9
