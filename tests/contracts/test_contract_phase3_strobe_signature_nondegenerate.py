import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim
from ratchet_gpu.diagnostics import compute_snapshot


@pytest.mark.contract
def test_strobe_signature_non_degenerate():
    shape = (12, 12)
    N = shape[0] * shape[1]
    params = Params(
        shape=shape,
        layers=2,
        p3_on=False,
        p6_on=False,
        beta=0.5,
        J=1.0,
        kappa_T=1.0,
        eta=0.5,
        eta_drive=0.0,
        l_s=0,
        l_w=3,
        l_k=2,
        B_w=10,
        B_k=2,
        radius_w=1,
        radius_k=2,
        kernel_weights={
            "spin_flip_color0": 1.0,
            "spin_flip_color1": 1.0,
            "w_local": 1.0,
            "w_neighbor": 0.0,
            "k_local": 1.0,
            "k_neighbor_trade": 1.0,
            "k_p5_exchange": 0.0,
            "n_flip": 0.0,
            "s_step": 0.0,
        },
        report_every=1,
        device="cpu",
        strobe_on=True,
        strobe_signature="mag_stag",
    )

    burn_sweeps = 5
    window_sweeps = 10
    burn_steps = burn_sweeps * N
    window_steps = window_sweeps * N
    steps = burn_steps + window_steps
    diag_state = None
    snapshot_holder = {}

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal diag_state, snapshot_holder
        if step <= burn_steps:
            return
        snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
        snapshot_holder = snapshot

    run_sim(
        params,
        seed=1,
        steps=steps,
        report_every=window_steps,
        device="cpu",
        report_callback=report_cb,
    )

    assert snapshot_holder.get("strobe_transitions_window", 0) >= 50
    assert snapshot_holder.get("strobe_unique_states_window", 0) >= 3
    assert snapshot_holder.get("strobe_bidirectional_edges_window", 0) >= 1
