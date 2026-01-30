import sys
from pathlib import Path
from typing import Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim
from ratchet_gpu.diagnostics import compute_snapshot


@pytest.mark.contract
def test_phase2_drive_contract():
    null_mean = _run_case(p6_on=False, eta_drive=0.0)
    drive_mean = _run_case(p6_on=True, eta_drive=2.0)

    assert abs(null_mean) <= 5e-4
    assert drive_mean >= 1e-3
    assert (drive_mean - null_mean) >= 5e-4


def _k_drive_rate(snapshot: Dict[str, float], k_names: List[str]) -> float:
    rates = snapshot.get("ep_rate_by_kernel_proposal_window", {})
    return sum(float(rates.get(k, 0.0)) for k in k_names)


def _run_case(p6_on: bool, eta_drive: float) -> float:
    shape = (8, 8)
    N = shape[0] * shape[1]
    params = Params(
        shape=shape,
        layers=2,
        p3_on=False,
        p6_on=p6_on,
        beta=0.5,
        J=1.0,
        kappa_T=1.0,
        eta=0.5,
        eta_drive=eta_drive,
        l_s=0,
        l_w=3,
        l_k=3,
        B_w=5,
        B_k=2,
        stencil_policy_w="l1_ball_odd",
        stencil_policy_k="l1_ball_even",
        radius_w=1,
        radius_k=2,
        include_zero_k=False,
        kernel_weights={
            "spin_flip_color0": 1.0,
            "spin_flip_color1": 1.0,
            "w_local": 1.0,
            "w_neighbor": 0.0,
            "k_local": 2.0,
            "k_neighbor_trade": 2.0,
            "k_p5_exchange": 0.0,
            "n_flip": 0.0,
            "s_step": 0.0,
        },
        report_every=1,
        device="cpu",
    )
    burn_sweeps = 30
    window_sweeps = 25
    min_windows = 6
    max_windows = 12
    last_m = 5
    burn_steps = burn_sweeps * N
    window_steps = window_sweeps * N
    max_steps = burn_steps + max_windows * window_steps
    k_names = ["k_local", "k_neighbor_trade", "k_p5_exchange"]

    rates: List[float] = []
    diag_state = None

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal rates, diag_state
        if step <= burn_steps:
            return
        if len(rates) >= max_windows:
            return
        snapshot, diag_state = compute_snapshot(
            state, step, ep_ledger, accepted_frac, diag_state
        )
        rates.append(_k_drive_rate(snapshot, k_names))

    run_sim(
        params,
        seed=1,
        steps=max_steps,
        report_every=window_steps,
        device="cpu",
        report_callback=report_cb,
    )
    tail = rates[-last_m:] if rates else [0.0]
    return sum(tail) / len(tail)
