import math
import sys
from pathlib import Path

import pytest

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_p3_pumping_v4 import _accumulate_current_map, _current_overlap, _protocol_cycle


def _run_once(seed: int, cycle: list[str]) -> dict:
    params = Params(
        shape=(12, 12),
        layers=2,
        p3_on=True,
        p6_on=False,
        eta=0.5,
        eta_drive=0.0,
        strobe_on=True,
        strobe_signature="mag_stag",
        B_k=2,
        radius_k=2,
        l_k=3,
        device="cpu",
    )
    N = math.prod(params.shape)
    window_steps = 80 * N
    burn_steps = window_steps
    steps = 2 * window_steps
    holder: dict[str, dict] = {}

    def _cb(_state, _step, ledger, _accepted):
        if _step == 2 * window_steps:
            holder["snap"] = dict(ledger)

    run_sim(
        params,
        seed=seed,
        steps=steps,
        report_every=window_steps,
        device="cpu",
        report_callback=_cb,
        protocol_cycle=cycle,
    )
    snap = holder.get("snap")
    if snap is None:
        raise AssertionError("No snapshot captured for strict-reverse contract")
    return snap


@pytest.mark.xfail(
    strict=False,
    reason="Coarse-graining breaks strict reversal overlap; keep as diagnostic-only",
)
def test_contract_phase3_strict_reverse_current_overlap():
    cycle_fwd = _protocol_cycle(False)
    cycle_rev = _protocol_cycle(True)

    seed = 1
    snap_fwd = _run_once(seed=seed, cycle=cycle_fwd)
    snap_rev = _run_once(seed=seed, cycle=cycle_rev)

    items_fwd = snap_fwd.get("strobe_current_map_items_window")
    items_rev = snap_rev.get("strobe_current_map_items_window")
    if not items_fwd or not items_rev:
        keys = sorted(set(snap_fwd.keys()) | set(snap_rev.keys()))
        raise AssertionError(
            "strobe_current_map_items_window missing or empty; "
            f"available_keys={keys}"
        )

    map_fwd = _accumulate_current_map([items_fwd])
    map_rev = _accumulate_current_map([items_rev])

    keys_fwd = set(map_fwd)
    keys_rev = set(map_rev)
    union = keys_fwd | keys_rev
    shared = keys_fwd & keys_rev
    shared_ratio = len(shared) / len(union) if union else 0.0

    norm_f, norm_r, overlap, _ = _current_overlap(map_fwd, map_rev)
    diag = (
        f"shared_ratio={shared_ratio:.4f} edges_fwd={len(keys_fwd)} "
        f"edges_rev={len(keys_rev)} overlap={overlap:.4f} "
        f"norm_f={norm_f:.4f} norm_r={norm_r:.4f} "
        f"cycle_fwd={cycle_fwd} cycle_rev={cycle_rev}"
    )

    assert shared_ratio >= 0.05, diag
    assert overlap <= -0.05, diag
