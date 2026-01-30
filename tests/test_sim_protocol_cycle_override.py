import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


def _capture_ledger(params: Params, protocol_cycle):
    holder = {}

    def report_cb(state, step, ep_ledger, accepted_frac):
        holder.update(ep_ledger)

    run_sim(
        params,
        seed=1,
        steps=10,
        report_every=10,
        device="cpu",
        report_callback=report_cb,
        protocol_cycle=protocol_cycle,
    )
    return holder


def test_protocol_cycle_len_override():
    protocol = ["spin_flip_color0", "spin_flip_color1", "w_local", "w_neighbor"]
    params = Params(
        shape=(4, 4),
        layers=2,
        p3_on=True,
        p6_on=False,
        beta=0.5,
        J=1.0,
        kappa_T=1.0,
        eta=0.0,
        eta_drive=0.0,
        l_s=0,
        l_w=3,
        l_k=0,
        B_w=5,
        B_k=0,
        radius_w=1,
        radius_k=0,
        kernel_weights={"spin_flip_color0": 1.0, "spin_flip_color1": 1.0, "w_local": 1.0},
        report_every=10,
        device="cpu",
        strobe_on=True,
        strobe_signature="mag_stag",
    )
    ledger = _capture_ledger(params, protocol)
    assert ledger.get("strobe_cycle_len") == len(protocol)

    params_off = Params.from_dict(params, {"p3_on": False})
    ledger_off = _capture_ledger(params_off, protocol)
    assert ledger_off.get("strobe_cycle_len") == len(protocol)
