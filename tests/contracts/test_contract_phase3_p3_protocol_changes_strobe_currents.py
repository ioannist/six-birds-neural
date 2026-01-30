import math

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim, _cycle_list


def _match_cycle_weights(names: list[str]) -> dict[str, float]:
    cycle = set(_cycle_list())
    return {name: (1.0 if name in cycle else 0.0) for name in names}


def _run_case(seed: int, p3_on: bool, eta: float) -> dict:
    params = Params(
        shape=(12, 12),
        layers=2,
        p3_on=p3_on,
        p6_on=False,
        eta=eta,
        eta_drive=0.0,
        strobe_on=True,
        strobe_signature="mag_stag",
        B_k=2,
        radius_k=2,
        l_k=3,
        device="cpu",
    )
    cycle = _cycle_list()
    params = Params.from_dict(
        params,
        {"kernel_weights": _match_cycle_weights(list(params.kernel_weights.keys()))},
    )

    N = math.prod(params.shape)
    burn_steps = 50 * N
    window_steps = 20 * N
    windows = 10
    steps = burn_steps + window_steps * windows

    snapshots: list[dict] = []

    def _cb(_state, _step, ledger, _accepted):
        if _step > burn_steps:
            snapshots.append(dict(ledger))

    run_sim(
        params,
        seed=seed,
        steps=steps,
        report_every=window_steps,
        device="cpu",
        report_callback=_cb,
        protocol_cycle=cycle,
    )
    if not snapshots:
        raise AssertionError("No snapshots captured for Phase3 contract")
    return snapshots[-1]


def test_contract_phase3_p3_protocol_changes_strobe_currents():
    eta = 2.0
    control = _run_case(seed=1, p3_on=False, eta=eta)
    protocol = _run_case(seed=1, p3_on=True, eta=eta)

    for snap in (control, protocol):
        assert snap.get("strobe_unique_states_window", 0) >= 3
        assert snap.get("strobe_bidirectional_edges_window", 0) >= 1
        assert snap.get("window_accept_frac", 0.0) >= 0.005

    control_l2 = float(control.get("strobe_current_l2_window", 0.0))
    protocol_l2 = float(protocol.get("strobe_current_l2_window", 0.0))
    rel_change = abs(protocol_l2 - control_l2) / max(control_l2, 1e-9)

    assert rel_change >= 0.15, (
        f"control_l2={control_l2:.6g} protocol_l2={protocol_l2:.6g} "
        f"rel_change={rel_change:.6g}"
    )
