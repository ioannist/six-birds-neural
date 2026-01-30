import inspect

from ratchet_gpu.params import Params
from scripts import phase5_p3p6_combo_v1 as phase5


def test_phase5_case_overrides() -> None:
    base = Params(
        shape=(4, 4),
        layers=2,
        kernel_weights={"k_local": 0.0, "k_neighbor_trade": 0.0},
    )
    control = phase5.build_case_params(base, "combo_control", eta=1.0, strobe_sig="mag_stag")
    protocol = phase5.build_case_params(base, "combo_protocol", eta=1.0, strobe_sig="mag_stag")
    assert control.p6_on is True
    assert control.p3_on is False
    assert protocol.p6_on is True
    assert protocol.p3_on is True


def test_phase5_protocol_cycle_passed() -> None:
    src = inspect.getsource(phase5.run_case)
    assert "protocol_cycle=cycle" in src
    assert phase5._case_cycle()
