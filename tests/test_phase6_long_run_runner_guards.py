from ratchet_gpu.params import Params
from scripts import phase6_long_run_stability_v1 as phase6


def test_phase6_case_overrides() -> None:
    base = Params(shape=(4, 4), layers=2, eta=0.3, eta_drive=2.0)
    null_params = phase6.build_case_params(base, "null_full", eta_override=None)
    drive_params = phase6.build_case_params(base, "p6_drive", eta_override=None)
    combo_params = phase6.build_case_params(base, "p3p6_combo", eta_override=None)

    assert null_params.p3_on is False
    assert null_params.p6_on is False
    assert null_params.eta_drive == 0.0
    assert drive_params.p3_on is False
    assert drive_params.p6_on is True
    assert combo_params.p3_on is True
    assert combo_params.p6_on is True


def test_phase6_slim_snapshot() -> None:
    snap = {
        "strobe_current_map_items_window": [{"a": [0], "b": [1], "j": 1.0}],
        "strobe_currents_window": [{"a": [0], "b": [1], "j": 1.0}],
        "strobe_top_states_window": [{"state": [0], "count": 1}],
        "keep": 1,
    }
    slim = phase6._slim_snapshot(snap)
    assert "strobe_current_map_items_window" not in slim
    assert "strobe_currents_window" not in slim
    assert "strobe_top_states_window" not in slim
    assert slim["keep"] == 1
