import pytest

from scripts import phase7_meta_layer_sanity_v1 as phase7


def test_phase7_candidates_require_zero() -> None:
    assert phase7._parse_candidates("0,0.5,1.0", "eta-candidates") == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        phase7._parse_candidates("0.5,1.0", "eta-candidates")


def test_phase7_layers_guard() -> None:
    with pytest.raises(ValueError):
        phase7._validate_layers(2)


def test_phase7_drive_only_eta_mode() -> None:
    assert phase7._drive_only_eta(1.0, "eta_best") == 1.0
    assert phase7._drive_only_eta(1.0, "zero") == 0.0
    with pytest.raises(ValueError):
        phase7._drive_only_eta(1.0, "bad_mode")
