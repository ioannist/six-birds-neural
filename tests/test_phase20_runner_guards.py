import pytest

from scripts.phase20_phrase_decode_proto_syntax_v1 import (
    _parse_phrase_modes,
    _validate_bins,
    _validate_hazard_schedule,
    _validate_token_hold_windows,
)


def test_phase20_hazard_schedule_guard() -> None:
    _validate_hazard_schedule(1, 2, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 2, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(2, 0, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(5, 2, 5)


def test_phase20_phrase_mode_guard() -> None:
    assert _parse_phrase_modes("alternating,chunked") == ["alternating", "chunked"]
    with pytest.raises(ValueError):
        _parse_phrase_modes("")
    with pytest.raises(ValueError):
        _parse_phrase_modes("bad")


def test_phase20_token_hold_guard() -> None:
    _validate_token_hold_windows(1)
    with pytest.raises(ValueError):
        _validate_token_hold_windows(0)


def test_phase20_bins_guard() -> None:
    _validate_bins(2, 2, 2, ["k_radial_focus", "k_entropy"])
    with pytest.raises(ValueError):
        _validate_bins(1, 2, 2, ["k_axis_bias"])
    with pytest.raises(ValueError):
        _validate_bins(2, 1, 2, ["k_entropy"])
    with pytest.raises(ValueError):
        _validate_bins(2, 2, 1, ["k_radial_focus"])
