import pytest

from scripts.phase19_motif_phrase_semantics_v1 import (
    _validate_bins,
    _validate_hazard_schedule,
    _validate_phrase_args,
    _validate_ring_thickness,
)


def test_phase19_hazard_schedule_guards() -> None:
    _validate_hazard_schedule(1, 2, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 2, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(2, 0, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(4, 3, 5)


def test_phase19_phrase_guards() -> None:
    _validate_phrase_args("alternating", 0, 2, True)
    with pytest.raises(ValueError):
        _validate_phrase_args("other", 0, 2, True)
    with pytest.raises(ValueError):
        _validate_phrase_args("alternating", 2, 2, True)
    with pytest.raises(ValueError):
        _validate_phrase_args("alternating", 0, 1, True)
    with pytest.raises(ValueError):
        _validate_phrase_args("alternating", 0, 2, False)


def test_phase19_ring_thickness_guard() -> None:
    _validate_ring_thickness(1)
    with pytest.raises(ValueError):
        _validate_ring_thickness(0)


def test_phase19_bins_guard() -> None:
    _validate_bins(2, 2, 2)
    with pytest.raises(ValueError):
        _validate_bins(1, 2, 2)
    with pytest.raises(ValueError):
        _validate_bins(2, 1, 2)
    with pytest.raises(ValueError):
        _validate_bins(2, 2, 1)
