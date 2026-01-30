import pytest

from scripts.phase16_causal_motif_semantics_v1 import _validate_hazard_schedule


def test_validate_hazard_schedule_ok() -> None:
    _validate_hazard_schedule(3, 2, 10)


def test_validate_hazard_schedule_start() -> None:
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 1, 10)


def test_validate_hazard_schedule_duration() -> None:
    with pytest.raises(ValueError):
        _validate_hazard_schedule(1, 0, 10)


def test_validate_hazard_schedule_bounds() -> None:
    with pytest.raises(ValueError):
        _validate_hazard_schedule(6, 6, 10)
