import pytest

from scripts.phase13_pattern_memory_setpoints_v1 import _validate_injury_schedule


def test_validate_injury_schedule() -> None:
    _validate_injury_schedule(1, 1, 10)
    with pytest.raises(ValueError):
        _validate_injury_schedule(0, 1, 10)
    with pytest.raises(ValueError):
        _validate_injury_schedule(1, 0, 10)
    with pytest.raises(ValueError):
        _validate_injury_schedule(10, 2, 10)
