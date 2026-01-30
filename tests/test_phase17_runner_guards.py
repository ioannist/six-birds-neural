import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase17_directional_motif_semantics_v1 import _validate_hazard_schedule


def test_validate_hazard_schedule_ok() -> None:
    _validate_hazard_schedule(2, 2, 6)


def test_validate_hazard_schedule_errors() -> None:
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 1, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(2, 0, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(5, 2, 5)
