import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase15_motif_semantics_routing_v1 import _validate_hazard_schedule


def test_validate_hazard_schedule_errors():
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 1, 10)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(1, 0, 10)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(9, 3, 10)


def test_validate_hazard_schedule_ok():
    _validate_hazard_schedule(2, 3, 10)
