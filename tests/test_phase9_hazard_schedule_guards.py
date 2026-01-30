import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase9_hazard_attention_highways_v1 import _validate_hazard_schedule


def test_hazard_schedule_guards() -> None:
    _validate_hazard_schedule(1, 1, 5)
    _validate_hazard_schedule(3, 2, 5)

    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 2, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(2, 0, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(4, 3, 5)
