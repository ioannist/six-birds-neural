import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_p3_pumping_v4 import _should_check_reversal


def test_reversal_check_delay():
    assert _should_check_reversal(5, 10) is False
    assert _should_check_reversal(10, 10) is True
    assert _should_check_reversal(10, 20) is False
