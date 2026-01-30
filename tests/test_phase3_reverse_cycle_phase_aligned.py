import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_p3_pumping_v4 import _protocol_cycle


def test_reverse_cycle_phase_aligned():
    fwd = _protocol_cycle(False)
    rev = _protocol_cycle(True)
    assert fwd and rev
    assert fwd[0] == rev[-1]
    assert set(fwd) == set(rev)
    assert len(fwd) == len(rev)
    assert rev == list(reversed(fwd))
