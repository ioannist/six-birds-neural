import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_p3_pumping_v4 import _effective_min_strobe_transitions


def test_min_strobe_transitions_clamped():
    used = _effective_min_strobe_transitions(200, window_steps=1404, cycle_len=9)
    assert used <= 155
    assert _effective_min_strobe_transitions(50, window_steps=1404, cycle_len=9) == 50
    assert _effective_min_strobe_transitions(0, window_steps=1404, cycle_len=9) == 0
