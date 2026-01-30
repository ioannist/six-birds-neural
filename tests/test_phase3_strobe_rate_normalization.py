import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3_p3_pumping_v4 import _strobe_rate_per_proposal


def test_strobe_rate_per_proposal_basic():
    snap = {
        "strobe_rate_window": 0.02,
        "strobe_transitions_window": 200,
        "window_proposals": 50000,
    }
    expected = 0.02 * 200 / 50000
    assert abs(_strobe_rate_per_proposal(snap) - expected) < 1e-12


def test_strobe_rate_per_proposal_fallbacks():
    snap = {
        "strobe_rate_window": 0.02,
        "strobe_transitions_window": 200,
        "window_steps": 40000,
    }
    expected = 0.02 * 200 / 40000
    assert abs(_strobe_rate_per_proposal(snap) - expected) < 1e-12
    snap = {
        "strobe_rate_window": 0.02,
        "strobe_transitions_window": 0,
        "window_proposals": 0,
    }
    assert _strobe_rate_per_proposal(snap) == 0.0
