import sys
from pathlib import Path

from ratchet_gpu.ep import StrobeTracker

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_p3_pumping_v4 import _current_overlap


def test_current_map_canonical_keys():
    tracker = StrobeTracker()
    a = (1, 0)
    b = (0, 1)
    tracker.transitions = {a: {b: 10}, b: {a: 2}}
    tracker.total = 12
    cmap = tracker.current_map()
    assert len(cmap) == 1
    key = next(iter(cmap.keys()))
    assert key == tuple(sorted((a, b)))


def test_overlap_reversal_negative():
    edge = ((0,), (1,))
    fwd = {edge: 1.0, ((1,), (2,)): -2.0}
    rev = {edge: -1.0, ((1,), (2,)): 2.0}
    _, _, overlap, rev_error = _current_overlap(fwd, rev)
    assert overlap < -0.9
    assert rev_error < 0.2
