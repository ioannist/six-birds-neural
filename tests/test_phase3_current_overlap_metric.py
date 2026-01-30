import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_p3_pumping_v4 import _current_overlap


def test_current_overlap_signs():
    edge = ((0,), (1,))
    fwd = {edge: 1.0, ((1,), (2,)): -2.0}
    rev = {edge: -1.0, ((1,), (2,)): 2.0}
    norm_f, norm_r, overlap, rev_error = _current_overlap(fwd, rev)
    assert norm_f > 0
    assert norm_r > 0
    assert overlap < -0.9
    assert rev_error < 0.2

    rev_same = {edge: 1.0, ((1,), (2,)): -2.0}
    _, _, overlap2, rev_error2 = _current_overlap(fwd, rev_same)
    assert overlap2 > 0.9
    assert rev_error2 > 0.9
