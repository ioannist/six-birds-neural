import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_p3_pumping_v4 import _match_cycle_weights


def test_match_cycle_weights_overrides_non_cycle():
    kw = {
        "spin_flip_color0": 0.2,
        "spin_flip_color1": 0.2,
        "w_local": 0.1,
        "extra": 0.7,
    }
    cycle = ["spin_flip_color0", "w_local", "k_local"]
    matched = _match_cycle_weights(kw, cycle)
    assert matched["spin_flip_color0"] == 1.0
    assert matched["w_local"] == 1.0
    assert matched["k_local"] == 1.0
    assert matched["spin_flip_color1"] == 0.0
    assert matched["extra"] == 0.0
    assert sum(matched.values()) > 0.0
