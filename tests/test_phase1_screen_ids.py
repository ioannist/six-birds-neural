import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase1_null_screen_v4 import _config_id


def test_config_id_unique_w_fill_precision():
    a = _config_id(beta=0.25, J=1.0, w_fill=0.005, w_neighbor_weight=0.25)
    b = _config_id(beta=0.25, J=1.0, w_fill=0.01, w_neighbor_weight=0.25)
    assert a != b
