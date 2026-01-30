import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2_separability_v5 import build_case_params, _as_params, _load_preset
from pathlib import Path
import json


def test_case_overrides_v5(tmp_path):
    preset = {
        "shape": [4, 4],
        "layers": 2,
        "kernel_weights": {"w_neighbor": 0.25, "k_local": 0.0, "k_neighbor_trade": 0.0},
    }
    base = _as_params(preset, {"device": "cpu"})
    meta = build_case_params(base, "meta_null_k")
    assert meta.p6_on is False and meta.eta_drive == 0.0
    assert meta.B_k == 2 and meta.radius_k == 2 and meta.l_k == 3
    assert meta.kernel_weights.get("k_local", 0.0) > 0.0
    assert meta.kernel_weights.get("k_neighbor_trade", 0.0) > 0.0
    drive = build_case_params(base, "p6_drive_k")
    assert drive.p6_on is True and drive.eta_drive == 1.0
    assert drive.B_k == 2 and drive.radius_k == 2 and drive.l_k == 3
    assert drive.kernel_weights.get("k_neighbor_trade", 0.0) > 0.0
