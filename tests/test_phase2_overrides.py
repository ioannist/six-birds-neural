import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2_separability_v2 import build_case_params, _as_params, _load_preset
from pathlib import Path
import json


def test_case_overrides_meta_and_p6(tmp_path, monkeypatch):
    preset = {
        "shape": [4, 4],
        "layers": 2,
        "kernel_weights": {"k_local": 0.0, "k_neighbor_trade": 0.0, "w_neighbor": 0.25},
    }
    base = _as_params(preset, {"device": "cpu"})
    meta = build_case_params(base, "meta_null_k")
    assert not meta.p3_on and not meta.p6_on
    assert meta.B_k > 0 and meta.radius_k > 0
    assert meta.kernel_weights.get("k_local", 0.0) > 0.0 or meta.kernel_weights.get("k_neighbor_trade", 0.0) > 0.0

    p6 = build_case_params(base, "p6_drive_k")
    assert p6.p6_on is True
    assert p6.eta_drive > 0
    assert p6.B_k > 0 and p6.radius_k > 0
    assert p6.kernel_weights.get("k_local", 0.0) > 0.0 or p6.kernel_weights.get("k_neighbor_trade", 0.0) > 0.0
