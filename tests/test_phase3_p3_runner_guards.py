import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from scripts.phase3_p3_pumping_v1 import _case_overrides, _load_preset, _as_params


def test_case_overrides_force_p3_flags(tmp_path):
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(
        """
        {
            "shape": [4,4],
            "layers": 2,
            "beta": 0.5,
            "J": 1.0,
            "kappa_T": 1.0,
            "eta": 0.0,
            "eta_drive": 1.0,
            "l_s": 0,
            "l_w": 3,
            "l_k": 3,
            "B_w": 1,
            "B_k": 1,
            "stencil_policy_w": "l1_ball_odd",
            "stencil_policy_k": "l1_ball_even",
            "radius_w": 1,
            "radius_k": 1,
            "include_zero_k": false,
            "kernel_weights": {"spin_flip_color0":1.0,"spin_flip_color1":1.0},
            "report_every": 10,
            "device": "cpu"
        }
        """
    )
    preset = _load_preset(preset_path)
    base = _as_params(preset, {})

    ctrl = _case_overrides(base, "control_p3_off")
    assert ctrl.p3_on is False
    assert ctrl.p6_on is False
    assert ctrl.eta_drive == 0.0
    assert ctrl.strobe_on is True

    proto = _case_overrides(base, "protocol_p3_on")
    assert proto.p3_on is True
    assert proto.p6_on is False
    assert proto.eta_drive == 0.0
    assert proto.strobe_on is True


def test_missing_preset_raises():
    with pytest.raises(FileNotFoundError):
        _load_preset(Path("nonexistent.json"))
