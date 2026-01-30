import importlib.util
import pathlib
import sys


def _load_screen_module():
    path = pathlib.Path(__file__).parent.parent / "scripts" / "phase1_null_screen_v4.py"
    spec = importlib.util.spec_from_file_location("phase1_screen_v4", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase1_screen_v4"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_rate_micro():
    mod = _load_screen_module()
    rate = mod._rate_micro(10.0, 100)
    assert rate == 0.1
    rate_zero = mod._rate_micro(10.0, 0)
    assert rate_zero == 10.0
