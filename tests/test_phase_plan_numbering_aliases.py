import os
import subprocess
import sys
from pathlib import Path


def _run_help(script_name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / script_name
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        env=env,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, output
    return output


def test_phase_plan_numbering_aliases_help() -> None:
    for script in ("phase3_p6_drive_v1.py", "phase4_p3_pumping_v1.py"):
        output = _run_help(script)
        assert "--device" in output
        assert "--out-dir" in output
