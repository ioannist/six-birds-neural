import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("imageio")
pytest.importorskip("PIL")


def _write_npz_sequence(npz_dir: Path, key: str, frames: int = 10) -> None:
    npz_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, frames + 1):
        arr = np.zeros((8, 8), dtype=np.float32)
        arr[:, i % 8] = 1.0
        path = npz_dir / f"seed1_win{i:04d}.npz"
        np.savez(path, **{key: arr})


def test_phase10_render_npz_smoke():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "phase10_clockwork_render_npz_v1.py"
    spec = importlib.util.spec_from_file_location("phase10_clockwork_render_npz_v1", module_path)
    assert spec and spec.loader, "failed to load render module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        npz_dir = root / "npz"
        _write_npz_sequence(npz_dir, "sigma_l0", frames=8)
        out_dir = root / "renders"
        outputs = module.render_npz_dir(
            npz_dir=npz_dir,
            out_dir=out_dir,
            seed=1,
            keys=["sigma_l0"],
            max_frames=8,
            fps=5,
        )
        assert outputs, "no render outputs produced"
        assert outputs[0].exists(), "render output missing"
        assert outputs[0].stat().st_size > 0, "render output is empty"
