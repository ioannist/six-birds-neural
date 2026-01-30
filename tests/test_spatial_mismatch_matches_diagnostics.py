import torch

from ratchet_gpu.diagnostics import cross_mismatch
from ratchet_gpu.params import Params
from ratchet_gpu.spatial import mismatch_abs_grid
from ratchet_gpu.state import State


def test_mismatch_grid_matches_diagnostics() -> None:
    params = Params(
        shape=(6, 6),
        layers=3,
        B_k=2,
        l_k=2,
        radius_k=2,
        device="cpu",
    )
    state = State.initialize(params, seed=1)
    metrics = cross_mismatch(state, p=1)
    grid = mismatch_abs_grid(state, p=1.0)

    mean_by_interface = grid.mean(dim=(1, 2)).cpu()
    expected = torch.tensor(
        metrics["mismatch_abs_by_interface"], dtype=mean_by_interface.dtype
    )
    torch.testing.assert_close(mean_by_interface, expected, rtol=0.0, atol=1e-12)
