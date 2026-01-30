import torch

from ratchet_gpu.params import Params
from ratchet_gpu.state import State
from ratchet_gpu.spatial import compute_spatial_maps, finite_check


def test_spatial_maps_shapes() -> None:
    params = Params(
        shape=(8, 8),
        layers=3,
        B_w=200,
        B_k=2,
        radius_w=1,
        radius_k=2,
    )
    state = State.initialize(params, seed=1)
    maps = compute_spatial_maps(
        state,
        [
            "sigma",
            "w_mass",
            "w_entropy",
            "w_axis_bias",
            "k_entropy",
            "k_axis_bias",
            "mismatch",
        ],
    )
    assert maps["sigma"].shape == (3, 8, 8)
    assert maps["w_mass"].shape == (3, 8, 8)
    assert maps["w_entropy"].shape == (3, 8, 8)
    assert maps["w_axis_bias"].shape == (3, 8, 8)
    assert maps["k_entropy"].shape == (2, 8, 8)
    assert maps["k_axis_bias"].shape == (2, 8, 8)
    assert maps["mismatch"].shape == (2, 8, 8)
    ok, bad = finite_check(maps)
    assert ok, f"Non-finite maps: {bad}"
