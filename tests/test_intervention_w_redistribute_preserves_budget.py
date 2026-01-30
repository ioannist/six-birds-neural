import torch

from ratchet_gpu.interventions import apply_w_lesion_redistribute, parse_rect
from ratchet_gpu.params import Params
from ratchet_gpu.state import State


def test_w_lesion_preserves_budget() -> None:
    params = Params(
        shape=(8, 8),
        layers=2,
        B_w=600,
        radius_w=1,
    )
    state = State.initialize(params, seed=2)
    mask, flat_idx = parse_rect("0:4,0:4", (8, 8))
    device = state.device
    flat_idx = flat_idx.to(device=device)
    sum_before = int(state.W.sum().item())
    region_before = int(state.W[:, flat_idx].sum().item())
    apply_w_lesion_redistribute(state, params, flat_idx, layers="all", frac=1.0)
    sum_after = int(state.W.sum().item())
    region_after = int(state.W[:, flat_idx].sum().item())
    assert sum_before == sum_after
    assert int(state.W.min().item()) >= 0
    assert int(state.W.max().item()) <= params.l_w
    assert region_after <= region_before
