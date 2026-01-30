import torch

from ratchet_gpu.interventions import apply_k_redistribute_uniform_in_region, parse_rect
from ratchet_gpu.params import Params
from ratchet_gpu.state import State


def test_k_redistribute_preserves_budget() -> None:
    params = Params(
        shape=(6, 6),
        layers=3,
        B_k=2,
        radius_k=2,
        l_k=2,
    )
    state = State.initialize(params, seed=1)
    _, flat_idx = parse_rect("0:2,0:2", params.shape)
    flat_idx = flat_idx.to(device=state.device)

    sum_before = int(state.K.sum().item())
    apply_k_redistribute_uniform_in_region(state, params, flat_idx, interfaces="0")
    sum_after = int(state.K.sum().item())

    assert sum_before == sum_after
    assert int(state.K.min().item()) >= 0
    assert int(state.K.max().item()) <= params.l_k
    K_layer = state.K[0, flat_idx]
    assert torch.all(K_layer.sum(dim=-1) == params.B_k)
