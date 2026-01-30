import random

import pytest
import torch

from ratchet_gpu.kernels import (
    k_local_exchange,
    k_neighbor_trade,
    k_p5_exchange,
    n_flip,
    s_step,
    spin_flip_color,
    w_local_exchange,
    w_neighbor_exchange,
)
from ratchet_gpu.params import Params
from ratchet_gpu.state import State


@pytest.mark.contract
def test_contract_budgets():
    params = Params(
        shape=(4, 4),
        layers=3,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=6,
        B_k=2,
    )
    state = State.initialize(params, seed=5)
    gen = torch.Generator(device=state.device)
    gen.manual_seed(7)

    kernels = [
        lambda: spin_flip_color(state, 0, gen),
        lambda: spin_flip_color(state, 1, gen),
        lambda: n_flip(state, gen),
        lambda: s_step(state, gen),
        lambda: w_local_exchange(state, gen),
        lambda: w_neighbor_exchange(state, gen),
        lambda: k_local_exchange(state, gen),
        lambda: k_neighbor_trade(state, gen),
        lambda: k_p5_exchange(state, gen),
    ]

    for _ in range(5000):
        kernels[int(torch.randint(0, len(kernels), (1,), generator=gen))]()

    state.check_invariants()

    sample_sites = random.sample(range(state.N), k=min(100, state.N))
    for layer in range(1, state.layers):
        k_layer = state.K_cross_for_layer(layer)
        for idx in sample_sites:
            assert int(k_layer[idx].sum().item()) == params.B_k

    assert int(state.W.sum().item()) == params.B_w
    assert torch.all(state.W >= 0)
    assert torch.all(state.W <= params.l_w)
