import math

import pytest
import torch

from ratchet_gpu.energy import (
    delta_e_spin_flip,
    delta_e_w_local_exchange,
    delta_e_w_neighbor_exchange,
    energy_total,
)
from ratchet_gpu.lattice import Lattice
from ratchet_gpu.params import Params
from ratchet_gpu.state import State


def _make_state(seed: int = 0) -> State:
    shape = (4, 4)
    layers = 2
    radius_w = 1
    l_w = 3
    K_W = 4  # l1_ball_odd with radius=1 in 2D
    capacity = l_w * layers * math.prod(shape) * K_W
    B_w = int(round(0.2 * capacity))

    params = Params(
        shape=shape,
        layers=layers,
        p3_on=False,
        p6_on=False,
        beta=1.0,
        J=1.0,
        kappa_T=1.0,
        eta=0.0,
        eta_drive=0.0,
        l_s=0,
        l_w=l_w,
        l_k=1,
        B_w=B_w,
        B_k=0,
        radius_w=radius_w,
        radius_k=0,
        stencil_policy_w="l1_ball_odd",
        stencil_policy_k="l1_ball_even",
        kernel_weights={
            "spin_flip_color0": 1.0,
            "spin_flip_color1": 1.0,
            "n_flip": 0.0,
            "s_step": 0.0,
            "w_local": 1.0,
            "k_local": 0.0,
            "k_neighbor_trade": 0.0,
            "k_p5_exchange": 0.0,
            "w_neighbor": 1.0,
        },
        report_every=100,
        device="cpu",
    )
    return State.initialize(params, seed=seed)


def _clone_state(state: State) -> State:
    return State(
        params=state.params,
        lattice=state.lattice,
        R_W=state.R_W,
        R_K=state.R_K,
        sigma=state.sigma.clone(),
        n=state.n.clone(),
        s=state.s.clone(),
        W=state.W.clone(),
        K=state.K.clone(),
        color_indices=state.color_indices,
    )


def _find_w_local_move(state: State) -> tuple[int, int, int, int]:
    for layer in range(state.layers):
        for idx in range(state.N):
            site = state.W[layer, idx]
            for k1 in range(state.K_W):
                if site[k1] <= 0:
                    continue
                for k2 in range(state.K_W):
                    if k2 == k1:
                        continue
                    if site[k2] >= state.params.l_w:
                        continue
                    return layer, idx, k1, k2
    raise AssertionError("no feasible w_local_exchange move found")


def _find_w_neighbor_move(state: State) -> tuple[int, int, int, int, int]:
    lattice = state.lattice
    for layer in range(state.layers):
        for u in range(state.N):
            coord_u = state.lattice.index_to_coord(torch.tensor(u))
            for axis in range(lattice.d):
                for sign in (-1, 1):
                    delta = [0] * lattice.d
                    delta[axis] = sign
                    coord_v = coord_u + state.R_W.new_tensor(delta)
                    v = int(
                        state.lattice.coord_to_index(state.lattice.wrap_coord(coord_v)).item()
                    )
                    for k_u in range(state.K_W):
                        if state.W[layer, u, k_u] >= state.params.l_w:
                            continue
                        for k_v in range(state.K_W):
                            if state.W[layer, v, k_v] <= 0:
                                continue
                            return layer, u, v, k_u, k_v
    raise AssertionError("no feasible w_neighbor_exchange move found")


def test_delta_e_spin_flip_matches_energy() -> None:
    state = _make_state(seed=1)
    layer = 0
    idx = 0

    delta_e, _ = delta_e_spin_flip(state, layer, idx)
    before = float(energy_total(state).item())

    next_state = _clone_state(state)
    next_state.sigma[layer, idx] *= -1
    after = float(energy_total(next_state).item())

    assert pytest.approx(after - before, abs=1e-6) == delta_e


def test_delta_e_w_local_matches_energy() -> None:
    state = _make_state(seed=2)
    layer, idx, k1, k2 = _find_w_local_move(state)

    delta_e = delta_e_w_local_exchange(state, layer, idx, k1, k2)
    before = float(energy_total(state).item())

    next_state = _clone_state(state)
    next_state.W[layer, idx, k1] -= 1
    next_state.W[layer, idx, k2] += 1
    after = float(energy_total(next_state).item())

    assert pytest.approx(after - before, abs=1e-6) == delta_e


def test_delta_e_w_neighbor_matches_energy() -> None:
    state = _make_state(seed=3)
    layer, u, v, k_u, k_v = _find_w_neighbor_move(state)

    delta_e = delta_e_w_neighbor_exchange(state, layer, u, v, k_u, k_v)
    before = float(energy_total(state).item())

    next_state = _clone_state(state)
    next_state.W[layer, v, k_v] -= 1
    next_state.W[layer, u, k_u] += 1
    after = float(energy_total(next_state).item())

    assert pytest.approx(after - before, abs=1e-6) == delta_e
