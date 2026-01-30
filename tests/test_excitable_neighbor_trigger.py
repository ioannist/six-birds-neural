import torch

from ratchet_gpu.kernels import excitable_step_color
from ratchet_gpu.lattice import Lattice, generate_stencil
from ratchet_gpu.params import Params
from ratchet_gpu.state import State


def test_excitable_neighbor_trigger() -> None:
    shape = (2, 2)
    lattice = Lattice(shape)
    R_W = generate_stencil(
        d=lattice.d,
        policy="l1_ball_odd",
        radius=1,
        bipartite=True,
        shape=shape,
    )
    K_W = int(R_W.shape[0])
    layers = 2
    l_w = 2
    total_capacity = l_w * layers * lattice.N * K_W

    params = Params(
        shape=shape,
        layers=layers,
        sigma_mode="excitable4",
        B_w=total_capacity,
        B_k=0,
        l_w=l_w,
        l_k=1,
        l_s=0,
        exc_init_frac=0.0,
        exc_p_spont=0.0,
        exc_theta=0.5,
        exc_beta=50.0,
        exc_p_recover=1.0,
    )
    state = State.initialize(params, seed=1)
    state.W.fill_(l_w)
    idx0 = state.color_indices[0][0]
    idx1 = state.color_indices[0][1]
    state.sigma[0, idx0] = 0
    state.sigma[0, idx1] = 1
    state.sigma[1].fill_(0)

    gen = torch.Generator(device=state.device).manual_seed(0)
    excitable_step_color(state, 0, gen)

    assert torch.all(state.sigma[0, idx0] == 1)
