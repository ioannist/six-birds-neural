import torch

from ratchet_gpu.kernels import excitable_step_color
from ratchet_gpu.params import Params
from ratchet_gpu.state import State


def test_excitable_state_cycle() -> None:
    params = Params(
        shape=(2, 2),
        layers=2,
        sigma_mode="excitable4",
        B_w=0,
        B_k=0,
        l_w=1,
        l_k=1,
        l_s=0,
        exc_init_frac=0.0,
        exc_p_spont=0.0,
        exc_theta=1e6,
        exc_beta=1.0,
        exc_p_recover=1.0,
    )
    state = State.initialize(params, seed=0)
    state.sigma.fill_(1)
    gen = torch.Generator(device=state.device).manual_seed(0)

    excitable_step_color(state, 0, gen)
    excitable_step_color(state, 1, gen)
    assert torch.all(state.sigma == 2)

    excitable_step_color(state, 0, gen)
    excitable_step_color(state, 1, gen)
    assert torch.all(state.sigma == 3)

    excitable_step_color(state, 0, gen)
    excitable_step_color(state, 1, gen)
    assert torch.all(state.sigma == 0)
