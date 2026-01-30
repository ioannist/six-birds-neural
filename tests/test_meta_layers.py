import torch

from ratchet_gpu.energy import mismatch_mean
from ratchet_gpu.kernels import k_local_exchange, k_neighbor_trade, k_p5_exchange
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim
from ratchet_gpu.state import State


def test_meta_layer_shapes_and_budgets():
    params = Params(
        shape=(4, 4),
        layers=3,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=4,
        B_k=2,
        device="cpu",
    )
    state = State.initialize(params, seed=1)
    N = state.N

    assert state.sigma.shape == (3, N)
    assert state.n.shape == (3, N)
    assert state.s.shape == (3, N)
    assert state.W.shape[0] == 3
    assert state.K_cross.shape[0] == 2

    gen = torch.Generator(device=state.device)
    gen.manual_seed(3)

    for _ in range(1000):
        pick = int(torch.randint(0, 3, (1,), generator=gen))
        if pick == 0:
            k_local_exchange(state, gen)
        elif pick == 1:
            k_neighbor_trade(state, gen)
        else:
            k_p5_exchange(state, gen)

    state.check_invariants()


def test_meta_coupling_reduces_mismatch():
    base_kwargs = dict(
        shape=(4, 4),
        layers=3,
        beta=1.0,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=4,
        B_k=2,
        device="cpu",
        p3_on=False,
        p6_on=False,
        kernel_weights={"k_local": 1.0},
    )

    params_weak = Params(eta=0.0, **base_kwargs)
    params_strong = Params(eta=0.6, **base_kwargs)

    weak = run_sim(params_weak, seed=2, steps=30000, report_every=30000)
    strong = run_sim(params_strong, seed=2, steps=30000, report_every=30000)

    assert strong["mismatchMean"] < weak["mismatchMean"] - 1e-3


def test_drive_only_reduces_mismatch_and_ep_positive():
    params = Params(
        shape=(4, 4),
        layers=3,
        beta=1.0,
        eta=0.0,
        eta_drive=0.8,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=4,
        B_k=2,
        device="cpu",
        p3_on=False,
        p6_on=True,
        kernel_weights={"k_local": 1.0},
        report_every=20000,
    )

    state = State.initialize(params, seed=5)
    mismatch_start = mismatch_mean(state)

    summary = run_sim(params, seed=5, steps=20000, report_every=20000)

    assert summary["mismatchMean"] < mismatch_start
    assert summary["epMicroTotal_k_local"] > 0.0
