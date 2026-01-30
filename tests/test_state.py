import pytest
import torch

from ratchet_gpu.energy import delta_e_k_local_exchange, energy_total
from ratchet_gpu.kernels import k_local_exchange
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_null
from ratchet_gpu.state import State


def test_k_local_exchange_invariants():
    params = Params(
        shape=(4, 4),
        layers=2,
        beta=1.0,
        eta=1.0,
        J=1.0,
        l_w=3,
        B_w=2,
        l_k=3,
        B_k=2,
        radius_k=2,
        device="cpu",
    )
    state = State.initialize(params, seed=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)

    for _ in range(1000):
        k_local_exchange(state, generator)

    state.check_invariants()


def test_delta_e_matches_energy():
    params = Params(
        shape=(3, 3),
        layers=2,
        beta=1.0,
        eta=1.0,
        J=1.0,
        l_w=3,
        B_w=2,
        l_k=2,
        B_k=2,
        radius_k=2,
        device="cpu",
    )
    state = State.initialize(params, seed=3)

    layer = 1
    found = False
    for i in range(state.N):
        k_site = state.K[layer - 1, i]
        k1_candidates = (k_site > 0).nonzero(as_tuple=False).flatten().tolist()
        k2_candidates = (k_site < params.l_k).nonzero(as_tuple=False).flatten().tolist()
        for k1 in k1_candidates:
            for k2 in k2_candidates:
                if k1 != k2:
                    found = True
                    break
            if found:
                break
        if found:
            break

    assert found, "no feasible exchange found"

    e_before = float(energy_total(state).item())
    delta_e = delta_e_k_local_exchange(state, layer, i, k1, k2)

    state.K[layer - 1, i, k1] -= 1
    state.K[layer - 1, i, k2] += 1

    e_after = float(energy_total(state).item())
    assert abs((e_after - e_before) - delta_e) <= 1e-6


def test_null_ep_rate_near_zero():
    params = Params(
        shape=(4, 4),
        layers=2,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        B_w=2,
        l_k=3,
        B_k=2,
        radius_k=2,
        device="cpu",
    )

    for seed in [1, 2, 3]:
        summary = run_null(
            params,
            seed=seed,
            steps=20000,
            report_every=20000,
            device="cpu",
        )
        assert abs(summary["epExactRateWindowLast"]) <= 2e-3


def test_w_init_randomized_seeded():
    params = Params(
        shape=(4, 4),
        layers=2,
        beta=1.0,
        eta=0.0,
        J=1.0,
        l_w=3,
        B_w=120,
        l_k=1,
        B_k=0,
        radius_w=1,
        radius_k=0,
        device="cpu",
    )
    state_a = State.initialize(params, seed=1)
    state_b = State.initialize(params, seed=2)

    assert int(state_a.W.sum().item()) == params.B_w
    assert int(state_b.W.sum().item()) == params.B_w
    assert int(state_a.W.max().item()) <= params.l_w
    assert int(state_b.W.max().item()) <= params.l_w
    assert int(state_a.W.min().item()) >= 0
    assert int(state_b.W.min().item()) >= 0
    assert torch.any(state_a.W != state_b.W)


@pytest.mark.cuda
def test_null_ep_cuda_smoke():
    if not torch.cuda.is_available():
        return

    params = Params(
        shape=(4, 4),
        layers=2,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        B_w=2,
        l_k=3,
        B_k=2,
        radius_k=2,
        device="cuda",
    )

    summary = run_null(
        params,
        seed=7,
        steps=2000,
        report_every=2000,
        device="cuda",
    )
    assert torch.isfinite(torch.tensor(summary["epExactRateWindowLast"]))
