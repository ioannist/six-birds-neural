import math

import pytest
import torch

from ratchet_gpu.kernels import (
    k_local_exchange,
    n_flip,
    s_step,
    spin_flip_color,
    w_local_exchange,
    w_neighbor_exchange,
)
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim
from ratchet_gpu.state import State


def test_null_invariants_after_steps():
    params = Params(
        shape=(4, 4),
        layers=2,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=2,
        B_k=2,
        device="cpu",
    )
    state = State.initialize(params, seed=1)
    gen = torch.Generator(device=state.device)
    gen.manual_seed(7)

    kernels = [
        lambda: spin_flip_color(state, 0, gen),
        lambda: spin_flip_color(state, 1, gen),
        lambda: n_flip(state, gen),
        lambda: s_step(state, gen),
        lambda: w_local_exchange(state, gen),
        lambda: k_local_exchange(state, gen),
        lambda: w_neighbor_exchange(state, gen),
    ]

    for _ in range(2000):
        kernels[int(torch.randint(0, len(kernels), (1,), generator=gen))]()

    state.check_invariants()


def test_null_ep_micro_near_zero():
    params = Params(
        shape=(4, 4),
        layers=2,
        p3_on=False,
        p6_on=False,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=2,
        B_k=2,
        device="cpu",
        report_every=5000,
    )
    summary = run_sim(params, seed=3, steps=50000, report_every=50000)
    assert abs(summary["epMicroRateWindowLast"]) <= 2e-3


def test_p6_micro_ep_positive():
    params = Params(
        shape=(4, 4),
        layers=2,
        p3_on=False,
        p6_on=True,
        beta=1.0,
        eta=0.2,
        eta_drive=0.6,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=2,
        B_k=2,
        device="cpu",
        report_every=5000,
    )
    summary = run_sim(params, seed=4, steps=50000, report_every=50000)
    assert summary["epMicroRateWindowLast"] > 1e-4


def test_p3_strobe_ep_positive():
    params = Params(
        shape=(4, 4),
        layers=2,
        p3_on=True,
        p6_on=False,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=2,
        B_k=2,
        device="cpu",
        report_every=1000,
    )
    summary = run_sim(params, seed=5, steps=35000, report_every=35000)
    assert summary["epStrobeRate"] > 0.0


@pytest.mark.cuda
def test_cpu_cuda_summary_close():
    if not torch.cuda.is_available():
        return

    params = Params(
        shape=(4, 4),
        layers=2,
        p3_on=False,
        p6_on=False,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=2,
        B_k=2,
        device="cpu",
        report_every=2000,
    )
    cpu_summary = run_sim(params, seed=6, steps=10000, report_every=10000)
    cuda_summary = run_sim(params, seed=6, steps=10000, report_every=10000, device="cuda")

    diff = abs(cpu_summary["acceptedFrac"] - cuda_summary["acceptedFrac"])
    assert diff < 0.2
    assert math.isfinite(cuda_summary["epMicroRateWindowLast"])
