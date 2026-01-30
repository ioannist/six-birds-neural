import pytest
import torch

from ratchet_gpu.lattice import Lattice, generate_stencil
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


@pytest.mark.cuda
def test_contract_cuda_w_budget():
    if not torch.cuda.is_available():
        return

    shape = (12, 12)
    layers = 2
    radius_w = 3
    l_w = 4
    w_fill = 0.05

    lattice = Lattice(shape)
    offsets = generate_stencil(
        d=lattice.d,
        policy="l1_ball_odd",
        radius=radius_w,
        bipartite=True,
        shape=shape,
    )
    K_W = int(offsets.shape[0])
    B_w = int(round(w_fill * l_w * layers * lattice.N * K_W))

    base_params = Params(shape=shape, layers=layers)
    kernel_weights = {name: 0.0 for name in base_params.kernel_weights}
    kernel_weights["spin_flip_color0"] = 1.0
    kernel_weights["spin_flip_color1"] = 1.0
    kernel_weights["w_local"] = 1.0
    kernel_weights["w_neighbor"] = 1.0

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
        kernel_weights=kernel_weights,
        report_every=10000,
        device="cuda",
    )

    holder = {}

    def _capture(state, step, ep_ledger, accepted_frac):
        holder["state"] = state

    run_sim(
        params,
        seed=7,
        steps=50000,
        report_every=10000,
        device="cuda",
        report_callback=_capture,
    )

    state = holder.get("state")
    assert state is not None
    assert int(state.W.sum().item()) == params.B_w
    assert int(state.W.min().item()) >= 0
    assert int(state.W.max().item()) <= params.l_w
