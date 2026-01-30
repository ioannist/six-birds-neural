import pytest

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


@pytest.mark.contract
def test_contract_null_ep():
    params = Params(
        shape=(6, 6),
        layers=3,
        p3_on=False,
        p6_on=False,
        beta=1.0,
        eta=0.2,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=4,
        B_k=2,
        kernel_weights={
            "k_local": 1.0,
            "k_neighbor_trade": 1.0,
        },
        report_every=50000,
    )

    for seed in (1, 2):
        summary = run_sim(params, seed=seed, steps=50000, report_every=50000)
        assert abs(summary["epMicroRateWindowLast"]) <= 2e-4
