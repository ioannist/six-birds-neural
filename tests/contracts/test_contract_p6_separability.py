import pytest

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


@pytest.mark.contract
def test_contract_p6_separability():
    params = Params(
        shape=(6, 6),
        layers=3,
        p3_on=False,
        p6_on=True,
        beta=1.0,
        eta=0.0,
        eta_drive=0.8,
        J=1.0,
        l_w=3,
        l_k=3,
        B_w=4,
        B_k=2,
        kernel_weights={
            "k_local": 1.0,
            "n_flip": 1.0,
        },
        report_every=50000,
    )

    summary = run_sim(params, seed=3, steps=50000, report_every=50000)
    assert summary["epMicroRateWindowLast"] >= 1e-4
    assert summary["epMicroRateWindowLast_k_local"] > 0.0
