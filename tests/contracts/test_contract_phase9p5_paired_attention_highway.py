import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase9p5_paired_hazard_baseline_v1 as p9p5

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim, _cycle_list


@pytest.mark.contract
def test_phase9p5_paired_attention_highway(tmp_path: Path) -> None:
    params = Params(
        shape=(10, 10),
        layers=3,
        eta=2.0,
        B_w=300,
        B_k=4,
        l_w=3,
        l_k=3,
        radius_w=1,
        radius_k=2,
        kernel_weights={
            "spin_flip_color0": 1.0,
            "spin_flip_color1": 1.0,
            "w_local": 0.2,
            "k_local": 20.0,
            "k_neighbor_trade": 20.0,
        },
        device="cpu",
    )

    N = math.prod(params.shape)
    expected = p9p5._expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(20 * N / expected))

    burn_summary = run_sim(params, seed=1, steps=burn_steps, report_every=burn_steps, return_state=True)
    state0 = burn_summary.get("state")
    rng_state = burn_summary.get("rng_state")
    assert state0 is not None and rng_state is not None

    cycle = _cycle_list()
    start_total = time.monotonic()
    hazard_rect = "2:8,2:8"
    hazard_layers = [0]
    window_sweeps = 20
    max_windows = 12

    baseline_metrics = p9p5.run_case(
        "baseline",
        params,
        1,
        tmp_path,
        window_sweeps,
        max_windows,
        1,
        ["sigma", "w_mass", "w_entropy", "w_axis_bias", "k_entropy", "k_axis_bias", "mismatch"],
        0.003,
        3,
        4,
        hazard_rect,
        "flip",
        hazard_layers,
        120.0,
        60.0,
        start_total,
        cycle,
        False,
        apply_hazard=False,
        hazard_refresh_each_window=True,
        initial_state=p9p5._clone_state(state0),
        initial_rng_state=rng_state.clone(),
    )

    hazard_metrics = p9p5.run_case(
        "hazard",
        params,
        1,
        tmp_path,
        window_sweeps,
        max_windows,
        1,
        ["sigma", "w_mass", "w_entropy", "w_axis_bias", "k_entropy", "k_axis_bias", "mismatch"],
        0.003,
        3,
        4,
        hazard_rect,
        "flip",
        hazard_layers,
        120.0,
        60.0,
        start_total,
        cycle,
        False,
        apply_hazard=True,
        hazard_refresh_each_window=True,
        initial_state=p9p5._clone_state(state0),
        initial_rng_state=rng_state.clone(),
    )

    paired = p9p5._paired_metrics(
        {
            "mismatch_region": baseline_metrics["mismatch_region"],
            "k_axis_bias_focus": baseline_metrics["k_axis_bias_focus"],
        },
        {
            "mismatch_region": hazard_metrics["mismatch_region"],
            "k_axis_bias_focus": hazard_metrics["k_axis_bias_focus"],
        },
        3,
        4,
        int(hazard_metrics["windows_completed"]),
    )

    assert paired["spike_paired"] >= 0.005
    assert paired["realloc_paired"] >= 0.002

    hazard_raw = tmp_path / "hazard" / "raw.csv"
    values = []
    with hazard_raw.open() as fh:
        header = fh.readline().strip().split(",")
        accept_idx = header.index("accept_window")
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            values.append(float(parts[accept_idx]))
    assert np.mean(values) >= 0.003
