#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
}


def _ci_half(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, float("inf")
    mean = sum(values) / n
    if n == 1:
        return mean, float("inf")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    df = n - 1
    tcrit = _T_CRIT_95.get(df, 1.96)
    return mean, tcrit * se


def _load_params(path: Path) -> Params:
    data = json.loads(path.read_text(encoding="utf-8"))
    params_keys = {
        "shape",
        "layers",
        "p3_on",
        "p6_on",
        "beta",
        "J",
        "kappa_T",
        "eta",
        "eta_drive",
        "l_s",
        "l_w",
        "l_k",
        "B_w",
        "B_k",
        "stencil_policy_w",
        "stencil_policy_k",
        "radius_w",
        "radius_k",
        "include_zero_k",
        "kernel_weights",
        "report_every",
        "device",
    }
    kwargs = {k: v for k, v in data.items() if k in params_keys}
    return Params(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1 preset (CPU)")
    parser.add_argument(
        "--preset",
        default="scripts/params/phase1_null_balanced_v3.json",
    )
    parser.add_argument("--shape", default="", help="override shape (e.g., 24,24)")
    parser.add_argument("--burn-sweeps", type=float, default=200.0)
    parser.add_argument("--measure-sweeps", type=float, default=100.0)
    parser.add_argument("--window-sweeps", type=float, default=50.0)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--seeds", default="1,2,3")

    args = parser.parse_args()

    params = _load_params(Path(args.preset))
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(",") if x.strip())
        params = Params.from_dict(params, {"shape": shape})

    params = Params.from_dict(params, {"device": "cpu"})
    N = math.prod(params.shape)
    burn_steps = int(args.burn_sweeps * N)
    measure_steps = int(args.measure_sweeps * N)
    report_every = max(1, int(args.window_sweeps * N))
    steps = burn_steps + measure_steps

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("seed,mean_ep_lastM,ci_half,acceptedFrac,pass")
    all_pass = True

    for seed in seeds:
        ep_rates = []

        def _report_callback(state, step, ep_ledger, accepted_frac):
            if step > burn_steps:
                ep_rates.append(float(ep_ledger.get("ep_total_exact", 0.0)))

        summary = run_sim(
            params,
            seed=seed,
            steps=steps,
            report_every=report_every,
            device="cpu",
            report_callback=_report_callback,
        )

        # convert totals to windowed rates
        window_rates = []
        if len(ep_rates) >= 2:
            prev = ep_rates[0]
            for total in ep_rates[1:]:
                window_rates.append((total - prev) / report_every)
                prev = total

        tail = window_rates[-args.last_m :] if window_rates else []
        mean_ep, ci_half = _ci_half(tail)
        accepted = summary.get("acceptedFrac", 0.0)
        pass_seed = (
            abs(mean_ep) <= 2e-4
            and ci_half <= 5e-4
            and 0.10 <= accepted <= 0.85
        )
        if not pass_seed:
            all_pass = False
        print(f"{seed},{mean_ep:.6f},{ci_half:.6f},{accepted:.4f},{str(pass_seed).lower()}")

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
