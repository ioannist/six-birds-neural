#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from ratchet_gpu.lattice import Lattice, generate_stencil
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
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
}


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [int(item) for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must be comma-separated ints")
    return tuple(parts)


def _ci_half(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, float("inf")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, float("inf")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    df = n - 1
    tcrit = _T_CRIT_95.get(df, 1.96)
    return mean, std, tcrit * se


def _kernel_weights(w_neighbor_weight: float) -> dict[str, float]:
    weights = {name: 0.0 for name in Params(shape=(1,), layers=2).kernel_weights}
    weights["spin_flip_color0"] = 1.0
    weights["spin_flip_color1"] = 1.0
    weights["w_local"] = 1.0
    weights["w_neighbor"] = w_neighbor_weight
    return weights


def _build_params(
    shape: tuple[int, ...],
    layers: int,
    beta: float,
    J: float,
    radius_w: int,
    w_fill: float,
    l_w: int,
    w_neighbor_weight: float,
    device: str,
) -> tuple[Params, int, int]:
    lattice = Lattice(shape)
    R_W = generate_stencil(
        d=lattice.d,
        policy="l1_ball_odd",
        radius=radius_w,
        bipartite=True,
        shape=shape,
    )
    K_W = int(R_W.shape[0])
    capacity = l_w * layers * lattice.N * K_W
    B_w = int(round(w_fill * capacity))

    params = Params(
        shape=shape,
        layers=layers,
        p3_on=False,
        p6_on=False,
        beta=beta,
        J=J,
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
        kernel_weights=_kernel_weights(w_neighbor_weight),
        report_every=1000,
        device=device,
    )
    return params, K_W, B_w


def _run_early_stop(
    params: Params,
    seed: int,
    burn_in_steps: int,
    window_steps: int,
    min_windows: int,
    max_windows: int,
    last_m: int,
    mean_thresh: float,
    ci_thresh: float,
    out_path: Path,
) -> dict[str, Any]:
    prev_total = 0.0
    prev_step = 0
    window_rates: list[float] = []
    window_accepts: list[float] = []
    best_ci_half = float("inf")
    stop_flag = {"value": False}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", encoding="utf-8")

    def _report_callback(state, step, ep_ledger, accepted_frac):
        nonlocal prev_total, prev_step, best_ci_half
        total = float(ep_ledger.get("ep_total_exact", 0.0))
        delta_steps = step - prev_step if prev_step else step
        rate = (total - prev_total) / max(1, delta_steps)
        prev_total = total
        prev_step = step
        if step <= burn_in_steps:
            return

        window_rates.append(rate)
        accept_window = float(ep_ledger.get("window_accept_frac", accepted_frac))
        window_accepts.append(accept_window)

        tail = window_rates[-last_m:]
        mean_ep, std_ep, ci_half = _ci_half(tail)
        best_ci_half = min(best_ci_half, ci_half)

        pass_now = (
            len(window_rates) >= min_windows
            and abs(mean_ep) <= mean_thresh
            and ci_half <= ci_thresh
            and 0.10 <= accept_window <= 0.85
        )
        if pass_now:
            stop_flag["value"] = True

        record = {
            "step": int(step),
            "window_index": len(window_rates),
            "ep_rate_exact_window": rate,
            "mean_ep_last_m": mean_ep,
            "ci_half": ci_half,
            "acceptedFracWindow": accept_window,
            "pass": pass_now,
        }
        handle.write(json.dumps(record) + "\n")

    def _stop_callback(state, step, ep_ledger, accepted_frac):
        return stop_flag["value"]

    max_steps = burn_in_steps + max_windows * window_steps
    run_sim(
        params,
        seed=seed,
        steps=max_steps,
        report_every=window_steps,
        device=params.device,
        report_callback=_report_callback,
        stop_callback=_stop_callback,
    )
    handle.close()

    tail = window_rates[-last_m:] if window_rates else []
    mean_ep, std_ep, ci_half = _ci_half(tail)
    accepted_last = window_accepts[-1] if window_accepts else 0.0
    pass_final = (
        len(window_rates) >= min_windows
        and abs(mean_ep) <= mean_thresh
        and ci_half <= ci_thresh
        and 0.10 <= accepted_last <= 0.85
    )

    return {
        "pass": pass_final,
        "windows_used": len(window_rates),
        "mean_ep": mean_ep,
        "ci_half": ci_half,
        "acceptedFracWindow": accepted_last,
        "best_ci_half": best_ci_half,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 W-only convergence (early stop)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--shape", type=_parse_shape, default=(24, 24))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--radius-w", type=int, default=3)
    parser.add_argument("--w-fill", type=float, default=0.05)
    parser.add_argument("--l-w", type=int, default=4)
    parser.add_argument("--w-neighbor-weight", type=float, default=1.0)
    parser.add_argument("--burn-in-sweeps", type=float, default=200.0)
    parser.add_argument("--window-sweeps", type=float, default=100.0)
    parser.add_argument("--min-windows", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--out-dir", default=".tmp/phase1_null_v4")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    N = math.prod(args.shape)
    burn_in_steps = int(args.burn_in_sweeps * N)
    window_steps = int(args.window_sweeps * N)

    params, K_W, B_w = _build_params(
        shape=args.shape,
        layers=args.layers,
        beta=args.beta,
        J=args.J,
        radius_w=args.radius_w,
        w_fill=args.w_fill,
        l_w=args.l_w,
        w_neighbor_weight=args.w_neighbor_weight,
        device=args.device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "convergence_summary.csv"

    fields = [
        "seed",
        "shape",
        "layers",
        "beta",
        "J",
        "radius_w",
        "K_W",
        "l_w",
        "B_w",
        "w_fill",
        "w_neighbor_weight",
        "burn_in_sweeps",
        "window_sweeps",
        "min_windows",
        "max_windows",
        "last_m",
        "windows_used",
        "mean_ep",
        "ci_half",
        "acceptedFracWindow",
        "best_ci_half",
        "pass",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            out_path = out_dir / f"convergence_seed{seed}.jsonl"
            result = _run_early_stop(
                params=params,
                seed=seed,
                burn_in_steps=burn_in_steps,
                window_steps=window_steps,
                min_windows=args.min_windows,
                max_windows=args.max_windows,
                last_m=args.last_m,
                mean_thresh=3e-4,
                ci_thresh=8e-4,
                out_path=out_path,
            )
            writer.writerow(
                {
                    "seed": seed,
                    "shape": str(args.shape),
                    "layers": args.layers,
                    "beta": args.beta,
                    "J": args.J,
                    "radius_w": args.radius_w,
                    "K_W": K_W,
                    "l_w": args.l_w,
                    "B_w": B_w,
                    "w_fill": args.w_fill,
                    "w_neighbor_weight": args.w_neighbor_weight,
                    "burn_in_sweeps": args.burn_in_sweeps,
                    "window_sweeps": args.window_sweeps,
                    "min_windows": args.min_windows,
                    "max_windows": args.max_windows,
                    "last_m": args.last_m,
                    "windows_used": result["windows_used"],
                    "mean_ep": result["mean_ep"],
                    "ci_half": result["ci_half"],
                    "acceptedFracWindow": result["acceptedFracWindow"],
                    "best_ci_half": result["best_ci_half"],
                    "pass": str(result["pass"]).lower(),
                }
            )


if __name__ == "__main__":
    main()
