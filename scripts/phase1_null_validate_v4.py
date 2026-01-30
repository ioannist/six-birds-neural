#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, TextIO

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


def _load_params(path: Path, device: str) -> Params:
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
    }
    kwargs = {k: v for k, v in data.items() if k in params_keys}
    kwargs["device"] = device
    return Params(**kwargs)


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
    progress_handle: TextIO | None,
    run_start_time: float,
    max_seconds_per_run: float,
) -> dict[str, Any]:
    prev_total = 0.0
    prev_step = 0
    window_rates: list[float] = []
    window_accepts: list[float] = []
    best_ci_half = float("inf")
    stop_flag = {"value": False}
    status = "FAIL_MAX_WINDOWS"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", encoding="utf-8")

    def _report_callback(state, step, ep_ledger, accepted_frac):
        nonlocal prev_total, prev_step, best_ci_half, status
        if time.monotonic() - run_start_time > max_seconds_per_run:
            status = "FAIL_BUDGET"
            stop_flag["value"] = True
            return
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
            status = "PASS_EARLY"

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
        handle.flush()
        if progress_handle is not None:
            progress_row = {
                "timestamp": time.time(),
                "seed": seed,
                "step": int(step),
                "window_index": len(window_rates),
                "ep_rate_exact_window": rate,
                "mean_ep_last_m": mean_ep,
                "ci_half": ci_half,
                "acceptedFracWindow": accept_window,
                "pass": pass_now,
            }
            progress_handle.write(
                ",".join(str(progress_row[k]) for k in progress_row) + "\n"
            )
            progress_handle.flush()
            os.fsync(progress_handle.fileno())

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
    if status == "FAIL_MAX_WINDOWS" and pass_final:
        status = "PASS_EARLY"

    return {
        "pass": pass_final,
        "windows_used": len(window_rates),
        "mean_ep": mean_ep,
        "ci_half": ci_half,
        "acceptedFracWindow": accepted_last,
        "best_ci_half": best_ci_half,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate phase1 v4 preset with early stop")
    parser.add_argument(
        "--preset",
        default="scripts/params/phase1_null_balanced_v4.json",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--burn-in-sweeps", type=float, default=200.0)
    parser.add_argument("--window-sweeps", type=float, default=100.0)
    parser.add_argument("--min-windows", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=30)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--mean-thresh", type=float, default=3e-4)
    parser.add_argument("--ci-thresh", type=float, default=8e-4)
    parser.add_argument("--max-seconds-total", type=float, default=5400.0)
    parser.add_argument("--max-seconds-per-run", type=float, default=1200.0)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--out-dir", default=".tmp/phase1_null_v4")
    parser.add_argument(
        "--progress-csv",
        default="",
        help="Path to append window progress (default: <out-dir>/validate_progress.csv)",
    )
    args = parser.parse_args()

    params = _load_params(Path(args.preset), args.device)
    N = math.prod(params.shape)
    burn_in_steps = int(args.burn_in_sweeps * N)
    window_steps = int(args.window_sweeps * N)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_path = out_dir / "validate.csv"
    progress_path = (
        Path(args.progress_csv)
        if args.progress_csv
        else out_dir / "validate_progress.csv"
    )
    start_time = time.monotonic()

    fields = [
        "seed",
        "windows_used",
        "mean_ep",
        "ci_half",
        "acceptedFracWindow",
        "best_ci_half",
        "seconds_used",
        "status",
        "pass",
    ]

    with validate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        for seed in seeds:
            if time.monotonic() - start_time > args.max_seconds_total:
                print("Total time budget exceeded; stopping.")
                break
            out_path = out_dir / f"validate_seed{seed}.jsonl"
            progress_handle = progress_path.open("a", encoding="utf-8")
            if progress_path.stat().st_size == 0:
                progress_handle.write(
                    "timestamp,seed,step,window_index,ep_rate_exact_window,"
                    "mean_ep_last_m,ci_half,acceptedFracWindow,pass\n"
                )
                progress_handle.flush()
                os.fsync(progress_handle.fileno())
            run_start = time.monotonic()
            try:
                result = _run_early_stop(
                    params=params,
                    seed=seed,
                    burn_in_steps=burn_in_steps,
                    window_steps=window_steps,
                    min_windows=args.min_windows,
                    max_windows=args.max_windows,
                    last_m=args.last_m,
                    mean_thresh=args.mean_thresh,
                    ci_thresh=args.ci_thresh,
                    out_path=out_path,
                    progress_handle=progress_handle,
                    run_start_time=run_start,
                    max_seconds_per_run=args.max_seconds_per_run,
                )
            except Exception:
                result = {
                    "pass": False,
                    "windows_used": 0,
                    "mean_ep": 0.0,
                    "ci_half": float("inf"),
                    "acceptedFracWindow": 0.0,
                    "best_ci_half": float("inf"),
                    "status": "ERROR",
                }
            progress_handle.close()
            seconds_used = time.monotonic() - run_start
            writer.writerow(
                {
                    "seed": seed,
                    "windows_used": result["windows_used"],
                    "mean_ep": result["mean_ep"],
                    "ci_half": result["ci_half"],
                    "acceptedFracWindow": result["acceptedFracWindow"],
                    "best_ci_half": result["best_ci_half"],
                    "seconds_used": round(seconds_used, 3),
                    "status": result["status"],
                    "pass": str(result["pass"]).lower(),
                }
            )
            handle.flush()
            os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
