#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim, _cycle_list

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
try:
    from phase1_null_screen_v4 import _expected_proposals_per_step  # type: ignore
except Exception:  # pragma: no cover
    def _expected_proposals_per_step(N: int, device: str, kernel_weights: Dict[str, float]) -> float:
        return float(N)


K_KERNELS = ("k_local", "k_neighbor_trade", "k_p5_exchange")
HEAVY_KEYS = {
    "strobe_current_map_items_window",
    "strobe_currents_window",
    "strobe_top_states_window",
}


def _load_preset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with path.open() as handle:
        return json.load(handle)


def _as_params(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Params:
    data = {k: v for k, v in preset.items() if k not in {"config_id", "pass", "note"}}
    data.update(overrides)
    if isinstance(data.get("shape"), list):
        data["shape"] = tuple(data["shape"])
    if isinstance(data.get("kernel_weights"), dict):
        data["kernel_weights"] = dict(data["kernel_weights"])
    data.pop("w_neighbor_weight", None)
    return Params(**data)


def _slim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    slim = dict(snapshot)
    for key in HEAVY_KEYS:
        slim.pop(key, None)
    return slim


def _effective_min_strobe_transitions(requested: int, window_steps: int, cycle_len: int) -> int:
    if requested <= 0:
        return 0
    if cycle_len <= 0:
        cycle_len = 1
    max_obs = window_steps // cycle_len
    max_transitions = max(0, max_obs - 1)
    return min(requested, max_transitions)


def _mean_ci(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) < 2:
        return mean_val, 0.0
    var = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
    ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(values))
    return mean_val, ci_half


def _linear_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _k_drive_rate(snapshot: Dict[str, Any]) -> float:
    rates = snapshot.get("ep_rate_by_kernel_proposal_window", {})
    total = 0.0
    for key in K_KERNELS:
        total += float(rates.get(key, 0.0))
    return total


def build_case_params(base: Params, case: str, eta_override: float | None) -> Params:
    overrides: Dict[str, Any] = {
        "strobe_on": True,
    }
    if eta_override is not None:
        overrides["eta"] = eta_override
    if case == "null_full":
        overrides.update({"p3_on": False, "p6_on": False, "eta_drive": 0.0})
    elif case == "p6_drive":
        overrides.update({"p3_on": False, "p6_on": True})
    elif case == "p3p6_combo":
        overrides.update({"p3_on": True, "p6_on": True})
    else:
        raise ValueError(f"Unknown case: {case}")
    return Params.from_dict(base, overrides)


def run_case(
    case: str,
    params: Params,
    seed: int,
    out_dir: Path,
    burn_sweeps: float,
    window_sweeps: float,
    min_windows: int,
    max_windows: int,
    last_m: int,
    accept_min: float,
    min_strobe_transitions: int,
    min_strobe_unique: int,
    min_strobe_bidir: int,
    start_total: float,
    max_seconds_total: float,
    max_seconds_per_run: float,
    cycle: List[str],
) -> Dict[str, Any]:
    N = math.prod(params.shape)
    expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected_props))
    window_steps = int(math.ceil(window_sweeps * N / expected_props))
    cycle_len = len(cycle) if cycle else 1
    min_transitions_used = _effective_min_strobe_transitions(
        min_strobe_transitions, window_steps, cycle_len
    )

    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.csv"
    summary_path = case_dir / "summary.csv"
    progress_path = case_dir / "progress.csv"
    params_path = case_dir / "effective_params.json"

    params_payload = dict(params.__dict__)
    params_payload["shape"] = list(params_payload.get("shape", ()))
    params_payload["device"] = str(params_payload.get("device", ""))
    params_path.write_text(json.dumps(params_payload, indent=2, sort_keys=True))

    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,step,window_index,ep_rate,accept_window,k_drive,strobe_l2,"
                "mismatch_abs_mean,k_entropy_mean,k_r2_mean,k_coh_mean,strobe_transitions,"
                "strobe_unique,strobe_edges,min_strobe_transitions_used\n"
            )

    ep_rates: List[float] = []
    accept_rates: List[float] = []
    k_drive_rates: List[float] = []
    strobe_l2_rates: List[float] = []
    status = "RUNNING"
    accept_low = 0
    run_start = time.monotonic()

    jsonl_path = jsonl_dir / f"seed{seed}.jsonl"
    jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    progress_handle = progress_path.open("a", encoding="utf-8")
    raw_rows: List[Dict[str, Any]] = []

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal status, accept_low
        if status != "RUNNING" or len(ep_rates) >= max_windows:
            return
        snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, None)
        diag_state  # unused, compute_snapshot already updated prev state internally
        window_props = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
        accept_window = 0.0
        if window_props:
            accept_window = float(ep_ledger.get("window_accepted", 0)) / window_props
        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        k_drive = _k_drive_rate(snapshot)
        strobe_l2 = (
            float(snapshot.get("strobe_current_l2_window", 0.0))
            if params.p3_on
            else None
        )
        mismatch_abs = snapshot.get("mismatch_abs_mean")
        k_entropy = snapshot.get("k_entropy_mean")
        k_r2 = snapshot.get("k_r2_mean")
        k_coh = snapshot.get("k_coh_mean")
        transitions = int(snapshot.get("strobe_transitions_window", 0))
        uniq = int(snapshot.get("strobe_unique_states_window", 0))
        edges = int(snapshot.get("strobe_bidirectional_edges_window", 0))

        slim = _slim_snapshot(snapshot)
        slim["acceptedFracWindow"] = accept_window
        slim["k_drive_ep_window"] = k_drive
        if not params.p3_on:
            slim["strobe_current_l2_window"] = None
        slim["min_strobe_transitions_used"] = min_transitions_used
        jsonl_handle.write(to_json_line(slim) + "\n")
        jsonl_handle.flush()

        progress_handle.write(
            f"{case},{seed},{step},{len(ep_rates)+1},{ep_rate},{accept_window},{k_drive},"
            f"{'' if strobe_l2 is None else strobe_l2},{mismatch_abs},{k_entropy},{k_r2},{k_coh},"
            f"{transitions},{uniq},{edges},{min_transitions_used}\n"
        )
        progress_handle.flush()
        os.fsync(progress_handle.fileno())

        if step <= burn_steps:
            return

        if not math.isfinite(ep_rate) or not math.isfinite(accept_window) or not math.isfinite(k_drive):
            status = "FAIL_NUMERIC"
            return

        ep_rates.append(ep_rate)
        accept_rates.append(accept_window)
        k_drive_rates.append(k_drive)
        if params.p3_on and strobe_l2 is not None:
            strobe_l2_rates.append(float(strobe_l2))

        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "window_index": len(ep_rates),
                "step": step,
                "ep_rate_exact_window": ep_rate,
                "acceptedFracWindow": accept_window,
                "k_drive_ep_window": k_drive,
                "strobe_current_l2_window": strobe_l2,
                "mismatch_abs_mean": mismatch_abs,
                "k_entropy_mean": k_entropy,
                "k_r2_mean": k_r2,
                "k_coh_mean": k_coh,
                "strobe_transitions_window": transitions,
                "strobe_unique_states_window": uniq,
                "strobe_bidirectional_edges_window": edges,
                "min_strobe_transitions_used": min_transitions_used,
            }
        )

        if accept_window < accept_min:
            accept_low += 1
        else:
            accept_low = 0
        if accept_low >= 5:
            status = "FAIL_ACCEPT_COLLAPSE"
            return

        if uniq < min_strobe_unique or edges < min_strobe_bidir or transitions < min_transitions_used:
            status = "FAIL_STROBE_SPARSE"
            return

        if (
            time.monotonic() - run_start > max_seconds_per_run
            or time.monotonic() - start_total > max_seconds_total
        ):
            status = "FAIL_TIME"

    def stop_cb(state, step, ep_ledger, accepted_frac):
        return status != "RUNNING" or len(ep_rates) >= max_windows

    max_steps = burn_steps + max_windows * window_steps
    run_sim(
        params,
        seed=seed,
        steps=max_steps,
        report_every=window_steps,
        device=params.device,
        report_callback=report_cb,
        stop_callback=stop_cb,
        protocol_cycle=cycle,
    )
    jsonl_handle.close()
    progress_handle.close()

    runtime = time.monotonic() - run_start
    if status == "RUNNING":
        status = "FAIL_TIME" if len(ep_rates) < min_windows else "OK"

    tail_ep = ep_rates[-last_m:] if ep_rates else []
    tail_acc = accept_rates[-last_m:] if accept_rates else []
    tail_k = k_drive_rates[-last_m:] if k_drive_rates else []
    tail_l2 = strobe_l2_rates[-last_m:] if strobe_l2_rates else []

    ep_mean, ep_ci = _mean_ci(tail_ep)
    acc_mean, _ = _mean_ci(tail_acc)
    k_mean, _ = _mean_ci(tail_k)
    l2_mean, _ = _mean_ci(tail_l2)
    ep_slope = _linear_slope(tail_ep)
    k_slope = _linear_slope(tail_k) if params.p6_on else 0.0
    l2_slope = _linear_slope(tail_l2) if params.p3_on else 0.0

    with raw_path.open("w", encoding="utf-8", newline="") as rh:
        writer = csv.DictWriter(rh, fieldnames=list(raw_rows[0].keys()) if raw_rows else [])
        if raw_rows:
            writer.writeheader()
            writer.writerows(raw_rows)

    with summary_path.open("w", encoding="utf-8", newline="") as sh:
        fieldnames = [
            "case",
            "status",
            "windows_completed",
            "runtime_sec",
            "accept_mean_last_m",
            "ep_mean_last_m",
            "ep_ci_half_last_m",
            "ep_slope_last_m",
            "k_drive_mean_last_m",
            "k_drive_slope_last_m",
            "strobe_l2_mean_last_m",
            "strobe_l2_slope_last_m",
        ]
        writer = csv.DictWriter(sh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "case": case,
                "status": status,
                "windows_completed": len(ep_rates),
                "runtime_sec": runtime,
                "accept_mean_last_m": acc_mean,
                "ep_mean_last_m": ep_mean,
                "ep_ci_half_last_m": ep_ci,
                "ep_slope_last_m": ep_slope,
                "k_drive_mean_last_m": k_mean if params.p6_on else None,
                "k_drive_slope_last_m": k_slope if params.p6_on else None,
                "strobe_l2_mean_last_m": l2_mean if params.p3_on else None,
                "strobe_l2_slope_last_m": l2_slope if params.p3_on else None,
            }
        )

    return {
        "case": case,
        "status": status,
        "windows_completed": len(ep_rates),
        "runtime_sec": runtime,
        "accept_mean_last_m": acc_mean,
        "ep_mean_last_m": ep_mean,
        "ep_ci_half_last_m": ep_ci,
        "ep_slope_last_m": ep_slope,
        "k_drive_mean_last_m": k_mean if params.p6_on else None,
        "k_drive_slope_last_m": k_slope if params.p6_on else None,
        "strobe_l2_mean_last_m": l2_mean if params.p3_on else None,
        "strobe_l2_slope_last_m": l2_slope if params.p3_on else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 long-run stability v1")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", default="scripts/params/phase5_p3p6_combo_balanced_v1.json")
    parser.add_argument("--out-dir", default=".tmp/phase6_longrun_v1")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--min-windows", type=int, default=120)
    parser.add_argument("--max-windows", type=int, default=200)
    parser.add_argument("--last-m", type=int, default=20)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--max-seconds-total", type=float, default=6600)
    parser.add_argument("--max-seconds-per-run", type=float, default=2100)
    parser.add_argument("--min-strobe-transitions", type=int, default=200)
    parser.add_argument("--min-strobe-unique", type=int, default=3)
    parser.add_argument("--min-strobe-bidirectional-edges", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preset = _load_preset(Path(args.preset))
    base = _as_params(preset, {"device": args.device})
    if args.eta is not None:
        base = Params.from_dict(base, {"eta": args.eta})

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle = _cycle_list()
    start_total = time.monotonic()
    summaries: List[Dict[str, Any]] = []
    for case in ("null_full", "p6_drive", "p3p6_combo"):
        for seed in seeds:
            if time.monotonic() - start_total > args.max_seconds_total:
                break
            params = build_case_params(base, case, args.eta)
            summary = run_case(
                case=case,
                params=params,
                seed=seed,
                out_dir=out_dir,
                burn_sweeps=args.burn_in_sweeps,
                window_sweeps=args.window_sweeps,
                min_windows=args.min_windows,
                max_windows=args.max_windows,
                last_m=args.last_m,
                accept_min=args.accept_min,
                min_strobe_transitions=args.min_strobe_transitions,
                min_strobe_unique=args.min_strobe_unique,
                min_strobe_bidir=args.min_strobe_bidirectional_edges,
                start_total=start_total,
                max_seconds_total=args.max_seconds_total,
                max_seconds_per_run=args.max_seconds_per_run,
                cycle=cycle,
            )
            summaries.append(summary)
            if summary["status"] == "FAIL_TIME" and summary["windows_completed"] < args.min_windows:
                break

    agg_path = out_dir / "agg.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        fieldnames = [
            "case",
            "status",
            "windows_completed",
            "accept_mean_last_m",
            "ep_mean_last_m",
            "ep_ci_half_last_m",
            "ep_slope_last_m",
            "k_drive_mean_last_m",
            "k_drive_slope_last_m",
            "strobe_l2_mean_last_m",
            "strobe_l2_slope_last_m",
        ]
        writer = csv.DictWriter(ah, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k) for k in fieldnames})

    report_path = out_dir / "PHASE6_LONGRUN_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 6 long-run stability v1\n\n")
        fh.write(f"preset: {args.preset}\n")
        fh.write(f"eta: {base.eta}\n\n")
        fh.write("| case | status | windows | accept_mean | ep_mean | ep_ci_half | ep_slope | ")
        fh.write("k_drive_mean | k_drive_slope | strobe_l2_mean | strobe_l2_slope |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries:
            fh.write(
                f"| {row.get('case','')} | {row.get('status','')} | {row.get('windows_completed',0)} | "
                f"{row.get('accept_mean_last_m',0.0):.6g} | {row.get('ep_mean_last_m',0.0):.6g} | "
                f"{row.get('ep_ci_half_last_m',0.0):.6g} | {row.get('ep_slope_last_m',0.0):.6g} | "
                f"{row.get('k_drive_mean_last_m','')} | {row.get('k_drive_slope_last_m','')} | "
                f"{row.get('strobe_l2_mean_last_m','')} | {row.get('strobe_l2_slope_last_m','')} |\n"
            )


if __name__ == "__main__":
    main()
