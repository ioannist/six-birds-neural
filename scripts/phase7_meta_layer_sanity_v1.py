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


def _parse_candidates(value: str, label: str) -> List[float]:
    vals = [float(x) for x in value.split(",") if x.strip()]
    if not any(abs(v) < 1e-12 for v in vals):
        raise ValueError(f"{label} must include 0")
    return vals


def _validate_layers(layers: int) -> None:
    if layers < 3:
        raise ValueError("layers must be >= 3 for meta-layer sanity")


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


def build_case_params(base: Params, case: str, layers: int, eta: float, eta_drive: float, p6_on: bool) -> Params:
    overrides = {
        "layers": layers,
        "eta": eta,
        "eta_drive": eta_drive,
        "p3_on": False,
        "p6_on": p6_on,
        "strobe_on": True,
    }
    return Params.from_dict(base, overrides)


def _case_name(prefix: str, value: float) -> str:
    tag = f"{value:.3f}".replace(".", "p")
    return f"{prefix}_{tag}"


def _drive_only_eta(eta_best: float, mode: str) -> float:
    if mode == "eta_best":
        return eta_best
    if mode == "zero":
        return 0.0
    raise ValueError(f"Unsupported drive-only eta mode: {mode}")


def run_case(
    case: str,
    params: Params,
    seeds: List[int],
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
    resume: bool,
) -> List[Dict[str, Any]]:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.csv"
    summary_path = case_dir / "summary.csv"
    progress_path = case_dir / "progress.csv"
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    params_path = case_dir / "effective_params.json"

    params_payload = dict(params.__dict__)
    params_payload["shape"] = list(params_payload.get("shape", ()))
    params_payload["device"] = str(params_payload.get("device", ""))
    params_path.write_text(json.dumps(params_payload, indent=2, sort_keys=True))

    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    completed_seeds: set[int] = set()
    if resume and raw_path.exists():
        with raw_path.open("r", encoding="utf-8", newline="") as rh:
            raw_rows = list(csv.DictReader(rh))
    if resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as sh:
            summary_rows = list(csv.DictReader(sh))
        for row in summary_rows:
            try:
                seed_val = int(row.get("seed", -1))
            except (TypeError, ValueError):
                continue
            if row.get("status") != "RUNNING":
                completed_seeds.add(seed_val)

    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,step,window_index,ep_rate,accept_window,k_drive,mismatch_abs_mean,"
                "strobe_transitions,strobe_unique,strobe_edges,min_strobe_transitions_used\n"
            )

    N = math.prod(params.shape)
    expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected_props))
    window_steps = int(math.ceil(window_sweeps * N / expected_props))
    cycle_len = len(cycle) if cycle else 1
    min_transitions_used = _effective_min_strobe_transitions(
        min_strobe_transitions, window_steps, cycle_len
    )

    for seed in seeds:
        if seed in completed_seeds:
            continue

        ep_rates: List[float] = []
        accept_rates: List[float] = []
        mismatch_vals: List[float] = []
        k_drive_rates: List[float] = []
        status = "RUNNING"
        accept_low = 0
        diag_state = None

        jsonl_path = jsonl_dir / f"seed{seed}.jsonl"
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        progress_handle = progress_path.open("a", encoding="utf-8")
        run_start = time.monotonic()

        def report_cb(state, step, ep_ledger, accepted_frac):
            nonlocal status, accept_low, diag_state
            if status != "RUNNING" or len(ep_rates) >= max_windows:
                return
            snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
            window_props = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
            accept_window = 0.0
            if window_props:
                accept_window = float(ep_ledger.get("window_accepted", 0)) / window_props
            ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
            mismatch_abs = snapshot.get("mismatch_abs_mean")
            k_drive = _k_drive_rate(snapshot)
            transitions = int(snapshot.get("strobe_transitions_window", 0))
            uniq = int(snapshot.get("strobe_unique_states_window", 0))
            edges = int(snapshot.get("strobe_bidirectional_edges_window", 0))

            slim = _slim_snapshot(snapshot)
            slim["acceptedFracWindow"] = accept_window
            slim["k_drive_ep_window"] = k_drive
            slim["min_strobe_transitions_used"] = min_transitions_used
            jsonl_handle.write(to_json_line(slim) + "\n")
            jsonl_handle.flush()
            progress_handle.write(
                f"{case},{seed},{step},{len(ep_rates)+1},{ep_rate},{accept_window},{k_drive},"
                f"{mismatch_abs},{transitions},{uniq},{edges},{min_transitions_used}\n"
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
            if mismatch_abs is not None:
                mismatch_vals.append(float(mismatch_abs))

            raw_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "window_index": len(ep_rates),
                    "step": step,
                    "ep_rate_exact_window": ep_rate,
                    "acceptedFracWindow": accept_window,
                    "k_drive_ep_window": k_drive,
                    "mismatch_abs_mean": mismatch_abs,
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
        tail_m = mismatch_vals[-last_m:] if mismatch_vals else []

        ep_mean, ep_ci = _mean_ci(tail_ep)
        acc_mean, _ = _mean_ci(tail_acc)
        k_mean, _ = _mean_ci(tail_k)
        mismatch_mean, _ = _mean_ci(tail_m)
        ep_slope = _linear_slope(tail_ep)
        k_slope = _linear_slope(tail_k)
        mismatch_slope = _linear_slope(tail_m)

        summary_rows.append(
            {
                "case": case,
                "seed": seed,
                "status": status,
                "windows_completed": len(ep_rates),
                "runtime_sec": runtime,
                "accept_mean_last_m": acc_mean,
                "ep_mean_last_m": ep_mean,
                "ep_ci_half_last_m": ep_ci,
                "ep_slope_last_m": ep_slope,
                "mismatch_mean_last_m": mismatch_mean,
                "mismatch_slope_last_m": mismatch_slope,
                "k_drive_mean_last_m": k_mean,
                "k_drive_slope_last_m": k_slope,
            }
        )

    if raw_rows:
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as sh:
            fieldnames = list(summary_rows[0].keys())
            writer = csv.DictWriter(sh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 meta-layer sanity v1")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", default="scripts/params/phase5_p3p6_combo_balanced_v1.json")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--min-windows", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--min-strobe-transitions", type=int, default=200)
    parser.add_argument("--min-strobe-unique", type=int, default=3)
    parser.add_argument("--min-strobe-bidirectional-edges", type=int, default=1)
    parser.add_argument("--eta-candidates", default="0,0.5,1.0")
    parser.add_argument("--eta-drive-candidates", default="0,1.0,2.0")
    parser.add_argument("--drive-only-eta-mode", default="eta_best", choices=["eta_best", "zero"])
    parser.add_argument("--mean-thresh-null", type=float, default=2e-3)
    parser.add_argument("--ci-thresh-null", type=float, default=2e-3)
    parser.add_argument("--mismatch-drop-frac", type=float, default=0.01)
    parser.add_argument("--k-drive-mean-min", type=float, default=1e-3)
    parser.add_argument("--max-seconds-total", type=float, default=6600)
    parser.add_argument("--max-seconds-per-run", type=float, default=900)
    parser.add_argument("--out-dir", default=".tmp/phase7_meta_layer_sanity_v1")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preset-out-dir", default="scripts/params")
    args = parser.parse_args()

    _validate_layers(args.layers)
    eta_candidates = _parse_candidates(args.eta_candidates, "eta-candidates")
    eta_drive_candidates = _parse_candidates(args.eta_drive_candidates, "eta-drive-candidates")

    preset = _load_preset(Path(args.preset))
    base = _as_params(preset, {"device": args.device, "layers": args.layers})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if 1 in seeds:
        seeds = [1] + [s for s in seeds if s != 1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycle = _cycle_list()
    start_total = time.monotonic()

    report_lines: List[str] = []
    screen_rows: List[Dict[str, Any]] = []
    confirm_rows: List[Dict[str, Any]] = []
    screen_seed = 1

    def _pick_seed_row(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
        for row in rows:
            try:
                if int(row.get("seed", -1)) == seed:
                    return row
            except (TypeError, ValueError):
                continue
        if rows:
            return rows[-1]
        raise ValueError(f"No summary rows available for case (seed={seed}).")

    # Equilibrium screen
    eta0_case = _case_name("null_eta", 0.0)
    eta0_params = build_case_params(base, eta0_case, args.layers, eta=0.0, eta_drive=0.0, p6_on=False)
    eta0_summaries = run_case(
        case=eta0_case,
        params=eta0_params,
        seeds=[screen_seed],
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        accept_min=args.accept_min,
        min_strobe_transitions=args.min_strobe_transitions,
        min_strobe_unique=3,
        min_strobe_bidir=1,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        cycle=cycle,
        resume=args.resume,
    )
    eta0_summary = _pick_seed_row(eta0_summaries, screen_seed)
    eta0_summary["eta"] = 0.0
    eta0_summary["mismatch_drop"] = 0.0
    screen_rows.append(eta0_summary)
    mismatch0 = float(eta0_summary.get("mismatch_mean_last_m") or 0.0)
    if eta0_summary.get("status") == "FAIL_TIME":
        report_lines.append("Equilibrium baseline failed (time cap).")
        _write_report(out_dir / "PHASE7_META_LAYER_REPORT.md", report_lines)
        return

    eligible: List[Dict[str, Any]] = []
    for eta in eta_candidates:
        if abs(eta) < 1e-12:
            continue
        case = _case_name("null_eta", eta)
        params = build_case_params(base, case, args.layers, eta=eta, eta_drive=0.0, p6_on=False)
        summaries = run_case(
            case=case,
            params=params,
            seeds=[screen_seed],
            out_dir=out_dir,
            burn_sweeps=args.burn_in_sweeps,
            window_sweeps=args.window_sweeps,
            min_windows=args.min_windows,
            max_windows=args.max_windows,
            last_m=args.last_m,
            accept_min=args.accept_min,
            min_strobe_transitions=args.min_strobe_transitions,
            min_strobe_unique=3,
            min_strobe_bidir=1,
            start_total=start_total,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            cycle=cycle,
            resume=args.resume,
        )
        summary = _pick_seed_row(summaries, screen_seed)
        mismatch = float(summary.get("mismatch_mean_last_m") or 0.0)
        mismatch_drop = (mismatch0 - mismatch) / mismatch0 if mismatch0 > 0 else 0.0
        summary["mismatch_drop"] = mismatch_drop
        summary["eta"] = eta
        screen_rows.append(summary)
        if (
            abs(float(summary.get("ep_mean_last_m", 0.0))) <= args.mean_thresh_null
            and float(summary.get("ep_ci_half_last_m", 0.0)) <= args.ci_thresh_null
            and float(summary.get("accept_mean_last_m", 0.0)) >= args.accept_min
            and mismatch_drop >= args.mismatch_drop_frac
        ):
            eligible.append(summary)

    if not eligible:
        report_lines.append("No eligible eta candidates in equilibrium screen.")
        _write_report(out_dir / "PHASE7_META_LAYER_REPORT.md", report_lines)
        return

    eta_best = max(eligible, key=lambda r: float(r.get("mismatch_drop", 0.0)))
    eta_best_val = float(eta_best["eta"])
    confirm_eta_case = _case_name("null_confirm_eta", eta_best_val)
    confirm_pass = 0
    params = build_case_params(base, confirm_eta_case, args.layers, eta=eta_best_val, eta_drive=0.0, p6_on=False)
    confirm_summaries = run_case(
        case=confirm_eta_case,
        params=params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        accept_min=args.accept_min,
        min_strobe_transitions=args.min_strobe_transitions,
        min_strobe_unique=3,
        min_strobe_bidir=1,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        cycle=cycle,
        resume=args.resume,
    )
    for summary in confirm_summaries:
        try:
            seed_val = int(summary.get("seed", -1))
        except (TypeError, ValueError):
            continue
        if seed_val not in seeds:
            continue
        mismatch = float(summary.get("mismatch_mean_last_m") or 0.0)
        mismatch_drop = (mismatch0 - mismatch) / mismatch0 if mismatch0 > 0 else 0.0
        summary["mismatch_drop"] = mismatch_drop
        summary["eta"] = eta_best_val
        confirm_rows.append(summary)
        if (
            abs(float(summary.get("ep_mean_last_m", 0.0))) <= args.mean_thresh_null
            and float(summary.get("ep_ci_half_last_m", 0.0)) <= args.ci_thresh_null
            and float(summary.get("accept_mean_last_m", 0.0)) >= args.accept_min
            and mismatch_drop >= args.mismatch_drop_frac
        ):
            confirm_pass += 1

    if confirm_pass < 2:
        report_lines.append("Equilibrium confirm failed (<2/3 seeds).")
        _write_report(out_dir / "PHASE7_META_LAYER_REPORT.md", report_lines)
        return

    preset_dir = Path(args.preset_out_dir)
    preset_dir.mkdir(parents=True, exist_ok=True)
    decoupled_path = preset_dir / f"meta_null_decoupled_layers{args.layers}.json"
    coupled_path = preset_dir / f"meta_null_coupled_eta{eta_best_val:.2f}_layers{args.layers}.json"

    def write_preset(path: Path, overrides: Dict[str, Any]) -> None:
        payload = dict(preset)
        payload.update(overrides)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    write_preset(
        decoupled_path,
        {
            "layers": args.layers,
            "p3_on": False,
            "p6_on": False,
            "eta": 0.0,
            "eta_drive": 0.0,
            "config_id": f"phase7_meta_null_decoupled_eta0_l{args.layers}",
        },
    )
    write_preset(
        coupled_path,
        {
            "layers": args.layers,
            "p3_on": False,
            "p6_on": False,
            "eta": eta_best_val,
            "eta_drive": 0.0,
            "config_id": f"phase7_meta_null_coupled_eta{eta_best_val:.2f}_l{args.layers}",
        },
    )

    # Drive-only screen (eta fixed via mode)
    eta_for_drive = _drive_only_eta(eta_best_val, args.drive_only_eta_mode)
    report_lines.append(
        f"DRIVE_ONLY_BASELINE eta={eta_for_drive:.3g} mode={args.drive_only_eta_mode}"
    )
    drive0_case = _case_name("drive_base_eta", eta_for_drive)
    drive0_params = build_case_params(
        base, drive0_case, args.layers, eta=eta_for_drive, eta_drive=0.0, p6_on=False
    )
    drive0_summaries = run_case(
        case=drive0_case,
        params=drive0_params,
        seeds=[screen_seed],
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        accept_min=args.accept_min,
        min_strobe_transitions=args.min_strobe_transitions,
        min_strobe_unique=3,
        min_strobe_bidir=1,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        cycle=cycle,
        resume=args.resume,
    )
    drive0_summary = _pick_seed_row(drive0_summaries, screen_seed)
    drive0_summary["eta_drive"] = 0.0
    drive0_summary["eta"] = eta_for_drive
    drive0_summary["mismatch_drop"] = 0.0
    screen_rows.append(drive0_summary)
    mismatch_drive0 = float(drive0_summary.get("mismatch_mean_last_m") or 0.0)

    eligible_drive: List[Dict[str, Any]] = []
    for eta_drive in eta_drive_candidates:
        if abs(eta_drive) < 1e-12:
            continue
        case = _case_name("drive_eta", eta_drive)
        params = build_case_params(
            base, case, args.layers, eta=eta_for_drive, eta_drive=eta_drive, p6_on=True
        )
        summaries = run_case(
            case=case,
            params=params,
            seeds=[screen_seed],
            out_dir=out_dir,
            burn_sweeps=args.burn_in_sweeps,
            window_sweeps=args.window_sweeps,
            min_windows=args.min_windows,
            max_windows=args.max_windows,
            last_m=args.last_m,
            accept_min=args.accept_min,
            min_strobe_transitions=args.min_strobe_transitions,
            min_strobe_unique=3,
            min_strobe_bidir=1,
            start_total=start_total,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            cycle=cycle,
            resume=args.resume,
        )
        summary = _pick_seed_row(summaries, screen_seed)
        mismatch = float(summary.get("mismatch_mean_last_m") or 0.0)
        mismatch_drop = (mismatch_drive0 - mismatch) / mismatch_drive0 if mismatch_drive0 > 0 else 0.0
        summary["mismatch_drop"] = mismatch_drop
        summary["eta_drive"] = eta_drive
        summary["eta"] = eta_for_drive
        screen_rows.append(summary)
        if (
            float(summary.get("k_drive_mean_last_m", 0.0)) >= args.k_drive_mean_min
            and float(summary.get("accept_mean_last_m", 0.0)) >= args.accept_min
            and mismatch_drop >= args.mismatch_drop_frac
        ):
            eligible_drive.append(summary)

    if not eligible_drive:
        report_lines.append("No eligible eta_drive candidates in drive-only screen.")
        _write_report(out_dir / "PHASE7_META_LAYER_REPORT.md", report_lines)
        return

    drive_best = max(
        eligible_drive,
        key=lambda r: (float(r.get("mismatch_drop", 0.0)), float(r.get("k_drive_mean_last_m", 0.0))),
    )
    eta_drive_best = float(drive_best["eta_drive"])
    confirm_drive_case = _case_name("drive_confirm_eta", eta_drive_best)
    drive_pass = 0
    params = build_case_params(
        base, confirm_drive_case, args.layers, eta=eta_for_drive, eta_drive=eta_drive_best, p6_on=True
    )
    drive_summaries = run_case(
        case=confirm_drive_case,
        params=params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        accept_min=args.accept_min,
        min_strobe_transitions=args.min_strobe_transitions,
        min_strobe_unique=3,
        min_strobe_bidir=1,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        cycle=cycle,
        resume=args.resume,
    )
    for summary in drive_summaries:
        try:
            seed_val = int(summary.get("seed", -1))
        except (TypeError, ValueError):
            continue
        if seed_val not in seeds:
            continue
        mismatch = float(summary.get("mismatch_mean_last_m") or 0.0)
        mismatch_drop = (mismatch_drive0 - mismatch) / mismatch_drive0 if mismatch_drive0 > 0 else 0.0
        summary["mismatch_drop"] = mismatch_drop
        summary["eta_drive"] = eta_drive_best
        summary["eta"] = eta_for_drive
        confirm_rows.append(summary)
        if (
            float(summary.get("k_drive_mean_last_m", 0.0)) >= args.k_drive_mean_min
            and float(summary.get("accept_mean_last_m", 0.0)) >= args.accept_min
            and mismatch_drop >= args.mismatch_drop_frac
        ):
            drive_pass += 1

    if drive_pass < 2:
        report_lines.append("Drive-only confirm failed (<2/3 seeds).")
        _write_report(out_dir / "PHASE7_META_LAYER_REPORT.md", report_lines)
        return

    drive_path = preset_dir / (
        f"meta_p6_drive_eta{eta_for_drive:.2f}_etaDrive{eta_drive_best:.2f}_layers{args.layers}.json"
    )
    write_preset(
        drive_path,
        {
            "layers": args.layers,
            "p3_on": False,
            "p6_on": True,
            "eta": eta_for_drive,
            "eta_drive": eta_drive_best,
            "config_id": (
                f"phase7_meta_p6_drive_eta{eta_for_drive:.2f}_etaDrive{eta_drive_best:.2f}_l{args.layers}"
            ),
        },
    )

    agg_path = out_dir / "agg.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        fieldnames = [
            "case",
            "seed",
            "status",
            "windows_completed",
            "accept_mean_last_m",
            "ep_mean_last_m",
            "ep_ci_half_last_m",
            "mismatch_mean_last_m",
            "mismatch_drop",
            "k_drive_mean_last_m",
        ]
        writer = csv.DictWriter(ah, fieldnames=fieldnames)
        writer.writeheader()
        for row in screen_rows + confirm_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    report_path = out_dir / "PHASE7_META_LAYER_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 7 meta-layer sanity v1\n\n")
        fh.write("## Equilibrium coupling screen\n\n")
        fh.write("| eta | ep_mean_last_m | ep_ci_half_last_m | mismatch_last_m | mismatch_drop | accept_last_m | status |\n")
        fh.write("| ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in screen_rows:
            if not str(row.get("case", "")).startswith("null_eta"):
                continue
            fh.write(
                f"| {row.get('eta',0.0):.3g} | {row.get('ep_mean_last_m',0.0):.6g} | "
                f"{row.get('ep_ci_half_last_m',0.0):.6g} | {row.get('mismatch_mean_last_m',0.0):.6g} | "
                f"{row.get('mismatch_drop',0.0):.6g} | {row.get('accept_mean_last_m',0.0):.6g} | "
                f"{row.get('status','')} |\n"
            )
        fh.write(f"\nSelected eta_best={eta_best_val:.3g}\n\n")
        fh.write("## Drive-only screen\n\n")
        fh.write("| eta_drive | k_drive_mean_last_m | mismatch_last_m | mismatch_drop | accept_last_m | status |\n")
        fh.write("| ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in screen_rows:
            if not str(row.get("case", "")).startswith("drive_eta"):
                continue
            fh.write(
                f"| {row.get('eta_drive',0.0):.3g} | {row.get('k_drive_mean_last_m',0.0):.6g} | "
                f"{row.get('mismatch_mean_last_m',0.0):.6g} | {row.get('mismatch_drop',0.0):.6g} | "
                f"{row.get('accept_mean_last_m',0.0):.6g} | {row.get('status','')} |\n"
            )
        fh.write(f"\nSelected eta_drive_best={eta_drive_best:.3g}\n\n")
        fh.write("## Confirm (eta_best + eta_drive_best)\n\n")
        fh.write("| case | seed | status | ep_mean_last_m | ep_ci_half_last_m | mismatch_last_m | mismatch_drop | ")
        fh.write("k_drive_mean_last_m | accept_last_m |\n")
        fh.write("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in confirm_rows:
            fh.write(
                f"| {row.get('case','')} | {row.get('seed',0)} | {row.get('status','')} | "
                f"{row.get('ep_mean_last_m',0.0):.6g} | {row.get('ep_ci_half_last_m',0.0):.6g} | "
                f"{row.get('mismatch_mean_last_m',0.0):.6g} | {row.get('mismatch_drop',0.0):.6g} | "
                f"{row.get('k_drive_mean_last_m',0.0):.6g} | {row.get('accept_mean_last_m',0.0):.6g} |\n"
            )


def _write_report(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 7 meta-layer sanity v1\n\n")
        for line in lines:
            fh.write(f"{line}\n")


if __name__ == "__main__":
    main()
