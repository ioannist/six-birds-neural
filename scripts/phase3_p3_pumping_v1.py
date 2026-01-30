#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
try:
    from phase1_null_screen_v4 import _expected_proposals_per_step  # type: ignore
except Exception:  # pragma: no cover - fallback in case script import changes
    def _expected_proposals_per_step(N: int, device: str, kernel_weights: Dict[str, float]) -> float:
        return float(N)


def _load_preset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with path.open() as f:
        return json.load(f)


def _as_params(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Params:
    data = {k: v for k, v in preset.items() if k not in {"config_id", "pass", "note"}}
    data.update(overrides)
    if isinstance(data.get("shape"), list):
        data["shape"] = tuple(data["shape"])
    if isinstance(data.get("kernel_weights"), dict):
        data["kernel_weights"] = dict(data["kernel_weights"])
    # drop auxiliary preset-only fields if present
    data.pop("w_neighbor_weight", None)
    return Params(**data)


def _case_overrides(base: Params, case: str) -> Params:
    overrides = {
        "p6_on": False,
        "eta_drive": 0.0,
        "strobe_on": True,
    }
    if case == "control_p3_off":
        overrides["p3_on"] = False
    elif case == "protocol_p3_on":
        overrides["p3_on"] = True
    else:
        raise ValueError(f"unknown case {case}")
    return Params.from_dict(base, overrides)


def _strobe_fields(snapshot: Dict[str, Any]) -> Tuple[float, int]:
    if "strobe_rate_window" not in snapshot:
        raise RuntimeError("strobe metric missing in snapshot")
    return float(snapshot.get("strobe_rate_window", 0.0)), int(
        snapshot.get("strobe_transitions_window", 0)
    )


def run_case(
    case: str,
    params: Params,
    seeds: List[int],
    out_dir: Path,
    burn_sweeps: int,
    window_sweeps: int,
    min_windows: int,
    max_windows: int,
    last_m: int,
    mean_thresh: float,
    diff_thresh: float,
    ci_thresh: float,
    accept_min: float,
    min_strobe_transitions: int,
    max_seconds_per_run: float,
    start_total: float,
    max_seconds_total: float,
    control_stats: Dict[int, Dict[str, float]] | None = None,
) -> List[Dict[str, Any]]:
    N = math.prod(params.shape)
    expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected_props))
    window_steps = int(math.ceil(window_sweeps * N / expected_props))

    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    progress_path = case_dir / "progress.csv"
    raw_path = case_dir / "raw.csv"
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,step,window_index,strobe_rate,strobe_mean_last_m,strobe_ci_half,"
                "acceptedFracWindow,strobe_transitions,pass\n"
            )

    raw_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        if time.monotonic() - start_total > max_seconds_total:
            print("TOTAL TIME CAP HIT")
            break
        diag_state = None
        strobe_rates: List[float] = []
        accepts: List[float] = []
        status = "RUNNING"
        jsonl_path = jsonl_dir / f"{case}_seed{seed}.jsonl"
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        progress_handle = progress_path.open("a", encoding="utf-8")
        run_start = time.monotonic()
        burn_steps = burn_sweeps * N

        def report_cb(state, step, ep_ledger, accepted_frac):
            nonlocal diag_state, status
            if len(strobe_rates) >= max_windows or status != "RUNNING":
                return
            snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
            strobe_rate, strobe_transitions = _strobe_fields(snapshot)
            jsonl_handle.write(to_json_line(snapshot) + "\n")
            jsonl_handle.flush()
            progress_handle.write(
                f"{case},{seed},{snapshot['step']},{len(strobe_rates)+1},{strobe_rate},0.0,0.0,"
                f"{snapshot.get('acceptedFrac', 0.0)},{strobe_transitions},{step <= burn_steps}\n"
            )
            progress_handle.flush()
            os.fsync(progress_handle.fileno())

            if step <= burn_steps:
                return

            strobe_rates.append(strobe_rate)
            accepts.append(float(snapshot.get("acceptedFrac", 0.0)))

            stride = snapshot.get("strobe_stride", 0)
            cyc = snapshot.get("strobe_cycle_len", 0)
            if stride and cyc and stride != cyc:
                status = "FAIL_STROBE_NOT_STROBOSCOPIC"
                return

            if len(strobe_rates) >= min_windows:
                tail = strobe_rates[-last_m:]
                acc_tail = accepts[-last_m:]
                mean_val = sum(tail) / len(tail)
                var = sum((v - mean_val) ** 2 for v in tail) / max(1, len(tail) - 1)
                ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
                accept_mean = sum(acc_tail) / len(acc_tail)

                # sparsity guard
                if strobe_transitions < min_strobe_transitions:
                    status = "FAIL_STROBE_SPARSE"

                if accept_mean < accept_min and len(acc_tail) >= 2 and acc_tail[-2] < accept_min:
                    status = "FAIL_ACCEPT"

                if case == "control_p3_off" and status == "RUNNING":
                    if abs(mean_val) <= mean_thresh and ci_half <= ci_thresh and accept_mean >= accept_min:
                        status = "PASS_EARLY"
                    elif ci_half <= ci_thresh and abs(mean_val) > mean_thresh:
                        status = "FAIL_MEAN_EARLY"
                elif case == "protocol_p3_on" and status == "RUNNING" and control_stats is not None:
                    ctrl = control_stats.get(seed, {})
                    ctrl_mean = ctrl.get("mean", 0.0)
                    ctrl_ci = ctrl.get("ci_half", 0.0)
                    diff = mean_val - ctrl_mean
                    if diff >= diff_thresh and max(ci_half, ctrl_ci) <= ci_thresh and accept_mean >= accept_min:
                        status = "PASS_EARLY"
                    elif ci_half <= ci_thresh and diff < diff_thresh:
                        status = "FAIL_MEAN_EARLY"

            if status == "RUNNING" and time.monotonic() - run_start > max_seconds_per_run:
                status = "FAIL_TIME"

        def stop_cb(state, step, ep_ledger, accepted_frac):
            return status != "RUNNING" or len(strobe_rates) >= max_windows

        max_steps = burn_steps + max_windows * window_steps
        run_sim(
            params,
            seed=seed,
            steps=max_steps,
            report_every=window_steps,
            device=params.device,
            report_callback=report_cb,
            stop_callback=stop_cb,
        )
        jsonl_handle.close()
        progress_handle.close()
        if status == "RUNNING":
            status = "FAIL_MAX_WINDOWS"
        tail = strobe_rates[-last_m:] if strobe_rates else [0.0]
        mean_val = sum(tail) / len(tail)
        var = sum((v - mean_val) ** 2 for v in tail) / max(1, len(tail) - 1)
        ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
        accept_mean = sum(accepts[-last_m:]) / len(accepts[-last_m:]) if accepts else 0.0

        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "status": status,
                "windows_used": len(strobe_rates),
                "strobe_mean_last_m": mean_val,
                "strobe_ci_half": ci_half,
                "acceptedFracWindowMean": accept_mean,
            }
        )
        print(
            f"SUMMARY case={case} seed={seed} status={status} windows={len(strobe_rates)} "
            f"mean={mean_val} ci={ci_half} accept={accept_mean}"
        )

        if case == "control_p3_off" and status != "PASS_EARLY":
            break

    if raw_rows:
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
        counts: Dict[str, int] = {}
        for r in raw_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        with (case_dir / "status_counts.json").open("w", encoding="utf-8") as sh:
            json.dump(counts, sh, indent=2)
        pass_count = sum(1 for r in raw_rows if r["status"] == "PASS_EARLY")
        with (case_dir / "agg.csv").open("w", encoding="utf-8", newline="") as ah:
            writer = csv.DictWriter(
                ah, fieldnames=["case", "pass_count", "total", "pass_rate"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "case": case,
                    "pass_count": pass_count,
                    "total": len(raw_rows),
                    "pass_rate": pass_count / max(1, len(raw_rows)),
                }
            )
    return raw_rows


def main():
    parser = argparse.ArgumentParser(description="Phase 3 P3 pumping v1")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", default="scripts/params/phase2_drive_k_balanced_v6.json")
    parser.add_argument("--out-dir", default=".tmp/phase3_p3_pumping_v1")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--min-windows", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--accept-min", type=float, default=0.01)
    parser.add_argument("--mean-thresh-control", type=float, default=3e-4)
    parser.add_argument("--diff-thresh", type=float, default=3e-4)
    parser.add_argument("--ci-thresh", type=float, default=8e-4)
    parser.add_argument("--min-strobe-transitions", type=int, default=1000)
    parser.add_argument("--max-seconds-total", type=float, default=6600)
    parser.add_argument("--max-seconds-per-run", type=float, default=900)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preset = _load_preset(Path(args.preset))
    base = _as_params(preset, {"device": args.device})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.monotonic()

    control_params = _case_overrides(base, "control_p3_off")
    control_rows = run_case(
        case="control_p3_off",
        params=control_params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        mean_thresh=args.mean_thresh_control,
        diff_thresh=args.diff_thresh,
        ci_thresh=args.ci_thresh,
        accept_min=args.accept_min,
        min_strobe_transitions=args.min_strobe_transitions,
        max_seconds_per_run=args.max_seconds_per_run,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
    )

    control_stats: Dict[int, Dict[str, float]] = {}
    for row in control_rows:
        control_stats[row["seed"]] = {
            "mean": float(row.get("strobe_mean_last_m", 0.0)),
            "ci_half": float(row.get("strobe_ci_half", 0.0)),
        }
    if not control_rows or any(r["status"] != "PASS_EARLY" for r in control_rows):
        print("STATUS_COUNTS control_p3_off:", {r["status"]: 1 for r in control_rows})
        return

    protocol_params = _case_overrides(base, "protocol_p3_on")
    proto_rows = run_case(
        case="protocol_p3_on",
        params=protocol_params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        mean_thresh=args.mean_thresh_control,
        diff_thresh=args.diff_thresh,
        ci_thresh=args.ci_thresh,
        accept_min=args.accept_min,
        min_strobe_transitions=args.min_strobe_transitions,
        max_seconds_per_run=args.max_seconds_per_run,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        control_stats=control_stats,
    )

    print("STATUS_COUNTS control_p3_off:", {r["status"]: 1 for r in control_rows})
    print("STATUS_COUNTS protocol_p3_on:", {r["status"]: 1 for r in proto_rows})

    report_path = out_dir / "PHASE3_P3_PUMPING_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 3 P3 pumping v1\n\n")
        fh.write(f"preset: {args.preset}\n\n")
        for case, rows in [("control_p3_off", control_rows), ("protocol_p3_on", proto_rows)]:
            fh.write(f"## {case}\n\n")
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
