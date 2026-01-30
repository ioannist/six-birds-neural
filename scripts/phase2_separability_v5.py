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
sys.path.append(str(SCRIPT_DIR))
from phase1_null_screen_v4 import _expected_proposals_per_step  # type: ignore


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
    data.pop("w_neighbor_weight", None)
    return Params(**data)


def build_case_params(base: Params, case: str) -> Params:
    kw = dict(base.kernel_weights)
    overrides: Dict[str, Any] = {}
    if case == "meta_null_k":
        overrides.update(
            {
                "p3_on": False,
                "p6_on": False,
                "eta": 0.0,
                "eta_drive": 0.0,
                "B_k": 2,
                "radius_k": 2,
                "l_k": 3,
            }
        )
        kw["k_local"] = max(float(kw.get("k_local", 0.0) or 0.0), 0.25)
        kw["k_neighbor_trade"] = max(float(kw.get("k_neighbor_trade", 0.0) or 0.0), 0.25)
    elif case == "p6_drive_k":
        overrides.update(
            {
                "p3_on": False,
                "p6_on": True,
                "eta": 0.0,
                "eta_drive": 1.0,
                "B_k": 2,
                "radius_k": 2,
                "l_k": 3,
            }
        )
        kw["k_local"] = max(float(kw.get("k_local", 0.0) or 0.0), 0.25)
        kw["k_neighbor_trade"] = max(float(kw.get("k_neighbor_trade", 0.0) or 0.0), 0.25)
    else:
        raise ValueError(f"unknown case {case}")
    if kw["k_local"] + kw["k_neighbor_trade"] <= 0:
        raise ValueError(
            f"Phase2 case requires K kernels enabled; got k_local={kw.get('k_local',0)} "
            f"k_neighbor_trade={kw.get('k_neighbor_trade',0)}"
        )
    overrides["kernel_weights"] = kw
    return Params.from_dict(base, overrides)


def _pass_fail_null(tail: List[float], accept_tail: List[float], mean_thresh: float, ci_thresh: float, accept_min: float) -> Tuple[str, Dict[str, float]]:
    mean_ep = sum(tail) / len(tail)
    var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
    ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
    accept_mean = sum(accept_tail) / len(accept_tail)
    status = "RUNNING"
    if abs(mean_ep) <= mean_thresh and ci_half <= ci_thresh and accept_mean >= accept_min:
        status = "PASS_EARLY"
    elif ci_half <= ci_thresh and abs(mean_ep) > mean_thresh:
        status = "FAIL_MEAN_EARLY"
    elif accept_mean < accept_min and len(accept_tail) >= 2 and accept_tail[-2] < accept_min:
        status = "FAIL_MIXING"
    return status, {"mean_ep": mean_ep, "ci_half": ci_half, "accept_mean": accept_mean}


def _pass_fail_drive(
    tail: List[float],
    accept_tail: List[float],
    mean_thresh: float,
    ci_thresh: float,
    accept_min: float,
    mismatch_first: float | None,
    mismatch_last: float | None,
    mismatch_drop_frac: float,
) -> Tuple[str, Dict[str, float]]:
    mean_ep = sum(tail) / len(tail)
    var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
    ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
    accept_mean = sum(accept_tail) / len(accept_tail)
    status = "RUNNING"
    if (
        mean_ep >= mean_thresh
        and (mean_ep - ci_half) > 0
        and accept_mean >= accept_min
        and mismatch_first is not None
        and mismatch_last is not None
        and mismatch_last <= (1 - mismatch_drop_frac) * mismatch_first
    ):
        status = "PASS_EARLY"
    elif (mean_ep + ci_half) < 0:
        status = "FAIL_WRONG_SIGN"
    elif ci_half <= ci_thresh and mean_ep < mean_thresh:
        status = "FAIL_MEAN_EARLY"
    elif accept_mean < accept_min and len(accept_tail) >= 2 and accept_tail[-2] < accept_min:
        status = "FAIL_MIXING"
    return status, {"mean_ep": mean_ep, "ci_half": ci_half, "accept_mean": accept_mean}


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
    mean_thresh_null: float,
    ci_thresh_null: float,
    mean_thresh_drive: float,
    ci_thresh_drive: float,
    accept_min: float,
    mismatch_drop_frac: float,
    max_seconds_per_run: float,
    start_total: float,
    max_seconds_total: float,
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
    agg_path = case_dir / "agg.csv"
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,step,window_index,ep_rate_exact_window,mean_ep_last_m,ci_half,"
                "acceptedFracWindow,window_steps,window_proposals,window_accepted,mismatch_abs_mean,pass\n"
            )

    raw_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        if time.monotonic() - start_total > max_seconds_total:
            print("TOTAL TIME CAP HIT")
            break
        diag_state = None
        window_rates: List[float] = []
        window_accepts: List[float] = []
        mismatch_first = None
        mismatch_last = None
        status = "RUNNING"
        jsonl_path = jsonl_dir / f"{case}_seed{seed}.jsonl"
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        progress_handle = progress_path.open("a", encoding="utf-8")

        run_start = time.monotonic()

        def report_cb(state, step, ep_ledger, accepted_frac):
            nonlocal diag_state, status, mismatch_first, mismatch_last
            if len(window_rates) >= max_windows or status != "RUNNING":
                return
            snapshot, diag_state = compute_snapshot(
                state, step, ep_ledger, accepted_frac, diag_state
            )
            jsonl_handle.write(to_json_line(snapshot) + "\n")
            progress_handle.write(
                f"{case},{seed},{snapshot['step']},{len(window_rates)+1},"
                f"{snapshot.get('ep_rate_exact_window', 0.0)},"
                f"{snapshot.get('mean_ep_last_m', snapshot.get('ep_rate_exact_window', 0.0))},"
                f"{snapshot.get('ci_half', 0.0)},"
                f"{snapshot.get('acceptedFrac', 0.0)},"
                f"{snapshot.get('window_steps', ep_ledger.get('window_steps', 0))},"
                f"{ep_ledger.get('window_proposals', 0)},"
                f"{ep_ledger.get('window_accepted', 0)},"
                f"{snapshot.get('mismatch_abs_mean', '')},"
                f"{snapshot.get('pass', False)}\n"
            )
            progress_handle.flush()
            os.fsync(progress_handle.fileno())

            rate = float(snapshot.get("ep_rate_exact_window", 0.0))
            accept_window = float(snapshot.get("acceptedFrac", 0.0) or 0.0)
            mismatch = snapshot.get("mismatch_abs_mean", None)
            if mismatch_first is None:
                mismatch_first = mismatch
            mismatch_last = mismatch
            window_rates.append(rate)
            window_accepts.append(accept_window)

            if len(window_rates) >= min_windows:
                tail = window_rates[-last_m:]
                acc_tail = window_accepts[-last_m:]
                if case == "meta_null_k":
                    st, _ = _pass_fail_null(tail, acc_tail, mean_thresh_null, ci_thresh_null, accept_min)
                    if st != "RUNNING":
                        status = st
                elif case == "p6_drive_k":
                    st, _ = _pass_fail_drive(
                        tail,
                        acc_tail,
                        mean_thresh_drive,
                        ci_thresh_drive,
                        accept_min,
                        mismatch_first,
                        mismatch_last,
                        mismatch_drop_frac,
                    )
                    if st != "RUNNING":
                        status = st
            if status == "RUNNING" and time.monotonic() - run_start > max_seconds_per_run:
                status = "FAIL_TIME"

        def stop_cb(state, step, ep_ledger, accepted_frac):
            return status != "RUNNING" or len(window_rates) >= max_windows

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
        tail = window_rates[-last_m:] if window_rates else [0.0]
        mean_ep = sum(tail) / len(tail)
        var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
        ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
        accept_mean = sum(window_accepts[-last_m:]) / len(window_accepts[-last_m:]) if window_accepts else 0.0
        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "status": status,
                "windows_used": len(window_rates),
                "mean_ep": mean_ep,
                "ci_half": ci_half,
                "acceptedFracWindow": accept_mean,
                "mismatch_first": mismatch_first,
                "mismatch_last": mismatch_last,
            }
        )
        print(
            f"SUMMARY case={case} seed={seed} status={status} windows={len(window_rates)} "
            f"lastm_mean={mean_ep} lastm_ci={ci_half} lastm_accept={accept_mean}"
        )
        if case == "meta_null_k" and status != "PASS_EARLY":
            # Fail-fast: stop entire script
            break

    with raw_path.open("w", encoding="utf-8", newline="") as rh:
        if raw_rows:
            writer = csv.DictWriter(rh, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
    agg_rows = []
    if raw_rows:
        agg_rows.append(
            {
                "case": case,
                "pass_count": sum(1 for r in raw_rows if r["status"] == "PASS_EARLY"),
                "total": len(raw_rows),
                "pass_rate": sum(1 for r in raw_rows if r["status"] == "PASS_EARLY") / max(1, len(raw_rows)),
            }
        )
        with agg_path.open("w", encoding="utf-8", newline="") as ah:
            writer = csv.DictWriter(ah, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
    return raw_rows


def main():
    parser = argparse.ArgumentParser(description="Phase 2 separability v5 (fail-fast)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", default=".tmp/phase2_v5")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--min-windows", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--mean-thresh-null", type=float, default=3e-4)
    parser.add_argument("--ci-thresh-null", type=float, default=8e-4)
    parser.add_argument("--mean-thresh-drive", type=float, default=3e-4)
    parser.add_argument("--ci-thresh-drive", type=float, default=8e-4)
    parser.add_argument("--accept-min", type=float, default=0.01)
    parser.add_argument("--mismatch-drop-frac", type=float, default=0.01)
    parser.add_argument("--max-seconds-total", type=float, default=6900)
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

    # meta_null_k first
    meta_params = build_case_params(base, "meta_null_k")
    meta_rows = run_case(
        case="meta_null_k",
        params=meta_params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        mean_thresh_null=args.mean_thresh_null,
        ci_thresh_null=args.ci_thresh_null,
        mean_thresh_drive=args.mean_thresh_drive,
        ci_thresh_drive=args.ci_thresh_drive,
        accept_min=args.accept_min,
        mismatch_drop_frac=args.mismatch_drop_frac,
        max_seconds_per_run=args.max_seconds_per_run,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
    )
    meta_pass_all = all(r["status"] == "PASS_EARLY" for r in meta_rows) and len(meta_rows) == len(seeds)
    if not meta_pass_all:
        print("meta_null_k did not pass all seeds; stopping before p6_drive_k.")
        return

    drive_params = build_case_params(base, "p6_drive_k")
    drive_rows = run_case(
        case="p6_drive_k",
        params=drive_params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        mean_thresh_null=args.mean_thresh_null,
        ci_thresh_null=args.ci_thresh_null,
        mean_thresh_drive=args.mean_thresh_drive,
        ci_thresh_drive=args.ci_thresh_drive,
        accept_min=args.accept_min,
        mismatch_drop_frac=args.mismatch_drop_frac,
        max_seconds_per_run=args.max_seconds_per_run,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
    )

    # write top-level agg
    all_rows = meta_rows + drive_rows
    counts = {"meta_null_k": {}, "p6_drive_k": {}}
    for r in meta_rows:
        counts["meta_null_k"][r["status"]] = counts["meta_null_k"].get(r["status"], 0) + 1
    for r in drive_rows:
        counts["p6_drive_k"][r["status"]] = counts["p6_drive_k"].get(r["status"], 0) + 1
    print(f"STATUS_COUNTS meta_null_k: {counts['meta_null_k']}")
    if drive_rows:
        print(f"STATUS_COUNTS p6_drive_k: {counts['p6_drive_k']}")

    report = out_dir / "PHASE2_V5_REPORT.md"
    lines = ["# Phase2 v5 separability summary", ""]
    lines.append(f"meta_null_k counts: {counts['meta_null_k']}")
    if drive_rows:
        lines.append(f"p6_drive_k counts: {counts['p6_drive_k']}")
    report.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
