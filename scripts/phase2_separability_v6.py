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


def _params_to_preset(params: Params) -> Dict[str, Any]:
    data = dict(params.__dict__)
    shape = data.get("shape")
    if isinstance(shape, tuple):
        data["shape"] = list(shape)
    data["device"] = str(data.get("device", ""))
    if isinstance(data.get("kernel_weights"), dict):
        data["kernel_weights"] = dict(data["kernel_weights"])
    return data


def _ensure_k_weights(kw: Dict[str, float]) -> Dict[str, float]:
    kw = dict(kw)
    kw["k_local"] = max(float(kw.get("k_local", 0.0) or 0.0), 0.25)
    kw["k_neighbor_trade"] = max(float(kw.get("k_neighbor_trade", 0.0) or 0.0), 0.25)
    return kw


def build_params(base: Params, case: str, eta_drive: float | None = None) -> Params:
    kw = _ensure_k_weights(base.kernel_weights)
    overrides: Dict[str, Any] = {
        "B_k": 2,
        "radius_k": 2,
        "l_k": 3,
        "eta": 0.0,
        "kernel_weights": kw,
    }
    if case == "meta_null_k":
        overrides.update({"p3_on": False, "p6_on": False, "eta_drive": 0.0})
    elif case == "p6_drive_k":
        overrides.update(
            {"p3_on": False, "p6_on": True, "eta_drive": float(eta_drive) if eta_drive is not None else 1.0}
        )
    else:
        raise ValueError(f"unknown case {case}")
    return Params.from_dict(base, overrides)


def _k_drive_ep(snapshot: Dict[str, Any], kernels: List[str]) -> Tuple[float, float]:
    rates = snapshot.get("ep_rate_by_kernel_proposal_window", {})
    props = snapshot.get("window_proposals_by_kernel", {})
    num = 0.0
    den = 0
    for k in kernels:
        w = int(props.get(k, 0))
        num += w * float(rates.get(k, 0.0))
        den += w
    return (num / den) if den else 0.0, den


def _k_drive_accept(snapshot: Dict[str, Any], kernels: List[str]) -> float:
    acc = snapshot.get("window_accept_frac_by_kernel", {})
    props = snapshot.get("window_proposals_by_kernel", {})
    num = 0.0
    den = 0
    for k in kernels:
        w = int(props.get(k, 0))
        num += w * float(acc.get(k, 0.0))
        den += w
    return (num / den) if den else 0.0


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
    ci_thresh: float,
    accept_min: float,
    mismatch_drop_frac: float,
    max_seconds_per_run: float,
    start_total: float,
    max_seconds_total: float,
    eta_drive: float = 0.0,
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
                "case,seed,step,window_index,k_drive_ep_rate,k_drive_mean_last_m,k_drive_ci_half,"
                "k_drive_accept_window,window_steps,window_proposals,mismatch_abs_mean,pass\n"
            )

    raw_rows: List[Dict[str, Any]] = []
    kernels = ["k_local", "k_neighbor_trade"]
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
        jsonl_path = jsonl_dir / f"{case}_seed{seed}_eta{eta_drive}.jsonl"
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
            jsonl_handle.flush()
            k_rate, _ = _k_drive_ep(snapshot, kernels)
            k_accept = _k_drive_accept(snapshot, kernels)
            progress_handle.write(
                f"{case},{seed},{snapshot['step']},{len(window_rates)+1},"
                f"{k_rate},{0.0},{0.0},{k_accept},"
                f"{snapshot.get('window_steps', 0)},{snapshot.get('window_proposals', 0)},"
                f"{snapshot.get('mismatch_abs_mean', '')},False\n"
            )
            progress_handle.flush()
            os.fsync(progress_handle.fileno())

            mismatch = snapshot.get("mismatch_abs_mean", None)
            if mismatch_first is None:
                mismatch_first = mismatch
            mismatch_last = mismatch
            window_rates.append(k_rate)
            window_accepts.append(k_accept)

            if len(window_rates) >= min_windows:
                tail = window_rates[-last_m:]
                acc_tail = window_accepts[-last_m:]
                mean_ep = sum(tail) / len(tail)
                var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
                ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
                accept_mean = sum(acc_tail) / len(acc_tail)
                pass_now = False
                if case == "meta_null_k":
                    pass_now = abs(mean_ep) <= mean_thresh and ci_half <= ci_thresh and accept_mean >= accept_min
                    if pass_now:
                        status = "PASS_EARLY"
                    elif ci_half <= ci_thresh and abs(mean_ep) > mean_thresh:
                        status = "FAIL_MEAN_EARLY"
                    elif accept_mean < accept_min and len(acc_tail) >= 2 and acc_tail[-2] < accept_min:
                        status = "FAIL_ACCEPT_LOW"
                else:
                    ok_mismatch = True
                    if mismatch_first is not None and mismatch_last is not None:
                        ok_mismatch = mismatch_last <= (1 - mismatch_drop_frac) * mismatch_first
                    pass_now = (
                        mean_ep >= mean_thresh
                        and (mean_ep - ci_half) > 0
                        and accept_mean >= accept_min
                        and ok_mismatch
                    )
                    if pass_now:
                        status = "PASS_EARLY"
                    elif (mean_ep + ci_half) < 0:
                        status = "FAIL_WRONG_SIGN"
                    elif ci_half <= ci_thresh and mean_ep < mean_thresh:
                        status = "FAIL_MEAN_EARLY"
                    elif accept_mean < accept_min and len(acc_tail) >= 2 and acc_tail[-2] < accept_min:
                        status = "FAIL_ACCEPT_LOW"
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
                "eta_drive": eta_drive,
                "seed": seed,
                "status": status,
                "windows_used": len(window_rates),
                "k_drive_mean_last_m": mean_ep,
                "k_drive_ci_half": ci_half,
                "k_drive_accept_mean": accept_mean,
                "mismatch_first": mismatch_first,
                "mismatch_last": mismatch_last,
            }
        )
        print(
            f"SUMMARY case={case} seed={seed} eta={eta_drive} status={status} "
            f"windows={len(window_rates)} mean={mean_ep} ci={ci_half} accept={accept_mean}"
        )
        if case == "meta_null_k" and status != "PASS_EARLY":
            break

    if raw_rows:
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)
        pass_count = sum(1 for r in raw_rows if r["status"] == "PASS_EARLY")
        with agg_path.open("w", encoding="utf-8", newline="") as ah:
            writer = csv.DictWriter(
                ah,
                fieldnames=["case", "eta_drive", "pass_count", "total", "pass_rate"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "case": case,
                    "eta_drive": eta_drive,
                    "pass_count": pass_count,
                    "total": len(raw_rows),
                    "pass_rate": pass_count / max(1, len(raw_rows)),
                }
            )
    return raw_rows


def main():
    parser = argparse.ArgumentParser(description="Phase 2 separability v6")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", default=".tmp/phase2_v6")
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
    parser.add_argument("--accept-min-k", type=float, default=0.01)
    parser.add_argument("--mismatch-drop-frac", type=float, default=0.01)
    parser.add_argument("--eta-drive-sweep", default="1,2,4")
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=900)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    preset = _load_preset(Path(args.preset))
    base = _as_params(preset, {"device": args.device})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_total = time.monotonic()

    # meta_null_k
    meta_params = build_params(base, "meta_null_k")
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
        mean_thresh=args.mean_thresh_null,
        ci_thresh=args.ci_thresh_null,
        accept_min=args.accept_min_k,
        mismatch_drop_frac=args.mismatch_drop_frac,
        max_seconds_per_run=args.max_seconds_per_run,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
    )
    meta_pass = all(r["status"] == "PASS_EARLY" for r in meta_rows) and len(meta_rows) == len(seeds)
    if not meta_pass:
        print("STATUS_COUNTS meta_null_k:", {r["status"]: 1 for r in meta_rows})
        return
    # p6_drive_k sweep
    eta_vals = [float(x) for x in args.eta_drive_sweep.split(",") if x.strip()]
    best_rows = []
    selected_eta = None
    for eta in eta_vals:
        if time.monotonic() - start_total > args.max_seconds_total:
            break
        drive_params = build_params(base, "p6_drive_k", eta_drive=eta)
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
            mean_thresh=args.mean_thresh_drive,
            ci_thresh=args.ci_thresh_drive,
            accept_min=args.accept_min_k,
            mismatch_drop_frac=args.mismatch_drop_frac,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
            max_seconds_total=args.max_seconds_total,
            eta_drive=eta,
        )
        if all(r["status"] == "PASS_EARLY" for r in drive_rows):
            selected_eta = eta
            best_rows = drive_rows
            print(f"STATUS_COUNTS meta_null_k: {{'PASS_EARLY': {len(meta_rows)}}}")
            print(f"STATUS_COUNTS p6_drive_k: {{'PASS_EARLY': {len(drive_rows)}}}")
            preset_out = Path("scripts/params/phase2_drive_k_balanced_v6.json")
            preset_out.parent.mkdir(parents=True, exist_ok=True)
            preset_out.write_text(json.dumps(_params_to_preset(drive_params), indent=2))
            return
        best_rows = drive_rows
    print(f"STATUS_COUNTS meta_null_k: {{'PASS_EARLY': {len(meta_rows)}}}")
    if best_rows:
        counts = {}
        for r in best_rows:
            counts[r['status']] = counts.get(r['status'],0)+1
        print(f"STATUS_COUNTS p6_drive_k: {counts}")


if __name__ == "__main__":
    main()
