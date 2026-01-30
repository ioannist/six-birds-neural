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
from ratchet_gpu.sim import run_sim

# allow importing sibling screen helpers
SCRIPT_DIR = Path(__file__).resolve().parent
import sys

sys.path.append(str(SCRIPT_DIR))
from phase1_null_screen_v4 import _expected_proposals_per_step, _rate_micro  # type: ignore


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
                "B_k": max(1, base.B_k),
                "radius_k": 2,
                "l_k": max(1, base.l_k),
                "eta": 0.0,
                "eta_drive": 0.0,
            }
        )
        if kw.get("k_local", 0.0) == 0.0 and kw.get("k_neighbor_trade", 0.0) == 0.0:
            kw["k_local"] = 1.0
    elif case == "p6_drive_k":
        overrides.update(
            {
                "p3_on": False,
                "p6_on": True,
                "B_k": max(1, base.B_k),
                "radius_k": 2,
                "l_k": max(1, base.l_k),
                "eta": 0.0,
                "eta_drive": 1.0,
            }
        )
        if kw.get("k_local", 0.0) == 0.0 and kw.get("k_neighbor_trade", 0.0) == 0.0:
            kw["k_local"] = 1.0
    elif case == "p3_protocol_mix":
        overrides.update(
            {
                "p3_on": True,
                "p6_on": False,
                "B_k": max(1, base.B_k),
                "radius_k": 2,
                "l_k": max(1, base.l_k),
                "eta": 0.0,
                "eta_drive": 0.0,
            }
        )
        kw.setdefault("k_local", 0.5)
        kw.setdefault("k_neighbor_trade", 0.5)
        kw.setdefault("n_flip", 0.5)
    else:
        raise ValueError(f"unknown case {case}")
    overrides["kernel_weights"] = kw
    return Params.from_dict(base, overrides)


def _case_fail_config(params: Params, case: str) -> str | None:
    kw = params.kernel_weights
    k_enabled = any(kw.get(k, 0.0) > 0.0 for k in ["k_local", "k_neighbor_trade", "k_p5_exchange"])
    if case == "p6_drive_k":
        if params.B_k <= 0 or params.radius_k <= 0 or not k_enabled:
            return "FAIL_CONFIG_P6_NO_K"
    return None


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
    max_seconds_per_run: float,
    start_time: float,
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
        if time.monotonic() - start_time > max_seconds_total:
            print("TOTAL TIME CAP HIT")
            break
        config_fail = _case_fail_config(params, case)
        if config_fail:
            raw_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "status": config_fail,
                    "windows_used": 0,
                    "mean_ep": 0.0,
                    "ci_half": float("inf"),
                    "acceptedFracWindow": 0.0,
                    "mismatch_first": None,
                    "mismatch_last": None,
                    "strobe_rate": 0.0,
                }
            )
            continue

        diag_state = None
        window_rates: List[float] = []
        window_accepts: List[float] = []
        mismatch_first = None
        mismatch_last = None
        status = "RUNNING"
        jsonl_path = jsonl_dir / f"{case}_seed{seed}.jsonl"
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        progress_handle = progress_path.open("a", encoding="utf-8")

        def report_cb(state, step, ep_ledger, accepted_frac):
            nonlocal diag_state, status, mismatch_first, mismatch_last
            if len(window_rates) >= max_windows:
                status = "FAIL_MAX_WINDOWS"
                return
            snapshot, diag_state = compute_snapshot(
                state, step, ep_ledger, accepted_frac, diag_state
            )
            jsonl_handle.write(to_json_line(snapshot) + "\n")
            if progress_handle:
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
                    f"{snapshot.get('ep_rate_by_kernel_window', {})}\n"
                )
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
                mean_ep = sum(tail) / len(tail)
                var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
                ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
                accept_mean = sum(window_accepts[-last_m:]) / len(window_accepts[-last_m:])
                # pass rules
                if case == "meta_null_k":
                    if abs(mean_ep) <= 2e-4 and ci_half <= 1e-4 and accept_mean >= 0.005:
                        status = "PASS_EARLY"
                elif case == "p6_drive_k":
                    ok_mismatch = True
                    if mismatch_first is not None and mismatch_last is not None:
                        ok_mismatch = mismatch_last <= mismatch_first * 0.99
                    if abs(mean_ep) >= 2e-4 and ok_mismatch:
                        status = "PASS_EARLY"
                    elif mean_ep < 0 and len(window_rates) >= min_windows + 5:
                        status = "FAIL_WRONG_SIGN"
                elif case == "p3_protocol_mix":
                    strobe = snapshot.get("epStrobeRate", 0.0)
                    if strobe != 0.0 and abs(strobe) >= 1e-4:
                        status = "PASS_EARLY"
                if accept_mean < 0.002:
                    status = "FAIL_MIXING"

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
        summary = {}
        try:
            summary = run_sim(
                params,
                seed=seed,
                steps=max_steps,
                report_every=window_steps,
                device=params.device,
                report_callback=None,
                stop_callback=None,
            )
        except Exception:
            summary = {}
        jsonl_handle.close()
        progress_handle.close()
        if status == "RUNNING":
            status = "FAIL_MAX_WINDOWS"
        tail = window_rates[-last_m:] if window_rates else [0.0]
        mean_ep = sum(tail) / len(tail)
        var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
        ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
        accepted_mean = sum(window_accepts[-last_m:]) / len(window_accepts[-last_m:]) if window_accepts else 0.0
        strobe_rate = float(summary.get("epStrobeRate", 0.0)) if summary else 0.0
        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "status": status,
                "windows_used": len(window_rates),
                "mean_ep": mean_ep,
                "ci_half": ci_half,
                "acceptedFracWindow": accepted_mean,
                "mismatch_first": mismatch_first,
                "mismatch_last": mismatch_last,
                "strobe_rate": strobe_rate,
            }
        )
        print(
            f"SUMMARY seed={seed} case={case} status={status} windows={len(window_rates)} "
            f"lastm_mean={mean_ep} lastm_ci={ci_half} lastm_accept={accepted_mean}"
        )

    with raw_path.open("w", encoding="utf-8", newline="") as rh:
        if raw_rows:
            writer = csv.DictWriter(rh, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

    # agg
    agg = {}
    for r in raw_rows:
        agg.setdefault(r["case"], []).append(r)
    agg_rows = []
    for case_name, lst in agg.items():
        strobe_vals = [float(r.get("strobe_rate", 0.0)) for r in lst]
        agg_rows.append(
            {
                "case": case_name,
                "pass_count": sum(1 for r in lst if r["status"] == "PASS_EARLY"),
                "total": len(lst),
                "pass_rate": sum(1 for r in lst if r["status"] == "PASS_EARLY") / max(1, len(lst)),
                "strobe_rate_mean": sum(strobe_vals) / len(strobe_vals) if strobe_vals else 0.0,
            }
        )
    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        if agg_rows:
            writer = csv.DictWriter(ah, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
    return raw_rows


def main():
    parser = argparse.ArgumentParser(description="Phase 2 separability v2")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", default=".tmp/phase2_separability_v2")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--min-windows", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--max-seconds-total", type=float, default=6500)
    parser.add_argument("--max-seconds-per-run", type=float, default=900)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = _as_params(_load_preset(Path(args.preset)), {"device": args.device})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    raw_all = []
    for case in ["meta_null_k", "p6_drive_k", "p3_protocol_mix"]:
        params = build_case_params(base, case)
        raw_all.extend(
            run_case(
                case=case,
                params=params,
                seeds=seeds,
                out_dir=out_dir,
                burn_sweeps=args.burn_in_sweeps,
                window_sweeps=args.window_sweeps,
                min_windows=args.min_windows,
                max_windows=args.max_windows,
                last_m=args.last_m,
                max_seconds_per_run=args.max_seconds_per_run,
                start_time=start,
                max_seconds_total=args.max_seconds_total,
            )
        )
    # summary counts
    from collections import Counter, defaultdict

    counts = defaultdict(Counter)
    for r in raw_all:
        counts[r["case"]][r["status"]] += 1
    print(f"STATUS COUNTS: {dict((k, dict(v)) for k, v in counts.items())}")

    # write top-level raw/agg
    top_raw = out_dir / "raw.csv"
    top_agg = out_dir / "agg.csv"
    if raw_all:
        with top_raw.open("w", encoding="utf-8", newline="") as th:
            writer = csv.DictWriter(th, fieldnames=list(raw_all[0].keys()))
            writer.writeheader()
            writer.writerows(raw_all)
        agg_rows = []
        by_case = {}
        for r in raw_all:
            by_case.setdefault(r["case"], []).append(r)
        for case_name, lst in by_case.items():
            agg_rows.append(
                {
                    "case": case_name,
                    "pass_count": sum(1 for r in lst if r["status"] == "PASS_EARLY"),
                    "total": len(lst),
                    "pass_rate": sum(1 for r in lst if r["status"] == "PASS_EARLY") / max(1, len(lst)),
                }
            )
        with top_agg.open("w", encoding="utf-8", newline="") as ah:
            writer = csv.DictWriter(ah, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)


if __name__ == "__main__":
    main()
