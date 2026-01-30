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


def _ensure_k_weights(kw: Dict[str, float]) -> Dict[str, float]:
    kw = dict(kw)
    kw["k_local"] = max(float(kw.get("k_local", 0.0) or 0.0), 0.25)
    kw["k_neighbor_trade"] = max(float(kw.get("k_neighbor_trade", 0.0) or 0.0), 0.25)
    return kw


def _match_cycle_weights(kw: Dict[str, float], cycle: List[str]) -> Dict[str, float]:
    matched = {k: (1.0 if k in cycle else 0.0) for k in kw.keys()}
    for name in cycle:
        matched.setdefault(name, 1.0)
    return matched


def _effective_min_strobe_transitions(requested: int, window_steps: int, cycle_len: int) -> int:
    if requested <= 0:
        return 0
    if cycle_len <= 0:
        cycle_len = 1
    max_obs = window_steps // cycle_len
    max_transitions = max(0, max_obs - 1)
    return min(requested, max_transitions)


def _summarize_tail(values: List[float], last_m: int) -> float:
    tail = values[-last_m:] if values else [0.0]
    return sum(tail) / len(tail)


def _k_drive_rate(snapshot: Dict[str, Any]) -> float:
    rates = snapshot.get("ep_rate_by_kernel_proposal_window", {})
    total = 0.0
    for key in K_KERNELS:
        total += float(rates.get(key, 0.0))
    return total


def build_case_params(base: Params, case: str, eta: float, strobe_sig: str) -> Params:
    kw = _ensure_k_weights(base.kernel_weights)
    overrides = {
        "p6_on": True,
        "eta_drive": base.eta_drive,
        "eta": eta,
        "strobe_on": True,
        "strobe_signature": strobe_sig,
        "B_k": max(2, base.B_k),
        "radius_k": max(2, base.radius_k),
        "l_k": max(3, base.l_k),
        "kernel_weights": kw,
    }
    overrides["p3_on"] = case == "combo_protocol"
    params = Params.from_dict(base, overrides)
    if params.B_k <= 0 or params.radius_k <= 0 or params.l_k <= 0:
        raise ValueError("Phase5 requires K coupling enabled (B_k, radius_k, l_k > 0)")
    if kw["k_local"] <= 0 or kw["k_neighbor_trade"] <= 0:
        raise ValueError("Phase5 requires K kernels enabled")
    return params


def _case_cycle() -> List[str]:
    return _cycle_list()


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
    k_drive_mean_min: float,
    rel_change_min: float,
    cycle: List[str],
    control_stats: Dict[int, Dict[str, float]] | None,
    start_total: float,
    max_seconds_total: float,
    max_seconds_per_run: float,
    weights_mode: str,
    stop_on_seed1_fail: bool,
    resume: bool,
) -> List[Dict[str, Any]]:
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
    progress_path = case_dir / "progress.csv"
    raw_path = case_dir / "raw.csv"
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,weights_mode,step,window_index,k_drive_ep_window,strobe_current_l2,"
                "acceptedFrac,window_proposals,strobe_transitions,min_strobe_transitions_used,"
                "strobe_unique,strobe_edges,pass\n"
            )

    raw_rows: List[Dict[str, Any]] = []
    completed_seeds: set[int] = set()
    seed1_status: str | None = None
    if resume and raw_path.exists():
        with raw_path.open("r", encoding="utf-8", newline="") as rh:
            reader = csv.DictReader(rh)
            for row in reader:
                raw_rows.append(row)
                try:
                    seed_val = int(row.get("seed", 0))
                except (TypeError, ValueError):
                    continue
                status_val = row.get("status", "")
                if status_val and status_val != "RUNNING":
                    completed_seeds.add(seed_val)
                    if seed_val == 1:
                        seed1_status = status_val
    if stop_on_seed1_fail and seed1_status and seed1_status != "PASS_EARLY":
        print(
            f"STOP_REASON=SEED1_FAIL_{case.upper()} "
            f"seed=1 status={seed1_status} path={raw_path}"
        )
        return raw_rows
    for seed in seeds:
        if seed in completed_seeds:
            continue
        if time.monotonic() - start_total > max_seconds_total:
            print("TOTAL TIME CAP HIT")
            break
        diag_state = None
        k_drive_rates: List[float] = []
        current_l2_rates: List[float] = []
        accepts: List[float] = []
        status = "RUNNING"
        print(
            "STROBE_TRANSITIONS_THRESH "
            f"case={case} seed={seed} requested={min_strobe_transitions} "
            f"used={min_transitions_used} window_steps={window_steps} cycle_len={cycle_len}"
        )
        jsonl_path = jsonl_dir / f"{case}_seed{seed}.jsonl"
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        progress_handle = progress_path.open("a", encoding="utf-8")
        run_start = time.monotonic()

        def report_cb(state, step, ep_ledger, accepted_frac):
            nonlocal diag_state, status
            if len(k_drive_rates) >= max_windows or status != "RUNNING":
                return
            snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
            k_drive = _k_drive_rate(snapshot)
            transitions = int(snapshot.get("strobe_transitions_window", 0))
            uniq = int(snapshot.get("strobe_unique_states_window", 0))
            edges = int(snapshot.get("strobe_bidirectional_edges_window", 0))
            current_l2 = float(snapshot.get("strobe_current_l2_window", 0.0))
            snapshot["k_drive_ep_window"] = k_drive
            snapshot["min_strobe_transitions_used"] = min_transitions_used
            jsonl_handle.write(to_json_line(snapshot) + "\n")
            jsonl_handle.flush()
            progress_handle.write(
                f"{case},{seed},{weights_mode},{snapshot['step']},{len(k_drive_rates)+1},"
                f"{k_drive},{current_l2},{snapshot.get('acceptedFrac', 0.0)},"
                f"{snapshot.get('window_proposals', 0)},{transitions},{min_transitions_used},"
                f"{uniq},{edges},{step <= burn_steps}\n"
            )
            progress_handle.flush()
            os.fsync(progress_handle.fileno())

            if step <= burn_steps:
                return

            k_drive_rates.append(k_drive)
            current_l2_rates.append(current_l2)
            accepts.append(float(snapshot.get("acceptedFrac", 0.0)))

            if uniq < min_strobe_unique or edges < min_strobe_bidir or transitions < min_transitions_used:
                status = "FAIL_STROBE_SPARSE"
                return

            if len(k_drive_rates) >= min_windows:
                k_drive_mean = _summarize_tail(k_drive_rates, last_m)
                accept_mean = _summarize_tail(accepts, last_m)
                if accept_mean < accept_min:
                    status = "FAIL_ACCEPT"
                    return
                if case == "combo_control":
                    if k_drive_mean >= k_drive_mean_min:
                        status = "PASS_EARLY"
                else:
                    if not control_stats or seed not in control_stats:
                        status = "FAIL_CONTROL_MISSING"
                        return
                    control_l2 = float(control_stats[seed]["current_l2_mean_last_m"])
                    current_l2_mean = _summarize_tail(current_l2_rates, last_m)
                    rel_change = abs(current_l2_mean - control_l2) / max(control_l2, 1e-12)
                    if k_drive_mean >= k_drive_mean_min and rel_change >= rel_change_min:
                        status = "PASS_EARLY"

            if status == "RUNNING" and time.monotonic() - run_start > max_seconds_per_run:
                status = "FAIL_TIME"

        def stop_cb(state, step, ep_ledger, accepted_frac):
            return status != "RUNNING" or len(k_drive_rates) >= max_windows

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

        if status == "RUNNING":
            status = "FAIL_MAX_WINDOWS"

        k_drive_mean = _summarize_tail(k_drive_rates, last_m)
        current_l2_mean = _summarize_tail(current_l2_rates, last_m)
        accept_mean = _summarize_tail(accepts, last_m)
        control_l2_ref = None
        rel_change = 0.0
        if control_stats and seed in control_stats:
            control_l2_ref = float(control_stats[seed]["current_l2_mean_last_m"])
            rel_change = abs(current_l2_mean - control_l2_ref) / max(control_l2_ref, 1e-12)

        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "status": status,
                "windows_used": len(k_drive_rates),
                "k_drive_mean_last_m": k_drive_mean,
                "strobe_current_l2_mean_last_m": current_l2_mean,
                "acceptedFracWindowMean": accept_mean,
                "control_current_l2_ref": control_l2_ref,
                "rel_change": rel_change,
                "weights_mode": weights_mode,
                "min_strobe_transitions_used": min_transitions_used,
            }
        )
        print(
            f"SUMMARY case={case} seed={seed} status={status} windows={len(k_drive_rates)} "
            f"k_drive_mean={k_drive_mean} rel_change={rel_change} accept={accept_mean}"
        )

        if stop_on_seed1_fail and seed == 1 and status != "PASS_EARLY":
            break

    if raw_rows:
        fieldnames = list(raw_rows[0].keys())
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_rows:
                writer.writerow(row)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 P3+P6 combo runner v1")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", default="scripts/params/phase2_drive_k_balanced_v6.json")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--strobe-signature", default="mag_stag")
    parser.add_argument("--out-dir", default=".tmp/phase5_p3p6_combo_v1")
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
    parser.add_argument("--k-drive-mean-min", type=float, default=1e-3)
    parser.add_argument("--rel-change-min", type=float, default=0.05)
    parser.add_argument("--max-seconds-total", type=float, default=6600)
    parser.add_argument("--max-seconds-per-run", type=float, default=900)
    parser.add_argument("--preset-out", default="scripts/params/phase5_p3p6_combo_balanced_v1.json")
    parser.add_argument(
        "--match-control-cycle-weights",
        dest="match_control_cycle_weights",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-match-control-cycle-weights",
        dest="match_control_cycle_weights",
        action="store_false",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preset = _load_preset(Path(args.preset))
    base = _as_params(preset, {"device": args.device})

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if 1 in seeds:
        seeds = [1] + [s for s in seeds if s != 1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.monotonic()
    cycle = _case_cycle()

    control_params = build_case_params(base, "combo_control", eta=args.eta, strobe_sig=args.strobe_signature)
    weights_mode = "preset"
    if args.match_control_cycle_weights:
        matched = _match_cycle_weights(control_params.kernel_weights, cycle)
        matched = _ensure_k_weights(matched)
        control_params = Params.from_dict(control_params, {"kernel_weights": matched})
        weights_mode = "matched_cycle"

    control_rows = run_case(
        case="combo_control",
        params=control_params,
        seeds=seeds,
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
        k_drive_mean_min=args.k_drive_mean_min,
        rel_change_min=args.rel_change_min,
        cycle=cycle,
        control_stats=None,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        weights_mode=weights_mode,
        stop_on_seed1_fail=True,
        resume=args.resume,
    )

    if not control_rows or any(r["seed"] == 1 and r["status"] != "PASS_EARLY" for r in control_rows):
        print(
            "STOP_REASON=SEED1_FAIL_CONTROL "
            f"metrics={control_rows[0] if control_rows else {}} "
            f"path={out_dir / 'combo_control'}"
        )
        return

    control_stats: Dict[int, Dict[str, float]] = {}
    for r in control_rows:
        control_stats[r["seed"]] = {
            "current_l2_mean_last_m": float(r.get("strobe_current_l2_mean_last_m", 0.0))
        }

    proto_params = build_case_params(base, "combo_protocol", eta=args.eta, strobe_sig=args.strobe_signature)
    proto_rows = run_case(
        case="combo_protocol",
        params=proto_params,
        seeds=seeds,
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
        k_drive_mean_min=args.k_drive_mean_min,
        rel_change_min=args.rel_change_min,
        cycle=cycle,
        control_stats=control_stats,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        weights_mode="protocol_cycle",
        stop_on_seed1_fail=True,
        resume=args.resume,
    )

    if not proto_rows or any(r["seed"] == 1 and r["status"] != "PASS_EARLY" for r in proto_rows):
        print(
            "STOP_REASON=SEED1_FAIL_PROTOCOL "
            f"metrics={proto_rows[0] if proto_rows else {}} "
            f"path={out_dir / 'combo_protocol'}"
        )
        return

    control_pass = sum(1 for r in control_rows if r["status"] == "PASS_EARLY")
    proto_pass = sum(1 for r in proto_rows if r["status"] == "PASS_EARLY")
    overall_pass = control_pass >= 2 and proto_pass >= 2

    report_path = out_dir / "PHASE5_P3P6_COMBO_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 5 P3+P6 combo v1\n\n")
        fh.write(f"preset: {args.preset}\neta={args.eta}\n\n")
        fh.write("| seed | control_status | protocol_status | k_drive_control | k_drive_protocol | ")
        fh.write("control_l2 | protocol_l2 | rel_change | accept_control | accept_protocol |\n")
        fh.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        control_map = {r["seed"]: r for r in control_rows}
        proto_map = {r["seed"]: r for r in proto_rows}
        for seed in seeds:
            c = control_map.get(seed, {})
            p = proto_map.get(seed, {})
            fh.write(
                f"| {seed} | {c.get('status','')} | {p.get('status','')} | "
                f"{c.get('k_drive_mean_last_m',0.0):.6g} | {p.get('k_drive_mean_last_m',0.0):.6g} | "
                f"{c.get('strobe_current_l2_mean_last_m',0.0):.6g} | {p.get('strobe_current_l2_mean_last_m',0.0):.6g} | "
                f"{p.get('rel_change',0.0):.6g} | {c.get('acceptedFracWindowMean',0.0):.6g} | "
                f"{p.get('acceptedFracWindowMean',0.0):.6g} |\n"
            )
        fh.write("\n")
        fh.write(f"control_pass={control_pass} protocol_pass={proto_pass} overall_pass={overall_pass}\n")

    if overall_pass and args.preset_out:
        preset_out = Path(args.preset_out)
        preset_out.parent.mkdir(parents=True, exist_ok=True)
        config_id = f"p3p6_combo_eta{args.eta:.2f}"
        preset_payload = _load_preset(Path(args.preset))
        preset_payload.update(
            {
                "p3_on": True,
                "p6_on": True,
                "eta": args.eta,
                "strobe_signature": args.strobe_signature,
                "B_k": control_params.B_k,
                "radius_k": control_params.radius_k,
                "l_k": control_params.l_k,
                "kernel_weights": control_params.kernel_weights,
                "config_id": config_id,
                "pass": True,
            }
        )
        with preset_out.open("w", encoding="utf-8") as handle:
            json.dump(preset_payload, handle, indent=2, sort_keys=True)
        print(
            f"PRESET_SELECTED config_id={config_id} pass=true reason=PASS_FILTER path={preset_out}"
        )
    elif not overall_pass:
        print("PRESET_SELECTED pass=false reason=FAIL_FILTER")

    print("STATUS_COUNTS combo_control:", {r["status"]: 1 for r in control_rows})
    print("STATUS_COUNTS combo_protocol:", {r["status"]: 1 for r in proto_rows})


if __name__ == "__main__":
    main()
