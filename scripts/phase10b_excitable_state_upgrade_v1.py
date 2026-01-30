#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import sys

from ratchet_gpu.clockwork import fabric_scores
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim
from ratchet_gpu.state import State


def _load_preset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with path.open() as fh:
        return json.load(fh)


def _as_params(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Params:
    data = {k: v for k, v in preset.items() if k not in {"config_id", "pass", "note"}}
    data.update(overrides)
    if isinstance(data.get("shape"), list):
        data["shape"] = tuple(data["shape"])
    if isinstance(data.get("kernel_weights"), dict):
        data["kernel_weights"] = dict(data["kernel_weights"])
    data.pop("w_neighbor_weight", None)
    return Params(**data)


def _parse_seeds(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_windows <= 0:
        raise ValueError("max_windows must be > 0")
    if args.last_m <= 0 or args.last_m > args.max_windows:
        raise ValueError("last_m must be in [1, max_windows]")
    if args.excited_frac_min < 0.0 or args.excited_frac_max > 1.0:
        raise ValueError("excited_frac bounds must be within [0,1]")
    if args.excited_frac_min >= args.excited_frac_max:
        raise ValueError("excited_frac_min must be < excited_frac_max")


def _excitable_kernel_weights() -> Dict[str, float]:
    return {
        "excitable_color0": 1.0,
        "excitable_color1": 1.0,
    }


def _write_report(
    rows: List[Dict[str, Any]],
    report_path: Path,
    command: str,
    *,
    fabric_min: float,
    delta_min: float,
    r2_min: float,
    excited_frac_min: float,
    excited_frac_max: float,
) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 10b excitable state sanity v1\n\n")
        fh.write("## Command\n\n")
        fh.write(f"`{command}`\n\n")
        fh.write("## Thresholds\n\n")
        fh.write(
            f"- fabric_min={fabric_min}\n"
            f"- delta_min={delta_min}\n"
            f"- r2_min={r2_min}\n"
            f"- excited_frac_min={excited_frac_min}\n"
            f"- excited_frac_max={excited_frac_max}\n\n"
        )
        fh.write("## Summary\n\n")
        fh.write(
            "| seed | status | coupled_best | decoupled_best | delta | best_metric | "
            "excited_frac_mean | travel_r2 |\n"
        )
        fh.write("| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |\n")
        for row in rows:
            coupled_best = float(row["coupled_best"])
            decoupled_best = float(row["decoupled_best"])
            delta = float(row["delta"])
            exc_mean = float(row["excited_frac_mean"])
            travel_r2 = float(row["travel_r2"])
            fh.write(
                f"| {row['seed']} | {row['status']} | {coupled_best:.6g} | "
                f"{decoupled_best:.6g} | {delta:.6g} | {row['best_metric']} | "
                f"{exc_mean:.6g} | {travel_r2:.6g} |\n"
            )


def _compute_best(frames: List[np.ndarray], last_m: int, omega_min: float, r2_min: float) -> Tuple[str, float, float]:
    if len(frames) < 3:
        return "phase", 0.0, 0.0
    tail = np.stack(frames[-last_m:], axis=0) if len(frames) >= last_m else np.stack(frames, axis=0)
    scores = fabric_scores(tail, omega_min=omega_min)
    if scores["travel_score"] >= scores["phase_score"]:
        return "travel", float(scores["fabric_score"]), float(scores["travel_r2"])
    return "phase", float(scores["fabric_score"]), float(scores["travel_r2"])


def _run_case(
    case: str,
    params: Params,
    seed: int,
    out_dir: Path,
    burn_steps: int,
    window_steps: int,
    max_windows: int,
    snapshot_every: int,
    last_m: int,
    omega_min: float,
    r2_min: float,
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
) -> Dict[str, Any]:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = case_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.csv"
    progress_path = case_dir / "progress.csv"

    if not raw_path.exists():
        with raw_path.open("w", encoding="utf-8") as fh:
            fh.write(
                "case,seed,window_index,step,accept_window,excited_frac,ep_rate\n"
            )
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as fh:
            fh.write("case,seed,window_index,step\n")

    run_start = time.monotonic()
    burn_summary = run_sim(
        params,
        seed=seed,
        steps=burn_steps,
        report_every=0,
        return_state=True,
        protocol_cycle=["excitable_color0", "excitable_color1"],
    )
    state = burn_summary["state"]
    rng_state = burn_summary["rng_state"]

    frames_phase: List[np.ndarray] = []
    frames_excited: List[np.ndarray] = []
    window_idx = 0

    def report_cb(st: State, step: int, ledger: Dict[str, Any], accepted_frac: float) -> None:
        nonlocal window_idx
        window_idx += 1
        exc_state = st.sigma[0].detach().cpu().numpy().astype(np.int8)
        exc_state = exc_state.reshape(st.lattice.shape)
        exc_excited = (exc_state == 1).astype(np.float32)
        exc_phase = exc_state.astype(np.float32) * (2.0 * np.pi / 4.0)
        frames_phase.append(exc_phase)
        frames_excited.append(exc_excited)

        if window_idx % snapshot_every == 0:
            npz_path = npz_dir / f"seed{seed}_win{window_idx:04d}.npz"
            np.savez_compressed(
                npz_path,
                exc_state_l0=exc_state,
                exc_excited_l0=exc_excited,
                exc_phase_l0=exc_phase,
            )

        accept_window = float(ledger.get("window_accept_frac", 0.0))
        ep_rate = float(ledger.get("ep_rate_exact_window", 0.0))
        excited_frac = float(exc_excited.mean())

        with raw_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{case},{seed},{window_idx},{step},{accept_window:.6g},{excited_frac:.6g},{ep_rate:.6g}\n"
            )
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{case},{seed},{window_idx},{step}\n")

    def stop_cb(st: State, step: int, ledger: Dict[str, Any], accepted_frac: float) -> bool:
        if time.monotonic() - run_start > max_seconds_per_run:
            return True
        if time.monotonic() - start_total > max_seconds_total:
            return True
        return False

    run_sim(
        params,
        seed=seed,
        steps=window_steps * max_windows,
        report_every=window_steps,
        report_callback=report_cb,
        stop_callback=stop_cb,
        initial_state=state,
        initial_rng_state=rng_state,
        protocol_cycle=["excitable_color0", "excitable_color1"],
    )

    metric, best_score, travel_r2 = _compute_best(
        frames_phase, last_m, omega_min, r2_min
    )
    excited_tail = frames_excited[-last_m:] if frames_excited else []
    if excited_tail:
        exc_mean = float(np.stack(excited_tail).mean())
    else:
        exc_mean = 0.0

    return {
        "case": case,
        "seed": seed,
        "best_metric": metric,
        "best_score": best_score,
        "travel_r2": travel_r2,
        "excited_frac_mean": exc_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 10b excitable state sanity v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=int, default=50)
    parser.add_argument("--window-sweeps", type=int, default=40)
    parser.add_argument("--max-windows", type=int, default=80)
    parser.add_argument("--last-m", type=int, default=30)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--exc-init-frac", type=float, default=0.02)
    parser.add_argument("--exc-p-spont", type=float, default=1e-3)
    parser.add_argument("--exc-theta", type=float, default=1.0)
    parser.add_argument("--exc-beta", type=float, default=2.0)
    parser.add_argument("--exc-p-recover", type=float, default=1.0)
    parser.add_argument("--fabric-min", type=float, default=0.05)
    parser.add_argument("--delta-min", type=float, default=0.02)
    parser.add_argument("--r2-min", type=float, default=0.60)
    parser.add_argument("--excited-frac-min", type=float, default=0.01)
    parser.add_argument("--excited-frac-max", type=float, default=0.50)
    parser.add_argument("--max-seconds-total", type=float, default=3600)
    parser.add_argument("--max-seconds-per-run", type=float, default=1200)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _validate_args(args)

    preset = _load_preset(args.preset)
    seeds = _parse_seeds(args.seeds)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle = ["excitable_color0", "excitable_color1"]
    steps_per_sweep = len(cycle)
    burn_steps = args.burn_in_sweeps * steps_per_sweep
    window_steps = args.window_sweeps * steps_per_sweep

    base_params = _as_params(
        preset,
        {
            "sigma_mode": "excitable4",
            "p3_on": True,
            "p6_on": False,
            "strobe_on": False,
            "kernel_weights": _excitable_kernel_weights(),
            "exc_init_frac": args.exc_init_frac,
            "exc_p_spont": args.exc_p_spont,
            "exc_theta": args.exc_theta,
            "exc_beta": args.exc_beta,
            "exc_p_recover": args.exc_p_recover,
            "device": args.device,
        },
    )

    rows: List[Dict[str, Any]] = []
    agg_path = out_dir / "agg.csv"
    existing_rows: Dict[str, Dict[str, Any]] = {}
    if args.resume and agg_path.exists():
        with agg_path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("seed") is not None:
                    existing_rows[str(row["seed"])] = row
    start_total = time.monotonic()

    for seed in seeds:
        if args.resume and str(seed) in existing_rows:
            row = existing_rows[str(seed)]
            if row.get("status"):
                rows.append(row)
                continue
        coupled_params = base_params
        decoupled_params = Params.from_dict(base_params, {"B_w": 0})

        coupled = _run_case(
            "coupled",
            coupled_params,
            seed,
            out_dir,
            burn_steps,
            window_steps,
            args.max_windows,
            args.snapshot_every_windows,
            args.last_m,
            omega_min=0.1,
            r2_min=args.r2_min,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
        )
        decoupled = _run_case(
            "decoupled",
            decoupled_params,
            seed,
            out_dir,
            burn_steps,
            window_steps,
            args.max_windows,
            args.snapshot_every_windows,
            args.last_m,
            omega_min=0.1,
            r2_min=args.r2_min,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
        )

        best_coupled = coupled["best_score"]
        best_decoupled = decoupled["best_score"]
        delta = best_coupled - best_decoupled
        metric = coupled["best_metric"]
        travel_r2 = coupled["travel_r2"] if metric == "travel" else 0.0
        exc_mean = coupled["excited_frac_mean"]

        status = "PASS"
        if not (args.excited_frac_min <= exc_mean <= args.excited_frac_max):
            status = "FAIL_EXCITED_RANGE"
        elif best_coupled < args.fabric_min:
            status = "FAIL_FABRIC_MIN"
        elif delta < args.delta_min:
            status = "FAIL_DELTA_MIN"
        elif metric == "travel" and travel_r2 < args.r2_min:
            status = "FAIL_TRAVEL_R2"

        rows.append(
            {
                "seed": seed,
                "status": status,
                "coupled_best": best_coupled,
                "decoupled_best": best_decoupled,
                "delta": delta,
                "best_metric": metric,
                "excited_frac_mean": exc_mean,
                "travel_r2": travel_r2,
            }
        )

        if seed == seeds[0] and status != "PASS":
            break

    with agg_path.open("w", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "status",
                "coupled_best",
                "decoupled_best",
                "delta",
                "best_metric",
                "excited_frac_mean",
                "travel_r2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_path = out_dir / "PHASE10B_EXCITABLE_REPORT.md"
    _write_report(
        rows,
        report_path,
        " ".join(sys.argv),
        fabric_min=args.fabric_min,
        delta_min=args.delta_min,
        r2_min=args.r2_min,
        excited_frac_min=args.excited_frac_min,
        excited_frac_max=args.excited_frac_max,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
