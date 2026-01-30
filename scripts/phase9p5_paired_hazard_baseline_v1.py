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

import numpy as np
import torch

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.interventions import apply_sigma_flip, apply_sigma_randomize, parse_rect
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim, _cycle_list
from ratchet_gpu.spatial import compute_spatial_maps, finite_check
from ratchet_gpu.state import State

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
try:
    from phase1_null_screen_v4 import _expected_proposals_per_step  # type: ignore
except Exception:  # pragma: no cover
    def _expected_proposals_per_step(N: int, device: str, kernel_weights: Dict[str, float]) -> float:
        return float(N)


HEAVY_KEYS = {"strobe_current_map_items_window", "strobe_currents_window"}


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


def _parse_seeds(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_layers(value: str, total: int) -> List[int]:
    if value == "all":
        return list(range(total))
    return [int(x) for x in value.split(",") if x.strip()]


def _validate_hazard_schedule(start: int, duration: int, max_windows: int) -> None:
    if start < 1:
        raise ValueError("hazard_start_window must be >= 1")
    if duration < 1:
        raise ValueError("hazard_duration_windows must be >= 1")
    if start + duration - 1 > max_windows:
        raise ValueError("hazard window must fit within max_windows")


def _slim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    slim = dict(snapshot)
    for key in list(slim.keys()):
        if key in HEAVY_KEYS or key.endswith("_items_window"):
            slim.pop(key, None)
    return slim


def _region_mean(map_tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    data = map_tensor.to(dtype=torch.float32)
    if data.ndim == 3:
        data = data.mean(dim=0)
    region = data[mask]
    outside = data[~mask]
    region_mean = float(region.mean().item()) if region.numel() else 0.0
    outside_mean = float(outside.mean().item()) if outside.numel() else 0.0
    return region_mean, outside_mean


def _patchiness(map_tensor: torch.Tensor) -> float:
    data = map_tensor.to(dtype=torch.float32)
    return float(data.flatten().std(unbiased=False).item())


def _clone_state(state: State) -> State:
    return State(
        params=state.params,
        lattice=state.lattice,
        R_W=state.R_W,
        R_K=state.R_K,
        sigma=state.sigma.clone(),
        n=state.n.clone(),
        s=state.s.clone(),
        W=state.W.clone(),
        K=state.K.clone(),
        color_indices=state.color_indices,
    )


def _mean_at(vals: List[float], idx: List[int]) -> float:
    if not idx:
        return 0.0
    return float(np.mean([vals[i - 1] for i in idx]))


def _paired_metrics(
    baseline_vals: Dict[str, List[float]],
    hazard_vals: Dict[str, List[float]],
    hazard_start: int,
    hazard_duration: int,
    total_windows: int,
    pre_windows: int = 2,
    post_windows: int = 3,
) -> Dict[str, float]:
    pre_idx = list(range(max(1, hazard_start - pre_windows), hazard_start))
    haz_idx = list(range(hazard_start, min(total_windows, hazard_start + hazard_duration - 1) + 1))
    post_idx = list(range(max(1, total_windows - post_windows + 1), total_windows + 1))

    hazard_mismatch = hazard_vals["mismatch_region"]
    baseline_mismatch = baseline_vals["mismatch_region"]
    hazard_bias = hazard_vals["k_axis_bias_focus"]
    baseline_bias = baseline_vals["k_axis_bias_focus"]

    spike = max(
        0.0,
        max(hazard_mismatch[i - 1] - baseline_mismatch[i - 1] for i in haz_idx),
    )
    realloc = max(
        0.0,
        max(hazard_bias[i - 1] - baseline_bias[i - 1] for i in haz_idx),
    )

    haz_pre = _mean_at(hazard_mismatch, pre_idx)
    haz_post = _mean_at(hazard_mismatch, post_idx)
    base_pre = _mean_at(baseline_mismatch, pre_idx)
    base_post = _mean_at(baseline_mismatch, post_idx)
    recovery = (haz_pre - haz_post) - (base_pre - base_post)

    return {
        "spike_paired": spike,
        "realloc_paired": realloc,
        "recovery_paired": recovery,
    }


def _write_report(rows: List[Dict[str, Any]], report_path: Path, command: str) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 9.5 paired hazard vs baseline v1\n\n")
        fh.write("## Command\n\n")
        fh.write(f"`{command}`\n\n")
        fh.write("## Summary\n\n")
        fh.write("| seed | status | spike_paired | realloc_paired | recovery_paired |\n")
        fh.write("| ---: | --- | ---: | ---: | ---: |\n")
        for row in rows:
            fh.write(
                f"| {row['seed']} | {row['status']} | {row['spike_paired']:.6g} | "
                f"{row['realloc_paired']:.6g} | {row['recovery_paired']:.6g} |\n"
            )


def run_case(
    case: str,
    params: Params,
    seed: int,
    out_dir: Path,
    window_sweeps: float,
    max_windows: int,
    snapshot_every: int,
    maps: List[str],
    accept_min: float,
    hazard_start: int,
    hazard_duration: int,
    hazard_rect: str | None,
    hazard_sigma: str,
    hazard_layers: List[int],
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
    cycle: List[str],
    resume: bool,
    apply_hazard: bool,
    hazard_refresh_each_window: bool,
    initial_state: State,
    initial_rng_state: torch.Tensor,
) -> Dict[str, Any]:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.csv"
    summary_path = case_dir / "agg.csv"
    progress_path = case_dir / "progress.csv"
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = case_dir / "npz"
    maps_dir.mkdir(parents=True, exist_ok=True)
    params_path = case_dir / "effective_params.json"

    params_payload = dict(params.__dict__)
    params_payload["shape"] = list(params_payload.get("shape", ()))
    params_payload["device"] = str(params_payload.get("device", ""))
    params_path.write_text(json.dumps(params_payload, indent=2, sort_keys=True))

    if resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as sh:
            rows = list(csv.DictReader(sh))
        for row in rows:
            if str(row.get("seed")) == str(seed) and row.get("status") not in {"RUNNING", ""}:
                return row

    if not raw_path.exists():
        with raw_path.open("w", encoding="utf-8") as rh:
            rh.write(
                "case,seed,window_index,step,hazard_active,ep_rate,accept_window,k_drive,"
                "mismatch_abs_mean,mismatch_region,mismatch_outside,"
                "w_mass_region,w_mass_outside,k_entropy_region,k_entropy_outside,"
                "k_axis_bias_abs_region_if0,k_axis_bias_abs_outside_if0,k_axis_bias_abs_focus_if0,"
                "k_axis_bias_abs_focus\n"
            )
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,window_index,step,hazard_active,ep_rate,accept_window,k_drive,"
                "mismatch_region,mismatch_outside,k_axis_bias_abs_focus_if0\n"
            )

    if len(params.shape) != 2:
        raise ValueError("Phase 9.5 requires a 2D lattice shape")
    N = math.prod(params.shape)
    expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    window_steps = int(math.ceil(window_sweeps * N / expected_props))

    hazard_mask = None
    hazard_idx = None
    if hazard_rect is not None:
        hazard_mask, hazard_idx = parse_rect(hazard_rect, params.shape)
        hazard_mask = hazard_mask.to(device=params.device)
        hazard_idx = hazard_idx.to(device=params.device)
    if apply_hazard and hazard_rect is None:
        raise ValueError("hazard_rect must be set when hazard is enabled")

    window_idx = 0
    hazard_active_next = False
    status = "RUNNING"
    accept_low = 0
    diag_state = None
    run_start = time.monotonic()
    sigma_backup = None

    mismatch_region_vals: List[float] = []
    mismatch_outside_vals: List[float] = []
    k_axis_bias_focus_vals: List[float] = []

    jsonl_path = jsonl_dir / f"seed{seed}.jsonl"
    jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    progress_handle = progress_path.open("a", encoding="utf-8")

    def _apply_hazard(state: Any) -> None:
        if hazard_sigma == "flip":
            apply_sigma_flip(state, hazard_idx, hazard_layers)
        elif hazard_sigma == "random":
            apply_sigma_randomize(state, hazard_idx, hazard_layers)

    def _backup_sigma(state: Any) -> None:
        nonlocal sigma_backup
        if sigma_backup is not None or hazard_idx is None:
            return
        sigma_backup = state.sigma[hazard_layers][:, hazard_idx].clone()

    def _restore_sigma(state: Any) -> None:
        nonlocal sigma_backup
        if sigma_backup is None or hazard_idx is None:
            return
        state.sigma[hazard_layers][:, hazard_idx] = sigma_backup
        sigma_backup = None

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal status, accept_low, hazard_active_next, diag_state, window_idx
        if status != "RUNNING" or window_idx >= max_windows:
            return

        snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
        window_idx += 1
        hazard_active = hazard_active_next

        window_props = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
        accept_window = float(ep_ledger.get("window_accepted", 0)) / window_props if window_props else 0.0
        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        mismatch_abs = snapshot.get("mismatch_abs_mean")
        k_drive = float(
            sum(float(snapshot.get("ep_rate_by_kernel_proposal_window", {}).get(k, 0.0))
                for k in ("k_local", "k_neighbor_trade", "k_p5_exchange"))
        )

        maps_dict = compute_spatial_maps(state, maps)
        ok, bad = finite_check(maps_dict)
        if not ok:
            status = f"FAIL_NAN_MAP:{','.join(bad)}"
            return

        mismatch_grid = maps_dict.get("mismatch")
        k_axis_bias_grid = maps_dict.get("k_axis_bias")

        region_stats = {
            "mismatch_region": None,
            "mismatch_outside": None,
            "k_axis_bias_region_if0": None,
            "k_axis_bias_outside_if0": None,
        }
        k_axis_bias_focus_if0 = 0.0
        k_axis_bias_focus = 0.0

        if hazard_mask is not None:
            if mismatch_grid is not None:
                region_stats["mismatch_region"], region_stats["mismatch_outside"] = _region_mean(
                    mismatch_grid, hazard_mask
                )
            if k_axis_bias_grid is not None and k_axis_bias_grid.shape[0] > 0:
                bias0 = k_axis_bias_grid[0].abs()
                region_stats["k_axis_bias_region_if0"], region_stats["k_axis_bias_outside_if0"] = _region_mean(
                    bias0, hazard_mask
                )
                k_axis_bias_focus_if0 = (
                    (region_stats["k_axis_bias_region_if0"] or 0.0)
                    - (region_stats["k_axis_bias_outside_if0"] or 0.0)
                )
                k_axis_bias_focus = k_axis_bias_focus_if0

        mismatch_region_vals.append(float(region_stats["mismatch_region"] or 0.0))
        mismatch_outside_vals.append(float(region_stats["mismatch_outside"] or 0.0))
        k_axis_bias_focus_vals.append(float(k_axis_bias_focus))

        if accept_window < accept_min:
            accept_low += 1
        else:
            accept_low = 0
        if accept_low >= 5:
            status = "FAIL_ACCEPT_COLLAPSE"

        if time.monotonic() - run_start > max_seconds_per_run:
            status = "FAIL_TIME"
        if time.monotonic() - start_total > max_seconds_total:
            status = "FAIL_TIME"

        slim_snapshot = _slim_snapshot(snapshot)
        slim_snapshot.update(
            {
                "case": case,
                "seed": seed,
                "window_index": window_idx,
                "hazard_active": hazard_active,
                "ep_rate_exact_window": ep_rate,
                "acceptedFracWindow": accept_window,
                "k_drive_ep_window": k_drive,
                "mismatch_abs_mean": mismatch_abs,
                "mismatch_region": region_stats["mismatch_region"],
                "mismatch_outside": region_stats["mismatch_outside"],
                "k_axis_bias_abs_region_if0": region_stats["k_axis_bias_region_if0"],
                "k_axis_bias_abs_outside_if0": region_stats["k_axis_bias_outside_if0"],
                "k_axis_bias_abs_focus_if0": k_axis_bias_focus_if0,
                "k_axis_bias_abs_focus": k_axis_bias_focus,
            }
        )
        jsonl_handle.write(to_json_line(slim_snapshot) + "\n")
        jsonl_handle.flush()

        with raw_path.open("a", encoding="utf-8") as rh:
            rh.write(
                f"{case},{seed},{window_idx},{step},{hazard_active},{ep_rate},{accept_window},{k_drive},"
                f"{mismatch_abs},{region_stats['mismatch_region']},{region_stats['mismatch_outside']},"
                f"{region_stats['k_axis_bias_region_if0']},{region_stats['k_axis_bias_outside_if0']},"
                f"{k_axis_bias_focus_if0},{k_axis_bias_focus}\n"
            )

        progress_handle.write(
            f"{case},{seed},{window_idx},{step},{hazard_active},{ep_rate},{accept_window},{k_drive},"
            f"{region_stats['mismatch_region']},{region_stats['mismatch_outside']},"
            f"{k_axis_bias_focus_if0}\n"
        )
        progress_handle.flush()

        npz_payload = {k: v.detach().cpu().numpy() for k, v in maps_dict.items()}
        np.savez(maps_dir / f"seed{seed}_win{window_idx:04d}.npz", **npz_payload)

        if apply_hazard:
            next_window_idx = window_idx + 1
            hazard_active_next_new = hazard_start <= next_window_idx <= (hazard_start + hazard_duration - 1)
            if hazard_active_next_new:
                if hazard_refresh_each_window or not hazard_active_next:
                    _backup_sigma(state)
                    _apply_hazard(state)
            elif hazard_active_next:
                _restore_sigma(state)
            hazard_active_next = hazard_active_next_new

    def stop_cb(*_args) -> bool:
        return status != "RUNNING" or window_idx >= max_windows

    run_sim(
        params,
        seed=seed,
        steps=window_steps * max_windows,
        report_every=window_steps,
        report_callback=report_cb,
        stop_callback=stop_cb,
        protocol_cycle=cycle,
        initial_state=initial_state,
        initial_rng_state=initial_rng_state,
    )

    jsonl_handle.close()
    progress_handle.close()

    if status == "RUNNING":
        status = "OK"

    metrics = {
        "case": case,
        "seed": seed,
        "status": status,
        "windows_completed": window_idx,
        "mismatch_region": mismatch_region_vals,
        "mismatch_outside": mismatch_outside_vals,
        "k_axis_bias_focus": k_axis_bias_focus_vals,
    }
    metrics["runtime_sec"] = time.monotonic() - run_start

    with summary_path.open("w", encoding="utf-8", newline="") as sh:
        writer = csv.DictWriter(sh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9.5 paired hazard vs baseline v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", default="scripts/params/meta_null_coupled_eta1.00_layers3.json")
    parser.add_argument("--out-dir", default=".tmp/phase9p5_paired_v1")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--hazard-start-window", type=int, default=5)
    parser.add_argument("--hazard-duration-windows", type=int, default=5)
    parser.add_argument("--hazard-rect", type=str, required=False)
    parser.add_argument("--hazard-sigma", choices=["random", "flip"], default="flip")
    parser.add_argument("--hazard-layers", default="0")
    parser.add_argument("--hazard-refresh-each-window", action="store_true")
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--spike-paired-min", type=float, default=0.005)
    parser.add_argument("--realloc-paired-min", type=float, default=0.005)
    parser.add_argument("--max-seconds-total", type=float, default=1200)
    parser.add_argument("--max-seconds-per-run", type=float, default=600)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)
    if not args.hazard_rect:
        raise ValueError("--hazard-rect is required for Phase 9.5")

    preset = _load_preset(Path(args.preset))
    base_overrides: Dict[str, Any] = {"device": args.device}
    params = _as_params(preset, base_overrides)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    hazard_layers = _parse_layers(args.hazard_layers, params.layers)
    hazard_layers = [layer for layer in hazard_layers if layer < params.layers]
    hazard_rect = args.hazard_rect

    cycle = _cycle_list()
    start_total = time.monotonic()
    report_rows: List[Dict[str, Any]] = []

    command = " ".join([str(x) for x in sys.argv])
    state_path = out_dir / "paired_state_seed.pt"

    for idx, seed in enumerate(seeds):
        if time.monotonic() - start_total > args.max_seconds_total:
            break

        N = math.prod(params.shape)
        expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
        burn_steps = int(math.ceil(args.burn_in_sweeps * N / expected_props))

        burn_summary = run_sim(
            params,
            seed=seed,
            steps=burn_steps,
            report_every=burn_steps,
            return_state=True,
        )
        state0 = burn_summary.get("state")
        rng_state = burn_summary.get("rng_state")
        if state0 is None or rng_state is None:
            raise RuntimeError("Failed to capture burn-in state")

        torch.save(
            {
                "sigma": state0.sigma.detach().cpu(),
                "n": state0.n.detach().cpu(),
                "s": state0.s.detach().cpu(),
                "W": state0.W.detach().cpu(),
                "K": state0.K.detach().cpu(),
                "rng_state": rng_state.detach().cpu(),
            },
            state_path,
        )

        baseline_metrics = run_case(
            "baseline",
            params,
            seed,
            out_dir,
            args.window_sweeps,
            args.max_windows,
            args.snapshot_every_windows,
            ["sigma", "w_mass", "w_entropy", "w_axis_bias", "k_entropy", "k_axis_bias", "mismatch"],
            args.accept_min,
            args.hazard_start_window,
            args.hazard_duration_windows,
            hazard_rect,
            args.hazard_sigma,
            hazard_layers,
            args.max_seconds_total,
            args.max_seconds_per_run,
            start_total,
            cycle,
            args.resume,
            apply_hazard=False,
            hazard_refresh_each_window=args.hazard_refresh_each_window,
            initial_state=_clone_state(state0),
            initial_rng_state=rng_state.clone(),
        )

        hazard_metrics = run_case(
            "hazard",
            params,
            seed,
            out_dir,
            args.window_sweeps,
            args.max_windows,
            args.snapshot_every_windows,
            ["sigma", "w_mass", "w_entropy", "w_axis_bias", "k_entropy", "k_axis_bias", "mismatch"],
            args.accept_min,
            args.hazard_start_window,
            args.hazard_duration_windows,
            hazard_rect,
            args.hazard_sigma,
            hazard_layers,
            args.max_seconds_total,
            args.max_seconds_per_run,
            start_total,
            cycle,
            args.resume,
            apply_hazard=True,
            hazard_refresh_each_window=args.hazard_refresh_each_window,
            initial_state=_clone_state(state0),
            initial_rng_state=rng_state.clone(),
        )

        total_windows = int(hazard_metrics["windows_completed"])
        baseline_vals = {
            "mismatch_region": baseline_metrics["mismatch_region"],
            "k_axis_bias_focus": baseline_metrics["k_axis_bias_focus"],
        }
        hazard_vals = {
            "mismatch_region": hazard_metrics["mismatch_region"],
            "k_axis_bias_focus": hazard_metrics["k_axis_bias_focus"],
        }
        paired = _paired_metrics(
            baseline_vals,
            hazard_vals,
            args.hazard_start_window,
            args.hazard_duration_windows,
            total_windows,
        )

        pass_seed = (
            paired["spike_paired"] >= args.spike_paired_min
            and paired["realloc_paired"] >= args.realloc_paired_min
        )
        status = "PASS" if pass_seed else "FAIL"
        report_rows.append(
            {
                "seed": seed,
                "status": status,
                "spike_paired": paired["spike_paired"],
                "realloc_paired": paired["realloc_paired"],
                "recovery_paired": paired["recovery_paired"],
            }
        )

        _write_report(report_rows, out_dir / "PHASE9P5_PAIRED_REPORT.md", command)

        if idx == 0 and not pass_seed:
            break

    if args.progress:
        print(f"PHASE9P5_REPORT={out_dir / 'PHASE9P5_PAIRED_REPORT.md'}")


if __name__ == "__main__":
    main()
