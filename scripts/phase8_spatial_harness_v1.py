#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.interventions import (
    apply_sigma_flip,
    apply_sigma_randomize,
    apply_w_lesion_redistribute,
    check_w_invariants,
    parse_rect,
)
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim, _cycle_list
from ratchet_gpu.spatial import compute_spatial_maps, finite_check

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


def _parse_maps(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_layers(value: str, total: int) -> List[int]:
    if value == "all":
        return list(range(total))
    return [int(x) for x in value.split(",") if x.strip()]


def _slim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    slim = dict(snapshot)
    for key in list(slim.keys()):
        if key in HEAVY_KEYS or key.endswith("_items_window"):
            slim.pop(key, None)
    return slim


def _k_drive_rate(snapshot: Dict[str, Any]) -> float:
    rates = snapshot.get("ep_rate_by_kernel_proposal_window", {})
    total = 0.0
    for key in K_KERNELS:
        total += float(rates.get(key, 0.0))
    return total


def _mean_ci(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ci = 1.96 * std / math.sqrt(len(values))
    return mean, ci


def _linear_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = np.arange(len(values), dtype=float)
    ys = np.array(values, dtype=float)
    denom = float(((xs - xs.mean()) ** 2).sum())
    if denom == 0:
        return 0.0
    return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / denom)


def _region_mean(map_tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    data = map_tensor.to(dtype=torch.float32)
    if data.ndim == 3:
        data = data.mean(dim=0)
    region = data[mask]
    outside = data[~mask]
    region_mean = float(region.mean().item()) if region.numel() else 0.0
    outside_mean = float(outside.mean().item()) if outside.numel() else 0.0
    return region_mean, outside_mean


def run_case(
    case: str,
    params: Params,
    seed: int,
    out_dir: Path,
    burn_sweeps: float,
    window_sweeps: float,
    max_windows: int,
    last_m: int,
    accept_min: float,
    maps: List[str],
    snapshot_every: int,
    lesion_window: int,
    lesion_mask: torch.Tensor | None,
    lesion_idx: torch.Tensor | None,
    lesion_sigma: str,
    lesion_w_frac: float,
    lesion_layers: str,
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
    cycle: List[str],
    resume: bool,
) -> Dict[str, Any]:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.csv"
    summary_path = case_dir / "summary.csv"
    progress_path = case_dir / "progress.csv"
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = case_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    params_path = case_dir / "effective_params.json"

    params_payload = dict(params.__dict__)
    params_payload["shape"] = list(params_payload.get("shape", ()))
    params_payload["device"] = str(params_payload.get("device", ""))
    params_path.write_text(json.dumps(params_payload, indent=2, sort_keys=True))

    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    if resume and raw_path.exists():
        with raw_path.open("r", encoding="utf-8", newline="") as rh:
            raw_rows = list(csv.DictReader(rh))
    if resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as sh:
            summary_rows = list(csv.DictReader(sh))
        for row in summary_rows:
            if str(row.get("seed")) == str(seed) and row.get("status") not in {"RUNNING", ""}:
                return row

    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,step,window_index,ep_rate,accept_window,k_drive,mismatch_abs_mean,"
                "w_mass_region,w_mass_outside,k_entropy_region,k_entropy_outside,"
                "mismatch_region,mismatch_outside,lesion_applied\n"
            )

    if len(params.shape) != 2:
        raise ValueError("Phase 8 requires a 2D lattice shape")
    N = math.prod(params.shape)
    expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected_props))
    window_steps = int(math.ceil(window_sweeps * N / expected_props))

    ep_rates: List[float] = []
    accept_rates: List[float] = []
    k_drive_rates: List[float] = []
    mismatch_vals: List[float] = []
    strobe_vals: List[float] = []
    status = "RUNNING"
    accept_low = 0
    lesion_applied = False
    diag_state = None
    window_idx = 0

    jsonl_path = jsonl_dir / f"seed{seed}.jsonl"
    jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    progress_handle = progress_path.open("a", encoding="utf-8")
    run_start = time.monotonic()

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal status, accept_low, lesion_applied, diag_state, window_idx
        if status != "RUNNING" or window_idx >= max_windows:
            return

        snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
        window_props = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
        accept_window = float(ep_ledger.get("window_accepted", 0)) / window_props if window_props else 0.0
        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        mismatch_abs = snapshot.get("mismatch_abs_mean")
        k_drive = _k_drive_rate(snapshot)
        strobe_l2 = snapshot.get("strobe_current_l2_window")

        is_burn = step <= burn_steps
        if not is_burn:
            window_idx += 1

        region_stats = {
            "w_mass_region": None,
            "w_mass_outside": None,
            "k_entropy_region": None,
            "k_entropy_outside": None,
            "mismatch_region": None,
            "mismatch_outside": None,
        }

        if not is_burn and (window_idx % snapshot_every == 0):
            maps_dict = compute_spatial_maps(state, maps)
            ok, bad = finite_check(maps_dict)
            if not ok:
                status = f"FAIL_NAN_MAP:{','.join(bad)}"
                return

            if lesion_mask is not None:
                if "w_mass" in maps_dict:
                    region_stats["w_mass_region"], region_stats["w_mass_outside"] = _region_mean(
                        maps_dict["w_mass"], lesion_mask
                    )
                if "k_entropy" in maps_dict:
                    region_stats["k_entropy_region"], region_stats["k_entropy_outside"] = _region_mean(
                        maps_dict["k_entropy"], lesion_mask
                    )
                if "mismatch" in maps_dict:
                    region_stats["mismatch_region"], region_stats["mismatch_outside"] = _region_mean(
                        maps_dict["mismatch"], lesion_mask
                    )

            npz_payload = {k: v.detach().cpu().numpy() for k, v in maps_dict.items()}
            np.savez(maps_dir / f"seed{seed}_win{window_idx:04d}.npz", **npz_payload)

            if lesion_window >= 0 and window_idx == lesion_window and not lesion_applied:
                if lesion_sigma == "flip":
                    apply_sigma_flip(state, lesion_idx, lesion_layers)
                elif lesion_sigma == "random":
                    apply_sigma_randomize(state, lesion_idx, lesion_layers)

                lesion_info = apply_w_lesion_redistribute(
                    state,
                    params,
                    lesion_idx,
                    layers=lesion_layers,
                    frac=lesion_w_frac,
                )
                ok_inv, msg = check_w_invariants(state, params)
                if not ok_inv:
                    status = "FAIL_W_INVARIANT"
                    return
                lesion_applied = True
                post_maps = compute_spatial_maps(state, maps)
                ok_post, bad_post = finite_check(post_maps)
                if not ok_post:
                    status = f"FAIL_NAN_MAP:{','.join(bad_post)}"
                    return
                post_payload = {k: v.detach().cpu().numpy() for k, v in post_maps.items()}
                post_payload["lesion_info"] = np.array([lesion_info.get("removed_tokens", 0)])
                np.savez(maps_dir / f"seed{seed}_post_lesion_win{window_idx:04d}.npz", **post_payload)

        slim = _slim_snapshot(snapshot)
        slim["acceptedFracWindow"] = accept_window
        slim["k_drive_ep_window"] = k_drive
        slim["is_burn"] = is_burn
        slim["lesion_applied"] = lesion_applied
        slim.update(region_stats)
        jsonl_handle.write(to_json_line(slim) + "\n")
        jsonl_handle.flush()

        if not is_burn:
            ep_rates.append(ep_rate)
            accept_rates.append(accept_window)
            k_drive_rates.append(k_drive)
            if mismatch_abs is not None:
                mismatch_vals.append(float(mismatch_abs))
            if strobe_l2 is not None and params.p3_on:
                strobe_vals.append(float(strobe_l2))

            if accept_window < accept_min:
                accept_low += 1
            else:
                accept_low = 0
            if accept_low >= 5:
                status = "FAIL_ACCEPT_COLLAPSE"

        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "window_index": window_idx,
                "step": step,
                "ep_rate_exact_window": ep_rate,
                "acceptedFracWindow": accept_window,
                "k_drive_ep_window": k_drive,
                "mismatch_abs_mean": mismatch_abs,
                "w_mass_region": region_stats["w_mass_region"],
                "w_mass_outside": region_stats["w_mass_outside"],
                "k_entropy_region": region_stats["k_entropy_region"],
                "k_entropy_outside": region_stats["k_entropy_outside"],
                "mismatch_region": region_stats["mismatch_region"],
                "mismatch_outside": region_stats["mismatch_outside"],
                "lesion_applied": lesion_applied,
                "is_burn": is_burn,
            }
        )

        progress_handle.write(
            f"{case},{seed},{step},{window_idx},{ep_rate},{accept_window},{k_drive},"
            f"{mismatch_abs},{region_stats['w_mass_region']},{region_stats['w_mass_outside']},"
            f"{region_stats['k_entropy_region']},{region_stats['k_entropy_outside']},"
            f"{region_stats['mismatch_region']},{region_stats['mismatch_outside']},"
            f"{lesion_applied}\n"
        )
        progress_handle.flush()
        os.fsync(progress_handle.fileno())

        if time.monotonic() - run_start > max_seconds_per_run:
            status = "FAIL_TIME"
        if time.monotonic() - start_total > max_seconds_total:
            status = "FAIL_TIME"

    def stop_cb(state, step, ep_ledger, accepted_frac):
        return status != "RUNNING" or window_idx >= max_windows

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

    if lesion_window >= 0 and not lesion_applied:
        status = "FAIL_CONFIG"

    runtime = time.monotonic() - run_start
    if status == "RUNNING":
        status = "FAIL_TIME" if window_idx < max_windows else "OK"

    tail_ep = ep_rates[-last_m:] if ep_rates else []
    tail_acc = accept_rates[-last_m:] if accept_rates else []
    tail_k = k_drive_rates[-last_m:] if k_drive_rates else []
    tail_m = mismatch_vals[-last_m:] if mismatch_vals else []
    tail_s = strobe_vals[-last_m:] if strobe_vals else []

    ep_mean, ep_ci = _mean_ci(tail_ep)
    acc_mean, _ = _mean_ci(tail_acc)
    k_mean, _ = _mean_ci(tail_k)
    mismatch_mean, _ = _mean_ci(tail_m)
    strobe_mean, _ = _mean_ci(tail_s)
    ep_slope = _linear_slope(tail_ep)
    k_slope = _linear_slope(tail_k)
    mismatch_slope = _linear_slope(tail_m)
    strobe_slope = _linear_slope(tail_s)

    summary = {
        "case": case,
        "seed": seed,
        "status": status,
        "windows_completed": window_idx,
        "runtime_sec": runtime,
        "accept_mean_last_m": acc_mean,
        "ep_mean_last_m": ep_mean,
        "ep_ci_half_last_m": ep_ci,
        "ep_slope_last_m": ep_slope,
        "mismatch_mean_last_m": mismatch_mean,
        "mismatch_slope_last_m": mismatch_slope,
        "k_drive_mean_last_m": k_mean,
        "k_drive_slope_last_m": k_slope,
        "strobe_l2_mean_last_m": strobe_mean,
        "strobe_l2_slope_last_m": strobe_slope,
    }

    if raw_rows:
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

    summary_rows.append(summary)
    with summary_path.open("w", encoding="utf-8", newline="") as sh:
        writer = csv.DictWriter(sh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary


def _fmt(val: Any) -> str:
    try:
        return f"{float(val):.6g}"
    except (TypeError, ValueError):
        return "nan"


def _write_report(agg_rows: List[Dict[str, Any]], report_path: Path) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 8 spatial harness v1\n\n")
        fh.write("## Summary\n\n")
        fh.write("| case | seed | status | windows | accept_mean | ep_mean | ep_ci | ep_slope | ")
        fh.write("k_drive_mean | k_drive_slope | mismatch_mean | mismatch_slope | strobe_l2_mean | strobe_l2_slope |\n")
        fh.write("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in agg_rows:
            fh.write(
                f"| {row['case']} | {row['seed']} | {row['status']} | {row['windows_completed']} | "
                f"{_fmt(row['accept_mean_last_m'])} | {_fmt(row['ep_mean_last_m'])} | {_fmt(row['ep_ci_half_last_m'])} | "
                f"{_fmt(row['ep_slope_last_m'])} | {_fmt(row['k_drive_mean_last_m'])} | {_fmt(row['k_drive_slope_last_m'])} | "
                f"{_fmt(row['mismatch_mean_last_m'])} | {_fmt(row['mismatch_slope_last_m'])} | "
                f"{_fmt(row['strobe_l2_mean_last_m'])} | {_fmt(row['strobe_l2_slope_last_m'])} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 spatial harness v1")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", default="scripts/params/meta_null_coupled_eta1.00_layers3.json")
    parser.add_argument("--out-dir", default=".tmp/phase8_spatial_v1")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--last-m", type=int, default=20)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--maps", default="sigma,w_mass,w_entropy,w_axis_bias,k_entropy,k_axis_bias,mismatch")
    parser.add_argument("--max-seconds-total", type=float, default=6600)
    parser.add_argument("--max-seconds-per-run", type=float, default=2100)
    parser.add_argument("--lesion-window", type=int, default=-1)
    parser.add_argument("--lesion-rect", default="")
    parser.add_argument("--lesion-sigma", default="random", choices=["none", "flip", "random"])
    parser.add_argument("--lesion-w-frac", type=float, default=1.0)
    parser.add_argument("--lesion-layers", default="all")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lesion_window >= 0 and not args.lesion_rect:
        raise ValueError("--lesion-rect is required when --lesion-window is set")

    preset = _load_preset(Path(args.preset))
    overrides: Dict[str, Any] = {}
    if args.eta is not None:
        overrides["eta"] = args.eta
    params = _as_params(preset, overrides)
    if args.device:
        params = Params.from_dict(params, {"device": args.device})

    if len(params.shape) != 2:
        raise ValueError("Phase 8 requires a 2D lattice shape")

    seeds = _parse_seeds(args.seeds)
    maps = _parse_maps(args.maps)
    cycle = _cycle_list()
    start_total = time.monotonic()

    lesion_mask = None
    lesion_idx = None
    if args.lesion_window >= 0:
        lesion_mask, lesion_idx = parse_rect(args.lesion_rect, params.shape)
        lesion_mask = lesion_mask.to(dtype=torch.bool, device=params.resolved_device())
        lesion_idx = lesion_idx.to(dtype=torch.long, device=params.resolved_device())

    agg_rows: List[Dict[str, Any]] = []

    for case, overrides in [
        ("null_full", {"p3_on": False, "p6_on": False, "eta_drive": 0.0}),
        ("p6_drive", {"p3_on": False, "p6_on": True}),
        ("p3p6_combo", {"p3_on": True, "p6_on": True}),
    ]:
        case_params = Params.from_dict(params, overrides)
        for seed in seeds:
            summary = run_case(
                case=case,
                params=case_params,
                seed=seed,
                out_dir=out_dir,
                burn_sweeps=args.burn_in_sweeps,
                window_sweeps=args.window_sweeps,
                max_windows=args.max_windows,
                last_m=args.last_m,
                accept_min=args.accept_min,
                maps=maps,
                snapshot_every=args.snapshot_every_windows,
                lesion_window=args.lesion_window,
                lesion_mask=lesion_mask,
                lesion_idx=lesion_idx,
                lesion_sigma=args.lesion_sigma,
                lesion_w_frac=args.lesion_w_frac,
                lesion_layers=args.lesion_layers,
                max_seconds_total=args.max_seconds_total,
                max_seconds_per_run=args.max_seconds_per_run,
                start_total=start_total,
                cycle=cycle,
                resume=args.resume,
            )
            agg_rows.append(summary)
            if summary["status"].startswith("FAIL") and summary["status"] != "FAIL_TIME":
                break
        if time.monotonic() - start_total > args.max_seconds_total:
            break

    agg_path = out_dir / "agg.csv"
    if agg_rows:
        with agg_path.open("w", encoding="utf-8", newline="") as ah:
            writer = csv.DictWriter(ah, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)

    report_path = out_dir / "PHASE8_SPATIAL_REPORT.md"
    _write_report(agg_rows, report_path)


if __name__ == "__main__":
    main()
