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


def _patchiness(map_tensor: torch.Tensor) -> float:
    data = map_tensor.to(dtype=torch.float32)
    return float(data.flatten().std(unbiased=False).item())


def _contrast(region: float, outside: float) -> float:
    return abs(region - outside)


def _compute_realloc_scores(
    hazard_stats: Dict[str, List[float]],
    baseline_stats: Dict[str, List[float]],
    hazard_start: int,
    hazard_duration: int,
    total_windows: int,
    pre_windows: int = 2,
    post_windows: int = 3,
) -> Dict[str, float]:
    pre_idx = list(range(max(1, hazard_start - pre_windows), hazard_start))
    haz_idx = list(range(hazard_start, min(total_windows, hazard_start + hazard_duration - 1) + 1))
    post_idx = list(range(max(1, total_windows - post_windows + 1), total_windows + 1))

    def _mean_at(vals: List[float], idx: List[int]) -> float:
        if not idx:
            return 0.0
        return float(np.mean([vals[i - 1] for i in idx]))

    scores: Dict[str, float] = {}
    keys = (
        "k_contrast",
        "w_contrast",
        "k_patch",
        "w_patch",
        "k_delta_focus",
        "w_delta_focus",
        "k_focus",
        "w_focus",
        "k_axis_bias_focus",
    )
    for key in keys:
        h_pre = _mean_at(hazard_stats[key], pre_idx)
        h_haz = _mean_at(hazard_stats[key], haz_idx)
        b_pre = _mean_at(baseline_stats[key], pre_idx)
        b_haz = _mean_at(baseline_stats[key], haz_idx)
        scores[key] = abs((h_haz - h_pre) - (b_haz - b_pre))

    scores["best"] = max(scores.values()) if scores else 0.0
    if scores:
        best_key, best_val = max(scores.items(), key=lambda item: item[1])
        scores["best_key"] = best_key
        scores["best_val"] = best_val
    scores["post_idx_count"] = float(len(post_idx))
    return scores


def _hazard_metrics(
    mismatch_region: List[float],
    hazard_start: int,
    hazard_duration: int,
    total_windows: int,
    pre_windows: int = 2,
    post_windows: int = 3,
) -> Dict[str, float]:
    pre_idx = list(range(max(1, hazard_start - pre_windows), hazard_start))
    haz_idx = list(range(hazard_start, min(total_windows, hazard_start + hazard_duration - 1) + 1))
    post_idx = list(range(max(1, total_windows - post_windows + 1), total_windows + 1))
    pre_vals = [mismatch_region[i - 1] for i in pre_idx] if pre_idx else []
    haz_vals = [mismatch_region[i - 1] for i in haz_idx] if haz_idx else []
    post_vals = [mismatch_region[i - 1] for i in post_idx] if post_idx else []
    pre = float(np.mean(pre_vals)) if pre_vals else 0.0
    peak = float(np.max(haz_vals)) if haz_vals else pre
    post = float(np.mean(post_vals)) if post_vals else pre
    raw_spike = peak - pre
    spike = max(0.0, raw_spike)
    denom = max(spike, 1e-12)
    recovery = (peak - post) / denom if spike > 0 else 0.0
    return {
        "pre": pre,
        "peak": peak,
        "post": post,
        "raw_spike": raw_spike,
        "spike": spike,
        "recovery_frac": recovery,
    }


def run_case(
    case: str,
    params: Params,
    seed: int,
    out_dir: Path,
    burn_sweeps: float,
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
                "w_mass_patchiness,k_entropy_patchiness,w_mass_delta_l1,k_entropy_delta_l1,"
                "w_mass_delta_region,w_mass_delta_outside,k_entropy_delta_region,k_entropy_delta_outside,"
                "k_delta_focus,w_delta_focus,k_focus,w_focus,"
                "k_axis_bias_abs_region_if0,k_axis_bias_abs_outside_if0,k_axis_bias_abs_focus_if0,"
                "k_axis_bias_abs_focus\n"
            )
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,window_index,step,hazard_active,ep_rate,accept_window,k_drive,"
                "mismatch_region,mismatch_outside,w_mass_region,w_mass_outside,k_entropy_region,k_entropy_outside\n"
            )

    if len(params.shape) != 2:
        raise ValueError("Phase 9 requires a 2D lattice shape")
    N = math.prod(params.shape)
    expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected_props))
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

    prev_w_mass = None
    prev_k_entropy = None

    mismatch_region_vals: List[float] = []
    mismatch_outside_vals: List[float] = []
    k_contrast_vals: List[float] = []
    w_contrast_vals: List[float] = []
    k_patch_vals: List[float] = []
    w_patch_vals: List[float] = []
    k_delta_focus_vals: List[float] = []
    w_delta_focus_vals: List[float] = []
    k_focus_vals: List[float] = []
    w_focus_vals: List[float] = []
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
        nonlocal status, accept_low, hazard_active_next, diag_state, window_idx, prev_w_mass, prev_k_entropy
        if status != "RUNNING" or window_idx >= max_windows:
            return

        snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
        is_burn = step <= burn_steps
        if not is_burn:
            window_idx += 1

        hazard_active = hazard_active_next if not is_burn else False

        window_props = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
        accept_window = float(ep_ledger.get("window_accepted", 0)) / window_props if window_props else 0.0
        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        mismatch_abs = snapshot.get("mismatch_abs_mean")
        k_drive = _k_drive_rate(snapshot)

        maps_dict = {}
        region_stats = {
            "w_mass_region": None,
            "w_mass_outside": None,
            "k_entropy_region": None,
            "k_entropy_outside": None,
            "mismatch_region": None,
            "mismatch_outside": None,
        }
        w_mass_patch = 0.0
        k_entropy_patch = 0.0
        w_mass_delta = 0.0
        k_entropy_delta = 0.0
        w_mass_delta_region = 0.0
        w_mass_delta_outside = 0.0
        k_entropy_delta_region = 0.0
        k_entropy_delta_outside = 0.0
        k_delta_focus = 0.0
        w_delta_focus = 0.0
        k_focus = 0.0
        w_focus = 0.0
        k_axis_bias_region_if0 = None
        k_axis_bias_outside_if0 = None
        k_axis_bias_focus_if0 = 0.0
        k_axis_bias_focus = 0.0

        if not is_burn and (window_idx % snapshot_every == 0):
            maps_dict = compute_spatial_maps(state, maps)
            ok, bad = finite_check(maps_dict)
            if not ok:
                status = f"FAIL_NAN_MAP:{','.join(bad)}"
                return

            w_mass_grid = maps_dict.get("w_mass")
            k_entropy_grid = maps_dict.get("k_entropy")
            mismatch_grid = maps_dict.get("mismatch")
            k_axis_bias_grid = maps_dict.get("k_axis_bias")

            if w_mass_grid is not None:
                w_mass_grid = w_mass_grid.to(dtype=torch.float32)
                w_mass_patch = _patchiness(w_mass_grid)
            if k_entropy_grid is not None:
                k_entropy_patch = _patchiness(k_entropy_grid)

            if hazard_mask is not None:
                if w_mass_grid is not None:
                    region_stats["w_mass_region"], region_stats["w_mass_outside"] = _region_mean(
                        w_mass_grid, hazard_mask
                    )
                if k_entropy_grid is not None:
                    region_stats["k_entropy_region"], region_stats["k_entropy_outside"] = _region_mean(
                        k_entropy_grid, hazard_mask
                    )
                if mismatch_grid is not None:
                    region_stats["mismatch_region"], region_stats["mismatch_outside"] = _region_mean(
                        mismatch_grid, hazard_mask
                    )
                if k_axis_bias_grid is not None:
                    if0 = 0 if k_axis_bias_grid.shape[0] > 0 else None
                    if if0 is not None:
                        bias0 = k_axis_bias_grid[if0].abs()
                        k_axis_bias_region_if0, k_axis_bias_outside_if0 = _region_mean(bias0, hazard_mask)
                        k_axis_bias_focus_if0 = k_axis_bias_region_if0 - k_axis_bias_outside_if0
                    interfaces = [idx for idx in hazard_layers if idx < k_axis_bias_grid.shape[0]]
                    if interfaces:
                        focus_vals = []
                        for iface in interfaces:
                            bias = k_axis_bias_grid[iface].abs()
                            region, outside = _region_mean(bias, hazard_mask)
                            focus_vals.append(region - outside)
                        k_axis_bias_focus = max(focus_vals) if focus_vals else 0.0

            if w_mass_grid is not None and prev_w_mass is not None:
                delta = (w_mass_grid - prev_w_mass).abs()
                w_mass_delta = float(delta.mean().item())
                if hazard_mask is not None:
                    w_mass_delta_region, w_mass_delta_outside = _region_mean(delta, hazard_mask)
            if k_entropy_grid is not None and prev_k_entropy is not None:
                delta = (k_entropy_grid - prev_k_entropy).abs()
                k_entropy_delta = float(delta.mean().item())
                if hazard_mask is not None:
                    k_entropy_delta_region, k_entropy_delta_outside = _region_mean(delta, hazard_mask)

            if w_mass_grid is not None:
                prev_w_mass = w_mass_grid.detach()
            if k_entropy_grid is not None:
                prev_k_entropy = k_entropy_grid.detach()

            npz_payload = {k: v.detach().cpu().numpy() for k, v in maps_dict.items()}
            np.savez(maps_dir / f"seed{seed}_win{window_idx:04d}.npz", **npz_payload)

        if not is_burn:
            mismatch_region_vals.append(float(region_stats["mismatch_region"] or 0.0))
            mismatch_outside_vals.append(float(region_stats["mismatch_outside"] or 0.0))
            k_region = float(region_stats["k_entropy_region"] or 0.0)
            k_outside = float(region_stats["k_entropy_outside"] or 0.0)
            w_region = float(region_stats["w_mass_region"] or 0.0)
            w_outside = float(region_stats["w_mass_outside"] or 0.0)
            k_contrast_vals.append(_contrast(k_region, k_outside))
            w_contrast_vals.append(_contrast(w_region, w_outside))
            k_patch_vals.append(float(k_entropy_patch))
            w_patch_vals.append(float(w_mass_patch))
            k_delta_focus = k_entropy_delta_region - k_entropy_delta_outside
            w_delta_focus = w_mass_delta_region - w_mass_delta_outside
            k_delta_focus_vals.append(k_delta_focus)
            w_delta_focus_vals.append(w_delta_focus)
            k_focus = k_outside - k_region
            w_focus = w_region - w_outside
            k_focus_vals.append(k_focus)
            w_focus_vals.append(w_focus)
            k_axis_bias_focus_vals.append(float(k_axis_bias_focus))

        if accept_window < accept_min:
            accept_low += 1
        else:
            accept_low = 0
        if accept_low >= 5 and not is_burn:
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
                "w_mass_region": region_stats["w_mass_region"],
                "w_mass_outside": region_stats["w_mass_outside"],
                "k_entropy_region": region_stats["k_entropy_region"],
                "k_entropy_outside": region_stats["k_entropy_outside"],
                "w_mass_patchiness": w_mass_patch,
                "k_entropy_patchiness": k_entropy_patch,
                "w_mass_delta_l1": w_mass_delta,
                "k_entropy_delta_l1": k_entropy_delta,
                "w_mass_delta_region": w_mass_delta_region,
                "w_mass_delta_outside": w_mass_delta_outside,
                "k_entropy_delta_region": k_entropy_delta_region,
                "k_entropy_delta_outside": k_entropy_delta_outside,
                "k_delta_focus": k_delta_focus,
                "w_delta_focus": w_delta_focus,
                "k_focus": k_focus,
                "w_focus": w_focus,
                "k_axis_bias_abs_region_if0": k_axis_bias_region_if0,
                "k_axis_bias_abs_outside_if0": k_axis_bias_outside_if0,
                "k_axis_bias_abs_focus_if0": k_axis_bias_focus_if0,
                "k_axis_bias_abs_focus": k_axis_bias_focus,
                "is_burn": is_burn,
            }
        )
        jsonl_handle.write(to_json_line(slim_snapshot) + "\n")
        jsonl_handle.flush()

        with raw_path.open("a", encoding="utf-8") as rh:
            rh.write(
                f"{case},{seed},{window_idx},{step},{hazard_active},{ep_rate},{accept_window},{k_drive},"
                f"{mismatch_abs},{region_stats['mismatch_region']},{region_stats['mismatch_outside']},"
                f"{region_stats['w_mass_region']},{region_stats['w_mass_outside']},"
                f"{region_stats['k_entropy_region']},{region_stats['k_entropy_outside']},"
                f"{w_mass_patch},{k_entropy_patch},{w_mass_delta},{k_entropy_delta},"
                f"{w_mass_delta_region},{w_mass_delta_outside},{k_entropy_delta_region},{k_entropy_delta_outside},"
                f"{k_delta_focus},{w_delta_focus},{k_focus},{w_focus},"
                f"{k_axis_bias_region_if0},{k_axis_bias_outside_if0},{k_axis_bias_focus_if0},"
                f"{k_axis_bias_focus}\n"
            )

        progress_handle.write(
            f"{case},{seed},{window_idx},{step},{hazard_active},{ep_rate},{accept_window},{k_drive},"
            f"{region_stats['mismatch_region']},{region_stats['mismatch_outside']},"
            f"{region_stats['w_mass_region']},{region_stats['w_mass_outside']},"
            f"{region_stats['k_entropy_region']},{region_stats['k_entropy_outside']}\n"
        )
        progress_handle.flush()

        if not is_burn and apply_hazard:
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
        steps=burn_steps + window_steps * max_windows,
        report_every=window_steps,
        report_callback=report_cb,
        stop_callback=stop_cb,
        protocol_cycle=cycle,
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
        "k_contrast": k_contrast_vals,
        "w_contrast": w_contrast_vals,
        "k_patch": k_patch_vals,
        "w_patch": w_patch_vals,
        "k_delta_focus": k_delta_focus_vals,
        "w_delta_focus": w_delta_focus_vals,
        "k_focus": k_focus_vals,
        "w_focus": w_focus_vals,
        "k_axis_bias_focus": k_axis_bias_focus_vals,
    }
    if window_idx < max_windows and status == "OK":
        status = "FAIL_TIME"
        metrics["status"] = status
    metrics["runtime_sec"] = time.monotonic() - run_start

    with summary_path.open("w", encoding="utf-8", newline="") as sh:
        writer = csv.DictWriter(sh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics


def _fmt(val: Any) -> str:
    try:
        return f"{float(val):.6g}"
    except (TypeError, ValueError):
        return "nan"


def _write_report(rows: List[Dict[str, Any]], report_path: Path, command: str) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 9 hazard attention highways v1\n\n")
        fh.write("## Command\n\n")
        fh.write(f"`{command}`\n\n")
        fh.write("## Summary\n\n")
        fh.write("| case | seed | status | raw_spike | spike | recovery_frac | best_realloc | best_metric |\n")
        fh.write("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            fh.write(
                f"| {row['case']} | {row['seed']} | {row['status']} | {_fmt(row['raw_spike'])} | "
                f"{_fmt(row['spike'])} | {_fmt(row['recovery_frac'])} | {_fmt(row['best_realloc'])} | "
                f"{row['best_realloc_metric']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 hazard attention highways v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", default="scripts/params/meta_null_coupled_eta1.00_layers3.json")
    parser.add_argument("--out-dir", default=".tmp/phase9_hazard_v1")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--hazard-start-window", type=int, default=5)
    parser.add_argument("--hazard-duration-windows", type=int, default=5)
    parser.add_argument("--hazard-rect", type=str, required=False)
    parser.add_argument("--hazard-sigma", choices=["random", "flip"], default="random")
    parser.add_argument("--hazard-layers", default="0")
    parser.add_argument("--hazard-refresh-each-window", action="store_true")
    parser.add_argument("--mismatch-spike-min", type=float, default=0.01)
    parser.add_argument("--recovery-frac-min", type=float, default=0.20)
    parser.add_argument("--realloc-min", type=float, default=0.005)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--B-w-scale", type=float, default=1.0)
    parser.add_argument("--B-k-scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=None)
    args = parser.parse_args()

    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)
    if not args.hazard_rect:
        raise ValueError("--hazard-rect is required for Phase 9")

    preset = _load_preset(Path(args.preset))
    base_overrides: Dict[str, Any] = {
        "device": args.device,
    }
    if args.eta is not None:
        base_overrides["eta"] = float(args.eta)

    base_params = _as_params(preset, base_overrides)
    if args.B_w_scale != 1.0:
        base_params = Params.from_dict(base_params, {"B_w": int(round(base_params.B_w * args.B_w_scale))})
    if args.B_k_scale != 1.0:
        base_params = Params.from_dict(base_params, {"B_k": int(round(base_params.B_k * args.B_k_scale))})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    hazard_layers = _parse_layers(args.hazard_layers, base_params.layers)
    hazard_layers = [layer for layer in hazard_layers if layer < base_params.layers]
    hazard_rect = args.hazard_rect

    cycle = _cycle_list()
    start_total = time.monotonic()
    report_rows: List[Dict[str, Any]] = []

    command = " ".join([str(x) for x in sys.argv])

    for idx, seed in enumerate(seeds):
        if time.monotonic() - start_total > args.max_seconds_total:
            break

        baseline_metrics = run_case(
            "baseline",
            base_params,
            seed,
            out_dir,
            args.burn_in_sweeps,
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
        )

        if baseline_metrics["status"] != "OK":
            report_rows.append(
                {
                    "case": "baseline",
                    "seed": seed,
                    "status": baseline_metrics["status"],
                    "raw_spike": 0.0,
                    "spike": 0.0,
                    "recovery_frac": 0.0,
                    "best_realloc": 0.0,
                    "best_realloc_metric": "",
                }
            )
            _write_report(report_rows, out_dir / "PHASE9_HAZARD_ATTENTION_REPORT.md", command)
            break

        hazard_metrics = run_case(
            "hazard",
            base_params,
            seed,
            out_dir,
            args.burn_in_sweeps,
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
        )

        total_windows = int(hazard_metrics["windows_completed"])
        hazard_stats = {
            "mismatch_region": hazard_metrics["mismatch_region"],
            "k_contrast": hazard_metrics["k_contrast"],
            "w_contrast": hazard_metrics["w_contrast"],
            "k_patch": hazard_metrics["k_patch"],
            "w_patch": hazard_metrics["w_patch"],
            "k_delta_focus": hazard_metrics["k_delta_focus"],
            "w_delta_focus": hazard_metrics["w_delta_focus"],
            "k_focus": hazard_metrics["k_focus"],
            "w_focus": hazard_metrics["w_focus"],
            "k_axis_bias_focus": hazard_metrics["k_axis_bias_focus"],
        }
        baseline_stats = {
            "mismatch_region": baseline_metrics["mismatch_region"],
            "k_contrast": baseline_metrics["k_contrast"],
            "w_contrast": baseline_metrics["w_contrast"],
            "k_patch": baseline_metrics["k_patch"],
            "w_patch": baseline_metrics["w_patch"],
            "k_delta_focus": baseline_metrics["k_delta_focus"],
            "w_delta_focus": baseline_metrics["w_delta_focus"],
            "k_focus": baseline_metrics["k_focus"],
            "w_focus": baseline_metrics["w_focus"],
            "k_axis_bias_focus": baseline_metrics["k_axis_bias_focus"],
        }

        hazard_score = _hazard_metrics(
            hazard_stats["mismatch_region"],
            args.hazard_start_window,
            args.hazard_duration_windows,
            total_windows,
        )
        realloc_scores = _compute_realloc_scores(
            hazard_stats,
            baseline_stats,
            args.hazard_start_window,
            args.hazard_duration_windows,
            total_windows,
        )

        raw_spike = hazard_score["raw_spike"]
        spike = hazard_score["spike"]
        recovery = hazard_score["recovery_frac"]
        best_realloc = realloc_scores["best"]
        best_metric = realloc_scores.get("best_key", "")
        pass_seed = (
            spike >= args.mismatch_spike_min
            and recovery >= args.recovery_frac_min
            and best_realloc >= args.realloc_min
        )
        status = "PASS" if pass_seed else "FAIL"
        report_rows.append(
            {
                "case": "hazard",
                "seed": seed,
                "status": status,
                "raw_spike": raw_spike,
                "spike": spike,
                "recovery_frac": recovery,
                "best_realloc": best_realloc,
                "best_realloc_metric": best_metric,
            }
        )

        _write_report(report_rows, out_dir / "PHASE9_HAZARD_ATTENTION_REPORT.md", command)

        if idx == 0 and not pass_seed:
            break

    if args.progress:
        print(f"PHASE9_REPORT={out_dir / 'PHASE9_HAZARD_ATTENTION_REPORT.md'}")


if __name__ == "__main__":
    main()
