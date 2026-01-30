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
from ratchet_gpu.interventions import (
    apply_k_redistribute_radial_inward_in_ring,
    apply_k_redistribute_radial_random_in_ring,
    apply_k_redistribute_uniform_in_region,
    apply_sigma_flip,
    apply_sigma_randomize,
    check_k_invariants,
    parse_rect,
)
from ratchet_gpu.params import Params
from ratchet_gpu.semantics import hazard_center, radial_focus_shift_null, ring_masks_from_rect
from ratchet_gpu.sim import run_sim
from ratchet_gpu.spatial import k_radial_focus_grid, mismatch_abs_grid, sigma_grid
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


HEAVY_KEYS = {
    "strobe_current_map_items_window",
    "strobe_currents_window",
    "strobe_current_map_items_count_window",
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


def _parse_seeds(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_interfaces(value: str) -> str | List[int]:
    if value == "all":
        return value
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


def _get_accept(snapshot: Dict[str, Any]) -> float:
    value = snapshot.get("acceptedFracWindow", snapshot.get("acceptedFrac", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _apply_hazard(
    state: State,
    mode: str,
    flat_idx: torch.Tensor,
    layers: str,
    rng: torch.Generator,
) -> None:
    if mode == "none":
        return
    if mode == "flip":
        apply_sigma_flip(state, flat_idx, layers=layers)
    elif mode == "random":
        apply_sigma_randomize(state, flat_idx, layers=layers, rng=rng)
    else:
        raise ValueError(f"unknown hazard sigma mode {mode}")


def _apply_intervention(
    condition: str,
    state: State,
    params: Params,
    ring_idx: torch.Tensor,
    interfaces: str | List[int],
    center: Tuple[float, float],
    strength: float,
    rng: torch.Generator,
) -> None:
    if condition == "inject":
        apply_k_redistribute_radial_inward_in_ring(
            state,
            params,
            ring_idx,
            center=center,
            interfaces=interfaces,
            strength=strength,
            rng=rng,
        )
    elif condition == "ablate":
        apply_k_redistribute_uniform_in_region(
            state,
            params,
            ring_idx,
            interfaces=interfaces,
            rng=rng,
        )
    elif condition == "sham":
        apply_k_redistribute_radial_random_in_ring(
            state,
            params,
            ring_idx,
            interfaces=interfaces,
            rng=rng,
        )


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _alignment_delta(pre_vals: List[float], haz_vals: List[float]) -> float:
    if not pre_vals or not haz_vals:
        return 0.0
    return float(np.mean(haz_vals) - np.mean(pre_vals))


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


def _build_csv_writer(path: Path, fieldnames: List[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    os.fsync(handle.fileno())
    writer._handle = handle  # type: ignore[attr-defined]
    return writer


def _write_row(writer: csv.DictWriter, row: Dict[str, Any]) -> None:
    writer.writerow(row)
    handle = writer._handle  # type: ignore[attr-defined]
    handle.flush()
    os.fsync(handle.fileno())


def _close_writer(writer: csv.DictWriter) -> None:
    handle = writer._handle  # type: ignore[attr-defined]
    handle.close()


def _resume_rows(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("condition", ""), int(row.get("seed", 0)))
            rows[key] = row
    return rows


def _run_windows(
    state: State,
    params: Params,
    device: torch.device | str,
    seed: int,
    window_steps: int,
    window_start: int,
    window_end: int,
    hazard_start: int,
    hazard_duration: int,
    hazard_sigma: str,
    hazard_layers: str,
    hazard_refresh_each: bool,
    hazard_flat_idx: torch.Tensor,
    condition: str,
    interfaces: str | List[int],
    ring_idx: torch.Tensor,
    center: Tuple[float, float],
    strength: float,
    region_mask: torch.Tensor,
    ring_mask: np.ndarray,
    mismatch_interface: int,
    snapshot_dir: Path,
    snapshot_every: int,
    raw_writer: csv.DictWriter,
    progress_writer: csv.DictWriter,
    jsonl_handle,
    diag_state: Dict[str, Any] | None,
    max_seconds: float,
    progress: bool,
    rng_state: torch.Tensor | None,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[np.ndarray],
    List[float],
    Dict[str, Any] | None,
    torch.Tensor | None,
]:
    mismatch_region_vals: List[float] = []
    mismatch_outside_vals: List[float] = []
    radial_ring_vals: List[float] = []
    radial_maps: List[np.ndarray] = []
    accept_vals: List[float] = []

    rng = torch.Generator(device=state.device)
    rng.manual_seed(seed)
    rng_state_local = rng_state

    for window_idx in range(window_start, window_end + 1):
        hazard_active = hazard_start <= window_idx <= hazard_start + hazard_duration - 1
        if hazard_active and hazard_refresh_each:
            _apply_hazard(state, hazard_sigma, hazard_flat_idx, hazard_layers, rng)
        if hazard_active and condition in {"inject", "ablate", "sham"}:
            _apply_intervention(
                condition,
                state,
                params,
                ring_idx,
                interfaces,
                center,
                strength,
                rng,
            )
            ok, msg = check_k_invariants(state, params)
            if not ok:
                raise RuntimeError(f"K invariants violated: {msg}")

        snapshot: Dict[str, Any] | None = None

        def report_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> None:
            nonlocal snapshot, diag_state
            snap, diag_state = compute_snapshot(st, step, ep_ledger, accepted_frac, diag_state)
            snap = _slim_snapshot(snap)
            snap.update(
                {
                    "window_index": window_idx,
                    "condition": condition,
                    "seed": seed,
                    "hazard_active": hazard_active,
                }
            )
            jsonl_handle.write(to_json_line(snap))
            jsonl_handle.flush()
            snapshot = snap

        seed_for_run = seed if rng_state_local is None else None
        summary = run_sim(
            params,
            seed=seed_for_run,
            steps=window_steps,
            report_every=window_steps,
            report_callback=report_cb,
            initial_state=state,
            initial_rng_state=rng_state_local,
            return_state=True,
        )
        rng_state_local = summary.get("rng_state", rng_state_local)

        if snapshot is None:
            raise RuntimeError("Missing snapshot for window")

        # Spatial metrics
        mismatch = mismatch_abs_grid(state)[mismatch_interface]
        mismatch_region = float(mismatch[region_mask].mean().item()) if region_mask.any() else 0.0
        mismatch_outside = float(mismatch[~region_mask].mean().item()) if (~region_mask).any() else 0.0
        radial = k_radial_focus_grid(state, center)[mismatch_interface]
        radial_map = radial.cpu().numpy()
        radial_ring = float(radial_map[ring_mask].mean()) if ring_mask.any() else 0.0

        mismatch_region_vals.append(mismatch_region)
        mismatch_outside_vals.append(mismatch_outside)
        radial_ring_vals.append(radial_ring)
        radial_maps.append(radial_map)
        accept_vals.append(_get_accept(snapshot))

        raw_row = {
            "condition": condition,
            "seed": seed,
            "window_index": window_idx,
            "step": snapshot.get("step"),
            "hazard_active": hazard_active,
            "acceptedFracWindow": _get_accept(snapshot),
            "mismatch_region": mismatch_region,
            "mismatch_outside": mismatch_outside,
            "radial_focus_ring": radial_ring,
        }
        _write_row(raw_writer, raw_row)

        progress_row = {
            "condition": condition,
            "seed": seed,
            "window_index": window_idx,
            "radial_focus_ring": radial_ring,
            "mismatch_region": mismatch_region,
            "acceptedFracWindow": _get_accept(snapshot),
        }
        _write_row(progress_writer, progress_row)

        if snapshot_every > 0 and window_idx % snapshot_every == 0:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            sigma = sigma_grid(state).cpu().numpy()
            mismatch_grid = mismatch.cpu().numpy()
            radial_grid = radial_map
            np.savez_compressed(
                snapshot_dir / f"seed{seed}_win{window_idx:04d}.npz",
                step=snapshot.get("step", 0),
                window_index=window_idx,
                hazard_active=hazard_active,
                sigma_l0=sigma[0],
                **{
                    f"mismatch_i{mismatch_interface}": mismatch_grid,
                    f"k_radial_focus_i{mismatch_interface}": radial_grid,
                },
            )

        if progress:
            print(
                f"PROGRESS seed={seed} condition={condition} window={window_idx}/{window_end} "
                f"hazard_active={hazard_active} radial_focus_ring={radial_ring:.4g}",
                flush=True,
            )

        if time.monotonic() >= max_seconds:
            break

    return (
        mismatch_region_vals,
        mismatch_outside_vals,
        radial_ring_vals,
        radial_maps,
        accept_vals,
        diag_state,
        rng_state_local,
    )


def _summarize_condition(
    mismatch_region: List[float],
    radial_ring: List[float],
    accept_vals: List[float],
    hazard_start: int,
    hazard_duration: int,
    pre_windows: int,
    last_m: int,
) -> Dict[str, float]:
    pre_idx = list(range(max(1, hazard_start - pre_windows), hazard_start))
    haz_idx = list(
        range(hazard_start, hazard_start + hazard_duration)
    )

    def _mean_at(vals: List[float], idx: List[int]) -> float:
        if not idx:
            return 0.0
        return float(np.mean([vals[i - 1] for i in idx if i - 1 < len(vals)]))

    pre_mean = _mean_at(mismatch_region, pre_idx)
    haz_vals = [mismatch_region[i - 1] for i in haz_idx if i - 1 < len(mismatch_region)]
    peak = float(np.max(haz_vals)) if haz_vals else pre_mean
    raw_spike = peak - pre_mean
    spike = max(0.0, raw_spike)

    focus_pre = _mean_at(radial_ring, pre_idx)
    focus_haz = _mean_at(radial_ring, haz_idx)

    accept_mean = _mean(accept_vals[-last_m:])

    return {
        "pre_mean": pre_mean,
        "peak": peak,
        "raw_spike": raw_spike,
        "spike": spike,
        "focus_pre": focus_pre,
        "focus_haz": focus_haz,
        "focus_delta": focus_haz - focus_pre,
        "accept_mean": accept_mean,
    }


def _write_report(report_path: Path, rows: List[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 17 Directional Semantics\n"]
    lines.append("| seed | condition | status | spike | focus_delta | focus_p | accept |\n")
    lines.append("| ---: | :--- | :--- | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['condition']} | {row['status']} | "
            f"{row['spike']:.4g} | {row['focus_delta']:.4g} | {row['focus_p']:.4g} | {row['accept_mean']:.4g} |\n"
        )
    report_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 directional motif semantics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=int, default=150)
    parser.add_argument("--window-sweeps", type=int, default=80)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--hazard-start-window", type=int, default=6)
    parser.add_argument("--hazard-duration-windows", type=int, default=8)
    parser.add_argument("--hazard-rect", required=True)
    parser.add_argument("--hazard-sigma", default="random", choices=["random", "flip", "none"])
    parser.add_argument("--hazard-layers", default="0")
    parser.add_argument("--hazard-refresh-each-window", action="store_true")
    parser.add_argument("--ring-thickness", type=int, default=2)
    parser.add_argument("--interfaces", default="0")
    parser.add_argument("--motif-interface", default=None)
    parser.add_argument("--motif-features", default=None)
    parser.add_argument("--bins-axis-bias", type=int, default=None)
    parser.add_argument("--bins-entropy", type=int, default=None)
    parser.add_argument("--intervention-strength", type=float, default=1.0)
    parser.add_argument("--spike-min", type=float, default=0.01)
    parser.add_argument("--focus-delta-min", type=float, default=0.005)
    parser.add_argument("--p-max", type=float, default=0.10)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--sham-n", type=int, default=200)
    parser.add_argument("--shuffle-n", type=int, default=None)
    parser.add_argument("--align-delta-min", type=float, default=None)
    parser.add_argument("--align-p-max", type=float, default=None)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    if args.motif_interface is not None:
        args.interfaces = str(args.motif_interface)
    if args.shuffle_n is not None:
        args.sham_n = args.shuffle_n
    if args.align_delta_min is not None:
        args.focus_delta_min = args.align_delta_min
    if args.align_p_max is not None:
        args.p_max = args.align_p_max

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("no seeds specified")
    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)
    if args.ring_thickness < 1:
        raise ValueError("ring_thickness must be >= 1")

    preset = _load_preset(Path(args.preset))
    params = _as_params(preset, {"device": args.device})
    params = Params(**{**params.__dict__, "p3_on": False, "p6_on": False})
    device = params.resolved_device()
    device_str = str(device)

    shape = params.shape
    H, W = shape
    if len(shape) != 2:
        raise ValueError("Phase17 expects 2D lattice shape")

    region_np, ring_np, outside_np = ring_masks_from_rect(args.hazard_rect, shape, args.ring_thickness)
    center = hazard_center(args.hazard_rect, shape)
    hazard_mask, hazard_flat_idx = parse_rect(args.hazard_rect, shape)

    ring_idx = torch.as_tensor(np.flatnonzero(ring_np), dtype=torch.long)
    region_mask = hazard_mask.to(dtype=torch.bool, device=device)

    interface_idx = 0
    if args.interfaces != "all":
        interface_idx = int(args.interfaces.split(",")[0])

    N = math.prod(params.shape)
    expected = _expected_proposals_per_step(N, device_str, params.kernel_weights)
    burn_steps = int(math.ceil(args.burn_in_sweeps * N / expected))
    window_steps = int(math.ceil(args.window_sweeps * N / expected))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = out_dir / "agg.csv"
    agg_rows = _resume_rows(agg_path) if args.resume else {}
    agg_fields = [
        "condition",
        "seed",
        "status",
        "spike",
        "focus_delta",
        "focus_p",
        "accept_mean",
        "focus_delta_vs_sham",
    ]
    if not agg_path.exists():
        agg_writer = _build_csv_writer(agg_path, agg_fields)
    else:
        agg_writer = _build_csv_writer(agg_path, agg_fields)

    report_rows: List[Dict[str, Any]] = []

    total_start = time.monotonic()

    for seed in seeds:
        torch.manual_seed(seed)
        if device_str.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        base_state = State.initialize(params, seed=seed)
        rng_state = None

        if burn_steps > 0:
            summary = run_sim(
                params,
                seed=seed,
                steps=burn_steps,
                report_every=burn_steps,
                initial_state=base_state,
                return_state=True,
            )
            rng_state = summary.get("rng_state", rng_state)

        diag_state = None
        pre_mismatch: List[float] = []
        pre_radial: List[float] = []
        pre_radial_maps: List[np.ndarray] = []
        pre_accept: List[float] = []

        pre_windows = max(0, args.hazard_start_window - 1)
        for window_idx in range(1, pre_windows + 1):
            snapshot: Dict[str, Any] | None = None

            def pre_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> None:
                nonlocal snapshot, diag_state
                snap, diag_state = compute_snapshot(st, step, ep_ledger, accepted_frac, diag_state)
                snapshot = snap

            seed_for_run = seed if rng_state is None else None
            summary = run_sim(
                params,
                seed=seed_for_run,
                steps=window_steps,
                report_every=window_steps,
                report_callback=pre_cb,
                initial_state=base_state,
                initial_rng_state=rng_state,
                return_state=True,
            )
            rng_state = summary.get("rng_state", rng_state)
            if snapshot is None:
                raise RuntimeError("Missing pre snapshot")

            mismatch = mismatch_abs_grid(base_state)[interface_idx]
            mismatch_region = float(mismatch[region_mask].mean().item()) if region_mask.any() else 0.0
            radial = k_radial_focus_grid(base_state, center)[interface_idx]
            radial_map = radial.cpu().numpy()
            radial_ring = float(radial_map[ring_np].mean()) if ring_np.any() else 0.0

            pre_mismatch.append(mismatch_region)
            pre_radial.append(radial_ring)
            pre_radial_maps.append(radial_map)
            pre_accept.append(_get_accept(snapshot))

        base_state = _clone_state(base_state)

        conditions = ["control", "inject", "ablate", "sham"]
        focus_delta_by_condition: Dict[str, float] = {}
        focus_p_by_condition: Dict[str, float] = {}
        accept_by_condition: Dict[str, float] = {}
        spike_by_condition: Dict[str, float] = {}

        for condition in conditions:
            key = (condition, seed)
            if args.resume and key in agg_rows:
                row = agg_rows[key]
                report_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "status": row.get("status", "SKIP"),
                        "spike": float(row.get("spike", 0.0)),
                        "focus_delta": float(row.get("focus_delta", 0.0)),
                        "focus_p": float(row.get("focus_p", 1.0)),
                        "accept_mean": float(row.get("accept_mean", 0.0)),
                    }
                )
                continue

            run_start = time.monotonic()
            state = _clone_state(base_state)
            if device_str.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            rng_state_condition = rng_state.clone() if rng_state is not None else None

            condition_dir = out_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            (condition_dir / "jsonl").mkdir(parents=True, exist_ok=True)

            raw_writer = _build_csv_writer(
                condition_dir / "raw.csv",
                [
                    "condition",
                    "seed",
                    "window_index",
                    "step",
                    "hazard_active",
                    "acceptedFracWindow",
                    "mismatch_region",
                    "mismatch_outside",
                    "radial_focus_ring",
                ],
            )
            progress_writer = _build_csv_writer(
                condition_dir / "progress.csv",
                [
                    "condition",
                    "seed",
                    "window_index",
                    "radial_focus_ring",
                    "mismatch_region",
                    "acceptedFracWindow",
                ],
            )
            jsonl_path = condition_dir / "jsonl" / f"{condition}_seed{seed}.jsonl"
            jsonl_handle = jsonl_path.open("w")

            diag_state_condition = None

            (
                mismatch_region,
                mismatch_outside,
                radial_ring,
                radial_maps,
                accept_vals,
                diag_state_condition,
                rng_state_condition,
            ) = _run_windows(
                state,
                params,
                device,
                seed,
                window_steps,
                pre_windows + 1,
                args.max_windows,
                args.hazard_start_window,
                args.hazard_duration_windows,
                args.hazard_sigma,
                args.hazard_layers,
                args.hazard_refresh_each_window,
                hazard_flat_idx,
                condition,
                _parse_interfaces(args.interfaces),
                ring_idx,
                center,
                args.intervention_strength,
                region_mask,
                ring_np,
                interface_idx,
                condition_dir / "npz",
                args.snapshot_every_windows,
                raw_writer,
                progress_writer,
                jsonl_handle,
                diag_state_condition,
                run_start + args.max_seconds_per_run,
                args.progress,
                rng_state_condition,
            )

            jsonl_handle.close()
            _close_writer(raw_writer)
            _close_writer(progress_writer)

            full_mismatch = pre_mismatch + mismatch_region
            full_radial = pre_radial + radial_ring
            full_radial_maps = pre_radial_maps + radial_maps
            full_accept = pre_accept + accept_vals

            summary = _summarize_condition(
                full_mismatch,
                full_radial,
                full_accept,
                args.hazard_start_window,
                args.hazard_duration_windows,
                pre_windows,
                args.last_m,
            )
            spike_by_condition[condition] = summary["spike"]
            focus_delta_by_condition[condition] = summary["focus_delta"]
            accept_by_condition[condition] = summary["accept_mean"]

            pre_idx = list(range(max(1, args.hazard_start_window - pre_windows), args.hazard_start_window))
            haz_idx = list(
                range(args.hazard_start_window, args.hazard_start_window + args.hazard_duration_windows)
            )
            p_val, _, _ = radial_focus_shift_null(
                full_radial_maps,
                ring_np,
                [i - 1 for i in pre_idx],
                [i - 1 for i in haz_idx],
                int(args.sham_n),
                np.random.default_rng(seed),
            )
            focus_p_by_condition[condition] = p_val

            status = "OK"
            report_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "status": status,
                    "spike": summary["spike"],
                    "focus_delta": summary["focus_delta"],
                    "focus_p": p_val,
                    "accept_mean": summary["accept_mean"],
                }
            )

            agg_row = {
                "condition": condition,
                "seed": seed,
                "status": status,
                "spike": summary["spike"],
                "focus_delta": summary["focus_delta"],
                "focus_p": p_val,
                "accept_mean": summary["accept_mean"],
                "focus_delta_vs_sham": 0.0,
            }
            _write_row(agg_writer, agg_row)

            if time.monotonic() - total_start > args.max_seconds_total:
                break

        focus_delta_vs_sham_inject = focus_delta_by_condition.get("inject", 0.0) - focus_delta_by_condition.get("sham", 0.0)
        focus_delta_vs_sham_ablate = focus_delta_by_condition.get("ablate", 0.0) - focus_delta_by_condition.get("sham", 0.0)
        pass_focus = False
        if abs(focus_delta_vs_sham_inject) >= args.focus_delta_min and focus_p_by_condition.get("inject", 1.0) <= args.p_max:
            pass_focus = True
        if abs(focus_delta_vs_sham_ablate) >= args.focus_delta_min and focus_p_by_condition.get("ablate", 1.0) <= args.p_max:
            pass_focus = True

        pass_seed = (
            spike_by_condition.get("control", 0.0) >= args.spike_min
            and pass_focus
            and accept_by_condition.get("control", 0.0) >= args.accept_min
        )

        summary_row = {
            "condition": "seed_summary",
            "seed": seed,
            "status": "PASS" if pass_seed else "FAIL",
            "spike": spike_by_condition.get("control", 0.0),
            "focus_delta": focus_delta_by_condition.get("control", 0.0),
            "focus_p": focus_p_by_condition.get("control", 1.0),
            "accept_mean": accept_by_condition.get("control", 0.0),
        }
        report_rows.append(summary_row)

        if not pass_seed:
            _write_report(out_dir / "PHASE17_DIRECTIONAL_SEMANTICS_REPORT.md", report_rows)
            break

        if time.monotonic() - total_start > args.max_seconds_total:
            break

    _close_writer(agg_writer)
    _write_report(out_dir / "PHASE17_DIRECTIONAL_SEMANTICS_REPORT.md", report_rows)


if __name__ == "__main__":
    main()
