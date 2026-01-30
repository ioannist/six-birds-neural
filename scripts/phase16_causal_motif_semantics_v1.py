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
    apply_k_redistribute_axis_bias_random_in_region,
    apply_k_redistribute_axis_bias_in_region,
    apply_k_redistribute_uniform_in_region,
    apply_sigma_flip,
    apply_sigma_randomize,
    check_k_invariants,
)
from ratchet_gpu.params import Params
from ratchet_gpu.semantics import ring_masks_from_rect
from ratchet_gpu.setpoints import pre_peak_post
from ratchet_gpu.sim import run_sim, _cycle_list
from ratchet_gpu.spatial import k_axis_bias_grid, mismatch_abs_grid, sigma_grid
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


def _region_outside_mean(
    map_tensor: torch.Tensor, region_mask: torch.Tensor, outside_mask: torch.Tensor
) -> Tuple[float, float]:
    region = map_tensor[region_mask]
    outside = map_tensor[outside_mask]
    region_mean = float(region.mean().item()) if region.numel() else 0.0
    outside_mean = float(outside.mean().item()) if outside.numel() else 0.0
    return region_mean, outside_mean


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


def _apply_hazard(
    state: State,
    sigma_mode: str,
    flat_idx: torch.Tensor,
    layers: str,
    rng: torch.Generator,
) -> None:
    if sigma_mode == "none":
        return
    if sigma_mode == "flip":
        apply_sigma_flip(state, flat_idx, layers=layers)
    elif sigma_mode == "random":
        apply_sigma_randomize(state, flat_idx, layers=layers, rng=rng)
    else:
        raise ValueError(f"unknown hazard sigma mode {sigma_mode}")


def _apply_intervention(
    condition: str,
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    interfaces: str,
    rng: torch.Generator,
    axis: int,
) -> None:
    if condition == "ablate":
        apply_k_redistribute_uniform_in_region(
            state, params, flat_idx, interfaces=interfaces, rng=rng
        )
    elif condition == "sham":
        apply_k_redistribute_axis_bias_random_in_region(
            state, params, flat_idx, interfaces=interfaces, rng=rng
        )
    elif condition == "inject":
        apply_k_redistribute_axis_bias_in_region(
            state, params, flat_idx, interfaces=interfaces, axis=axis, rng=rng
        )


def _hazard_indices(hazard_start: int, hazard_duration: int, total_windows: int) -> List[int]:
    return list(
        range(hazard_start, min(total_windows, hazard_start + hazard_duration - 1) + 1)
    )


def _permute_diff_p(
    diffs: List[float],
    shuffle_n: int,
    rng: np.random.Generator,
) -> float:
    if not diffs:
        return 1.0
    obs = float(np.mean(diffs))
    if shuffle_n <= 0:
        return 1.0
    diffs_arr = np.array(diffs, dtype=np.float64)
    null_vals = []
    for _ in range(int(shuffle_n)):
        signs = rng.choice([-1.0, 1.0], size=diffs_arr.shape[0])
        null_vals.append(float(np.mean(diffs_arr * signs)))
    null_arr = np.array(null_vals, dtype=np.float64)
    return float(np.mean(np.abs(null_arr) >= abs(obs)))


def _condition_runner(
    condition: str,
    base_state: State,
    rng_state: torch.Tensor,
    params: Params,
    seed: int,
    out_dir: Path,
    window_steps: int,
    pre_windows: int,
    hazard_start: int,
    hazard_duration: int,
    hazard_sigma: str,
    hazard_layers: str,
    hazard_refresh_each: bool,
    region_idx: torch.Tensor,
    ring_idx: torch.Tensor,
    region_mask: torch.Tensor,
    ring_mask: torch.Tensor,
    outside_mask: torch.Tensor,
    interface: int,
    axis: int,
    snapshot_every: int,
    max_windows: int,
    max_seconds: float,
) -> Tuple[List[Dict[str, Any]], List[float], List[float], List[float]]:
    condition_dir = out_dir / condition
    jsonl_dir = condition_dir / "jsonl"
    npz_dir = condition_dir / "npz"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: List[Dict[str, Any]] = []
    mismatch_series: List[float] = []
    focus_series: List[float] = []
    accept_series: List[float] = []

    hazard_end = hazard_start + hazard_duration - 1
    generator = torch.Generator(device=base_state.device)
    generator.manual_seed(seed + 101)

    state = _clone_state(base_state)
    injury_applied: set[int] = set()

    def _apply_for_window(window_idx: int) -> bool:
        if hazard_start <= window_idx <= hazard_end:
            _apply_hazard(state, hazard_sigma, region_idx, hazard_layers, generator)
            if condition != "control":
                _apply_intervention(
                    condition, state, params, ring_idx, str(interface), generator, axis
                )
            return True
        return False

    if hazard_start == pre_windows + 1:
        _apply_for_window(hazard_start)
        injury_applied.add(hazard_start)

    diag_state = None
    window_idx = 0
    start_time = time.monotonic()
    stop_reason = ""

    def report_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> None:
        nonlocal window_idx, diag_state, stop_reason
        window_idx += 1
        full_window = pre_windows + window_idx
        hazard_active = hazard_start <= full_window <= hazard_end
        intervention_active = hazard_active and condition != "control"
        accepted_window = float(ep_ledger.get("window_accepted", 0))
        proposals_window = float(ep_ledger.get("window_proposals", 0))
        accept_window = accepted_window / proposals_window if proposals_window else 0.0

        mismatch_map = mismatch_abs_grid(st)[interface]
        region_mean, outside_mean = _region_outside_mean(
            mismatch_map, region_mask, outside_mask
        )

        k_bias_map = k_axis_bias_grid(st)[interface]
        ring_mean, ring_outside = _region_outside_mean(k_bias_map, ring_mask, outside_mask)
        focus = ring_mean - ring_outside

        snapshot, diag_state_next = compute_snapshot(
            st, step, ep_ledger, accepted_frac, diag_state
        )
        diag_state = diag_state_next
        snapshot = _slim_snapshot(snapshot)
        snapshot.update(
            {
                "condition": condition,
                "window_index": full_window,
                "hazard_active": hazard_active,
                "intervention_active": intervention_active,
                "injury_applied": full_window in injury_applied,
                "mismatch_region": region_mean,
                "mismatch_outside": outside_mean,
                "k_axis_bias_ring": ring_mean,
                "k_axis_bias_outside": ring_outside,
                "k_axis_bias_focus": focus,
                "accept_window": accept_window,
                "window_steps": int(ep_ledger.get("window_steps", 0)),
                "window_proposals": int(ep_ledger.get("window_proposals", 0)),
            }
        )
        raw_rows.append(snapshot)
        mismatch_series.append(region_mean)
        focus_series.append(focus)
        accept_series.append(accept_window)

        if snapshot_every > 0 and window_idx % snapshot_every == 0:
            maps = {
                "sigma_l0": sigma_grid(st)[0].to(device="cpu").numpy(),
                "k_axis_bias_i0": k_bias_map.to(device="cpu").numpy(),
                "mismatch_i0": mismatch_map.to(device="cpu").numpy(),
            }
            npz_path = npz_dir / f"seed{seed}_win{full_window:04d}.npz"
            np.savez_compressed(npz_path, **maps, step=step, window_index=full_window)

        if time.monotonic() - start_time > max_seconds:
            stop_reason = "FAIL_TIME"

        next_window = full_window + 1
        if hazard_start <= next_window <= hazard_end:
            if hazard_refresh_each or next_window not in injury_applied:
                _apply_for_window(next_window)
                injury_applied.add(next_window)
                ok, msg = check_k_invariants(st, params)
                if not ok:
                    stop_reason = f"FAIL_K_INVARIANT:{msg}"

    def stop_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> bool:
        return bool(stop_reason)

    remaining_windows = max(0, max_windows - pre_windows)
    steps = remaining_windows * window_steps
    run_sim(
        params,
        seed=None,
        steps=steps,
        report_every=window_steps,
        initial_state=state,
        initial_rng_state=rng_state.clone(),
        report_callback=report_cb,
        stop_callback=stop_cb,
    )

    return raw_rows, mismatch_series, focus_series, accept_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16 causal motif semantics runner")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=int, default=150)
    parser.add_argument("--window-sweeps", type=int, default=80)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--hazard-start-window", type=int, default=6)
    parser.add_argument("--hazard-duration-windows", type=int, default=8)
    parser.add_argument("--hazard-rect", required=True)
    parser.add_argument("--hazard-sigma", choices=["random", "flip", "none"], default="random")
    parser.add_argument("--hazard-layers", default="0")
    parser.add_argument("--hazard-refresh-each-window", action="store_true")
    parser.add_argument("--ring-thickness", type=int, default=2)
    parser.add_argument("--ablate-frac", type=float, default=1.0)
    parser.add_argument("--sham-n", type=int, default=100)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--spike-min", type=float, default=0.01)
    parser.add_argument("--focus-delta-min", type=float, default=0.005)
    parser.add_argument("--effect-recovery-min", type=float, default=0.05)
    parser.add_argument("--effect-spike-min", type=float, default=0.02)
    parser.add_argument("--p-max", type=float, default=0.10)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)

    preset = _load_preset(Path(args.preset))
    params = _as_params(preset, {"device": args.device})

    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = int(np.prod(params.shape))
    expected_props = _expected_proposals_per_step(N, str(args.device), params.kernel_weights)
    window_steps = max(1, int(math.ceil(args.window_sweeps * N / expected_props)))
    burn_steps = max(1, int(math.ceil(args.burn_in_sweeps * N / expected_props)))

    pre_windows = max(0, args.hazard_start_window - 1)
    pre_steps = pre_windows * window_steps

    region_mask_np, ring_mask_np, outside_mask_np = ring_masks_from_rect(
        args.hazard_rect, params.shape, width=args.ring_thickness
    )
    region_mask = torch.as_tensor(region_mask_np, device=params.resolved_device())
    ring_mask = torch.as_tensor(ring_mask_np, device=params.resolved_device())
    outside_mask = torch.as_tensor(outside_mask_np, device=params.resolved_device())

    region_idx = torch.nonzero(region_mask.view(-1), as_tuple=False).flatten()
    ring_idx = torch.nonzero(ring_mask.view(-1), as_tuple=False).flatten()
    region_idx = region_idx.to(device=params.resolved_device())
    ring_idx = ring_idx.to(device=params.resolved_device())

    start_time_total = time.monotonic()
    seed_results: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}

    for seed in seeds:
        if time.monotonic() - start_time_total > args.max_seconds_total:
            break

        prep_summary = run_sim(
            params,
            seed=seed,
            steps=burn_steps + pre_steps,
            report_every=window_steps,
            return_state=True,
        )
        base_state: State = prep_summary["state"]
        rng_state: torch.Tensor = prep_summary["rng_state"]

        pre_mismatch: List[float] = []
        pre_focus: List[float] = []
        pre_accept: List[float] = []
        pre_records: List[Dict[str, Any]] = []

        def pre_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> None:
            window_idx = len(pre_mismatch) + 1
            mismatch_map = mismatch_abs_grid(st)[0]
            region_mean, outside_mean = _region_outside_mean(
                mismatch_map, region_mask, outside_mask
            )
            k_bias_map = k_axis_bias_grid(st)[0]
            ring_mean, ring_outside = _region_outside_mean(
                k_bias_map, ring_mask, outside_mask
            )
            focus = ring_mean - ring_outside
            accepted_window = float(ep_ledger.get("window_accepted", 0))
            proposals_window = float(ep_ledger.get("window_proposals", 0))
            accept_window = accepted_window / proposals_window if proposals_window else 0.0
            pre_mismatch.append(region_mean)
            pre_focus.append(focus)
            pre_accept.append(accept_window)
            pre_records.append(
                {
                    "seed": seed,
                    "condition": "pre",
                    "window_index": window_idx,
                    "hazard_active": False,
                    "intervention_active": False,
                    "injury_applied": False,
                    "mismatch_region": region_mean,
                    "mismatch_outside": outside_mean,
                    "k_axis_bias_focus": focus,
                    "accept_window": accept_window,
                }
            )

        if pre_steps > 0:
            run_sim(
                params,
                seed=None,
                steps=pre_steps,
                report_every=window_steps,
                initial_state=_clone_state(base_state),
                initial_rng_state=rng_state.clone(),
                report_callback=pre_cb,
            )

        cond_results: Dict[str, Dict[str, Any]] = {}
        for condition in ("control", "ablate", "sham"):
            if time.monotonic() - start_time_total > args.max_seconds_total:
                break
            rows, mismatch_series, focus_series, accept_series = _condition_runner(
                condition,
                base_state,
                rng_state,
                params,
                seed,
                out_dir,
                window_steps,
                pre_windows,
                args.hazard_start_window,
                args.hazard_duration_windows,
                args.hazard_sigma,
                args.hazard_layers,
                args.hazard_refresh_each_window,
                region_idx,
                ring_idx,
                region_mask,
                ring_mask,
                outside_mask,
                0,
                0,
                args.snapshot_every_windows,
                args.max_windows,
                args.max_seconds_per_run,
            )
            full_mismatch = pre_mismatch + mismatch_series
            full_focus = pre_focus + focus_series
            full_accept = pre_accept + accept_series
            pre, peak, post = pre_peak_post(
                full_mismatch, args.hazard_start_window, args.hazard_duration_windows, 5
            )
            spike = peak - pre
            recovery = (peak - post) / max(spike, 1e-12) if spike > 0 else 0.0
            haz_idx = _hazard_indices(
                args.hazard_start_window, args.hazard_duration_windows, len(full_focus)
            )
            focus_pre = float(np.mean([full_focus[i - 1] for i in range(1, args.hazard_start_window)])) if args.hazard_start_window > 1 else 0.0
            focus_haz = float(np.mean([full_focus[i - 1] for i in haz_idx])) if haz_idx else focus_pre
            focus_delta = focus_haz - focus_pre
            accept_mean = float(np.mean(full_accept[-5:])) if full_accept else 0.0
            cond_results[condition] = {
                "rows": rows,
                "mismatch_series": full_mismatch,
                "focus_series": full_focus,
                "accept_mean": accept_mean,
                "spike": spike,
                "recovery": recovery,
                "focus_delta": focus_delta,
            }

        if len(cond_results) < 3:
            break

        control = cond_results["control"]
        ablate = cond_results["ablate"]
        sham = cond_results["sham"]

        effect_recovery = control["recovery"] - ablate["recovery"]
        effect_spike = ablate["spike"] - control["spike"]

        haz_idx = _hazard_indices(
            args.hazard_start_window, args.hazard_duration_windows, len(control["mismatch_series"])
        )
        diffs = [
            ablate["mismatch_series"][i - 1] - sham["mismatch_series"][i - 1]
            for i in haz_idx
        ]
        p_effect = _permute_diff_p(diffs, args.sham_n, np.random.default_rng(seed + 123))

        pass_seed = (
            control["spike"] >= args.spike_min
            and ablate["focus_delta"] <= -args.focus_delta_min
            and (abs(effect_recovery) >= args.effect_recovery_min or abs(effect_spike) >= args.effect_spike_min)
            and p_effect <= args.p_max
            and control["accept_mean"] >= args.accept_min
            and ablate["accept_mean"] >= args.accept_min
        )

        status = "PASS" if pass_seed else "FAIL"
        status_counts[status] = status_counts.get(status, 0) + 1
        seed_results.append(
            {
                "seed": seed,
                "status": status,
                "spike_control": control["spike"],
                "spike_ablate": ablate["spike"],
                "recovery_control": control["recovery"],
                "recovery_ablate": ablate["recovery"],
                "focus_delta_ablate": ablate["focus_delta"],
                "effect_recovery": effect_recovery,
                "effect_spike": effect_spike,
                "p_effect": p_effect,
                "accept_control": control["accept_mean"],
                "accept_ablate": ablate["accept_mean"],
            }
        )

        for condition in ("control", "ablate", "sham"):
            condition_dir = out_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            raw_path = condition_dir / "raw.csv"
            with raw_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(cond_results[condition]["rows"][0].keys()))
                writer.writeheader()
                writer.writerows(cond_results[condition]["rows"])
            agg_path = condition_dir / "agg.csv"
            with agg_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(seed_results[-1].keys()))
                writer.writeheader()
                writer.writerow(seed_results[-1])
            progress_path = condition_dir / "progress.csv"
            with progress_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(cond_results[condition]["rows"][0].keys()))
                writer.writeheader()
                writer.writerows(cond_results[condition]["rows"])

        report_path = out_dir / "PHASE16_CAUSAL_SEMANTICS_REPORT.md"
        with report_path.open("w") as handle:
            handle.write("# Phase 16 causal motif semantics\n\n")
            handle.write("| seed | status | spike_control | spike_ablate | recovery_control | recovery_ablate | focus_delta_ablate | effect_recovery | effect_spike | p_effect |\n")
            handle.write("| ---: | :----- | ------------: | -----------: | ---------------: | --------------: | -----------------: | -------------: | ----------: | -------: |\n")
            for row in seed_results:
                handle.write(
                    f"| {row['seed']} | {row['status']} | {row['spike_control']:.6g} | {row['spike_ablate']:.6g} | "
                    f"{row['recovery_control']:.6g} | {row['recovery_ablate']:.6g} | {row['focus_delta_ablate']:.6g} | "
                    f"{row['effect_recovery']:.6g} | {row['effect_spike']:.6g} | {row['p_effect']:.6g} |\n"
                )

        if seed == seeds[0] and status == "FAIL":
            print("STOP_REASON=SEED1_FAIL")
            break

    agg_path = out_dir / "agg.csv"
    if seed_results:
        with agg_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(seed_results[0].keys()))
            writer.writeheader()
            writer.writerows(seed_results)

    if args.progress:
        print(f"STATUS_COUNTS {status_counts}")


if __name__ == "__main__":
    main()
