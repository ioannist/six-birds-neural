#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.interventions import (
    apply_k_redistribute_radial_inward_in_ring,
    apply_k_redistribute_radial_outward_in_ring,
    apply_k_redistribute_radial_random_in_ring,
    apply_sigma_flip,
    apply_sigma_randomize,
    check_k_invariants,
    parse_rect,
)
from ratchet_gpu.motifs import (
    MotifBins,
    build_bins,
    jsd,
    motif_dictionary_eval,
    motif_histogram,
    motif_ids,
    top_n_coverage,
)
from ratchet_gpu.params import Params
from ratchet_gpu.semantics import hazard_center, ring_masks_from_rect
from ratchet_gpu.sim import run_sim
from ratchet_gpu.spatial import compute_spatial_maps, finite_check, mismatch_abs_grid
from ratchet_gpu.state import State

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


def _validate_hazard_schedule(start: int, duration: int, max_windows: int) -> None:
    if start < 1:
        raise ValueError("hazard_start_window must be >= 1")
    if duration < 1:
        raise ValueError("hazard_duration_windows must be >= 1")
    if start + duration - 1 > max_windows:
        raise ValueError("hazard window must fit within max_windows")


def _parse_motif_features(value: str) -> List[str]:
    feats = [k.strip() for k in value.split(",") if k.strip()]
    for feat in feats:
        if feat not in {"k_axis_bias", "k_entropy"}:
            raise ValueError(f"unsupported motif feature {feat}")
    return feats


def _validate_bins(bins_axis: int, bins_entropy: int) -> None:
    if bins_axis < 2:
        raise ValueError("bins_axis_bias must be >= 2")
    if bins_entropy < 2:
        raise ValueError("bins_entropy must be >= 2")


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
    center: Tuple[float, float],
    interfaces: List[int],
    strength: float,
    rng: torch.Generator,
) -> None:
    if condition == "inject_in":
        apply_k_redistribute_radial_inward_in_ring(
            state,
            params,
            ring_idx,
            center=center,
            interfaces=interfaces,
            strength=strength,
            rng=rng,
        )
    elif condition == "inject_out":
        apply_k_redistribute_radial_outward_in_ring(
            state,
            params,
            ring_idx,
            center=center,
            interfaces=interfaces,
            strength=strength,
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


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _select_map(maps_dict: Dict[str, torch.Tensor], key: str, interface: int) -> np.ndarray:
    if key not in maps_dict:
        raise ValueError(f"missing spatial map {key}")
    arr = maps_dict[key]
    if key in {"k_axis_bias", "k_entropy", "k_r2", "mismatch"}:
        arr = arr[interface]
    return arr.detach().cpu().numpy()


def _run_pre_windows(
    params: Params,
    state: State,
    seed: int,
    steps: int,
    report_every: int,
    interface: int,
    region_mask: torch.Tensor,
    motif_features: List[str],
) -> Tuple[State, torch.Tensor | None, Dict[str, List[np.ndarray]], List[float], List[float]]:
    diag_state = None
    feature_store: Dict[str, List[np.ndarray]] = {k: [] for k in motif_features}
    mismatch_vals: List[float] = []
    accept_vals: List[float] = []
    snapshot_cache: Dict[str, Any] | None = None

    def pre_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> None:
        nonlocal diag_state, snapshot_cache
        snap, diag_state = compute_snapshot(st, step, ep_ledger, accepted_frac, diag_state)
        snapshot_cache = snap
        maps_dict = compute_spatial_maps(st, ["k_axis_bias", "k_entropy", "mismatch"])
        ok, _ = finite_check(maps_dict)
        if not ok:
            raise RuntimeError("non-finite map during pre window")
        for feat in motif_features:
            feature_store[feat].append(_select_map(maps_dict, feat, interface))
        mismatch = maps_dict["mismatch"][interface]
        mismatch_region = float(mismatch[region_mask].mean().item()) if region_mask.any() else 0.0
        mismatch_vals.append(mismatch_region)
        accept_vals.append(_get_accept(snap))

    summary = run_sim(
        params,
        seed=seed,
        steps=steps,
        report_every=report_every,
        report_callback=pre_cb,
        initial_state=state,
        return_state=True,
    )
    state = summary["state"]
    rng_state = summary.get("rng_state")
    return state, rng_state, feature_store, mismatch_vals, accept_vals


def _baseline_ring_hists(
    baseline_features: Dict[str, List[np.ndarray]],
    motif_features: List[str],
    bins: MotifBins,
    ring_mask: np.ndarray,
    num_motifs: int,
) -> np.ndarray:
    if not motif_features:
        return np.zeros((0, num_motifs), dtype=np.float64)
    total_windows = len(baseline_features[motif_features[0]])
    hists: List[np.ndarray] = []
    for t in range(total_windows):
        features_t = {feat: baseline_features[feat][t] for feat in motif_features}
        ids = motif_ids(features_t, bins)
        hists.append(motif_histogram(ids, ring_mask, num_motifs))
    if not hists:
        return np.zeros((0, num_motifs), dtype=np.float64)
    return np.stack(hists, axis=0)


def _hazard_active_for_window(window_idx: int, hazard_start: int, hazard_duration: int) -> bool:
    if hazard_duration <= 0:
        return False
    return hazard_start <= window_idx <= hazard_start + hazard_duration - 1


def _run_condition(
    condition: str,
    state: State,
    params: Params,
    seed: int,
    rng_state: torch.Tensor | None,
    window_steps: int,
    max_windows: int,
    hazard_start_window: int,
    hazard_duration: int,
    hazard_sigma: str,
    hazard_layers: str,
    hazard_refresh_each_window: bool,
    hazard_flat_idx: torch.Tensor,
    ring_idx: torch.Tensor,
    interfaces: List[int],
    center: Tuple[float, float],
    strength: float,
    interface: int,
    region_mask: torch.Tensor,
    ring_mask: np.ndarray,
    outside_mask: np.ndarray,
    bins: MotifBins,
    motif_features: List[str],
    num_motifs: int,
    out_dir: Path,
    snapshot_every: int,
    max_seconds: float,
    progress: bool,
    rng_state_in: torch.Tensor | None,
    window_offset: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    condition_dir = out_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "jsonl").mkdir(parents=True, exist_ok=True)
    npz_dir = condition_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

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
        ],
    )
    progress_writer = _build_csv_writer(
        condition_dir / "progress.csv",
        [
            "condition",
            "seed",
            "window_index",
            "hazard_active",
            "acceptedFracWindow",
            "mismatch_region",
        ],
    )
    jsonl_path = condition_dir / "jsonl" / f"{condition}_seed{seed}.jsonl"
    jsonl_handle = jsonl_path.open("w")

    diag_state = None
    window_idx = 0
    hazard_active_next = False
    sigma_backup: torch.Tensor | None = None
    ring_hists: List[np.ndarray] = []
    outside_hists: List[np.ndarray] = []
    mismatch_region_vals: List[float] = []
    accept_vals: List[float] = []

    def _backup_sigma(st: State) -> None:
        nonlocal sigma_backup
        if sigma_backup is None:
            sigma_backup = st.sigma.clone()

    def _restore_sigma(st: State) -> None:
        nonlocal sigma_backup
        if sigma_backup is not None:
            st.sigma.copy_(sigma_backup)
            sigma_backup = None

    def report_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> None:
        nonlocal window_idx, diag_state, hazard_active_next
        window_idx += 1
        window_idx_global = window_offset + window_idx
        hazard_active = _hazard_active_for_window(window_idx_global, hazard_start_window, hazard_duration)
        snap, diag_state = compute_snapshot(st, step, ep_ledger, accepted_frac, diag_state)
        slim = _slim_snapshot(snap)
        slim.update(
            {
                "seed": seed,
                "condition": condition,
                "window": window_idx_global,
                "hazard_active": hazard_active,
            }
        )
        jsonl_handle.write(to_json_line(slim) + "\n")
        jsonl_handle.flush()

        maps_dict = compute_spatial_maps(st, ["k_axis_bias", "k_entropy", "mismatch"])
        ok, _ = finite_check(maps_dict)
        if not ok:
            raise RuntimeError("non-finite map during hazard run")
        features = {feat: _select_map(maps_dict, feat, interface) for feat in motif_features}
        ids = motif_ids(features, bins)
        ring_hists.append(motif_histogram(ids, ring_mask, num_motifs))
        outside_hists.append(motif_histogram(ids, outside_mask, num_motifs))

        mismatch_map = maps_dict["mismatch"][interface]
        mismatch_region = float(mismatch_map[region_mask].mean().item()) if region_mask.any() else 0.0
        mismatch_outside = float(mismatch_map[~region_mask].mean().item()) if (~region_mask).any() else 0.0
        mismatch_region_vals.append(mismatch_region)
        accept_window = _get_accept(snap)
        accept_vals.append(accept_window)

        _write_row(
            raw_writer,
            {
                "condition": condition,
                "seed": seed,
                "window_index": window_idx_global,
                "step": step,
                "hazard_active": hazard_active,
                "acceptedFracWindow": accept_window,
                "mismatch_region": mismatch_region,
                "mismatch_outside": mismatch_outside,
            },
        )
        _write_row(
            progress_writer,
            {
                "condition": condition,
                "seed": seed,
                "window_index": window_idx_global,
                "hazard_active": hazard_active,
                "acceptedFracWindow": accept_window,
                "mismatch_region": mismatch_region,
            },
        )

        if snapshot_every > 0 and window_idx % snapshot_every == 0:
            payload = {
                "k_axis_bias_i0": features["k_axis_bias"],
                "k_entropy_i0": features["k_entropy"],
                "mismatch_i0": _select_map(maps_dict, "mismatch", interface),
            }
            np.savez(npz_dir / f"seed{seed}_win{window_idx:04d}.npz", **payload)

        next_window_idx = window_idx + 1
        next_window_idx_global = window_offset + next_window_idx
        hazard_active_next_new = _hazard_active_for_window(
            next_window_idx_global, hazard_start_window, hazard_duration
        )
        if hazard_active_next_new:
            if hazard_refresh_each_window or not hazard_active_next:
                _backup_sigma(st)
                _apply_hazard(st, hazard_sigma, hazard_flat_idx, hazard_layers, torch.Generator(device=st.device))
                if condition != "control":
                    _apply_intervention(
                        condition,
                        st,
                        params,
                        ring_idx,
                        center,
                        interfaces,
                        strength,
                        torch.Generator(device=st.device),
                    )
                    ok, msg = check_k_invariants(st, params)
                    if not ok:
                        raise RuntimeError(f"K invariants failed: {msg}")
        elif hazard_active_next:
            _restore_sigma(st)
        hazard_active_next = hazard_active_next_new

    def stop_cb(*_args: Any) -> bool:
        return window_idx >= max_windows or time.monotonic() >= max_seconds

    if _hazard_active_for_window(window_offset + 1, hazard_start_window, hazard_duration):
        _backup_sigma(state)
        _apply_hazard(state, hazard_sigma, hazard_flat_idx, hazard_layers, torch.Generator(device=state.device))
        if condition != "control":
            _apply_intervention(
                condition,
                state,
                params,
                ring_idx,
                center,
                interfaces,
                strength,
                torch.Generator(device=state.device),
            )
            ok, msg = check_k_invariants(state, params)
            if not ok:
                raise RuntimeError(f"K invariants failed: {msg}")
        hazard_active_next = True

    seed_for_run = seed if rng_state_in is None else None
    summary = run_sim(
        params,
        seed=seed_for_run,
        steps=window_steps * max_windows,
        report_every=window_steps,
        report_callback=report_cb,
        stop_callback=stop_cb,
        initial_state=state,
        initial_rng_state=rng_state_in,
        return_state=True,
    )
    jsonl_handle.close()
    _close_writer(raw_writer)
    _close_writer(progress_writer)
    rng_state_out = summary.get("rng_state")
    return ring_hists, outside_hists, mismatch_region_vals, accept_vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18 motif dictionary semantics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=int, default=150)
    parser.add_argument("--window-sweeps", type=int, default=80)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--hazard-start-window", type=int, default=6)
    parser.add_argument("--hazard-duration-windows", type=int, default=8)
    parser.add_argument("--hazard-rect", required=True)
    parser.add_argument("--hazard-sigma", default="random", choices=["random", "flip", "none"])
    parser.add_argument("--hazard-layers", default="0")
    parser.add_argument("--hazard-refresh-each-window", action="store_true")
    parser.add_argument("--ring-thickness", type=int, default=2)
    parser.add_argument("--motif-interface", type=int, default=0)
    parser.add_argument("--motif-features", default="k_axis_bias,k_entropy")
    parser.add_argument("--bins-axis-bias", type=int, default=7)
    parser.add_argument("--bins-entropy", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--shuffle-n", type=int, default=200)
    parser.add_argument("--spike-min", type=float, default=0.01)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--jsd-inout-min", type=float, default=0.01)
    parser.add_argument("--dict-delta-min", type=float, default=0.005)
    parser.add_argument("--p-max", type=float, default=0.10)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("no seeds specified")
    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)
    if args.ring_thickness < 1:
        raise ValueError("ring_thickness must be >= 1")
    _validate_bins(args.bins_axis_bias, args.bins_entropy)
    motif_features = _parse_motif_features(args.motif_features)

    preset = _load_preset(Path(args.preset))
    params = _as_params(preset, {"device": args.device})
    params = Params(**{**params.__dict__, "p3_on": False, "p6_on": False})
    device = params.resolved_device()
    device_str = str(device)

    if len(params.shape) != 2:
        raise ValueError("Phase18 expects 2D lattice shape")
    shape = params.shape
    region_np, ring_np, outside_np = ring_masks_from_rect(args.hazard_rect, shape, args.ring_thickness)
    center = hazard_center(args.hazard_rect, shape)
    hazard_mask, hazard_flat_idx = parse_rect(args.hazard_rect, shape)
    ring_idx = torch.as_tensor(np.flatnonzero(ring_np), dtype=torch.long, device=device)
    region_mask = hazard_mask.to(dtype=torch.bool, device=device)

    interface_idx = int(args.motif_interface)
    interfaces = [interface_idx]

    N = math.prod(params.shape)
    expected = _expected_proposals_per_step(N, device_str, params.kernel_weights)
    burn_steps = int(math.ceil(args.burn_in_sweeps * N / expected))
    window_steps = int(math.ceil(args.window_sweeps * N / expected))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / "agg.csv"
    agg_fields = [
        "seed",
        "status",
        "best_condition",
        "spike_control",
        "jsd_inout",
        "dict_delta",
        "dict_p",
        "dict_eval_scope",
        "dict_shuffle_n",
        "dict_shuffle_mode",
        "accept_mean_control",
        "coverage_pre",
        "coverage_hazard",
    ]
    agg_writer = _build_csv_writer(agg_path, agg_fields)

    report_rows: List[Dict[str, Any]] = []
    total_start = time.monotonic()

    for seed in seeds:
        if time.monotonic() - total_start > args.max_seconds_total:
            break

        base_state = State.initialize(params, seed=seed)
        rng_state = None

        pre_windows = max(0, args.hazard_start_window - 1)
        if pre_windows > 0:
            pre_steps = window_steps * pre_windows
            base_state, rng_state, baseline_features, pre_mismatch, pre_accept = _run_pre_windows(
                params,
                base_state,
                seed,
                pre_steps,
                window_steps,
                interface_idx,
                region_mask,
                motif_features,
            )
        else:
            baseline_features = {k: [] for k in motif_features}
            pre_mismatch = []
            pre_accept = []

        if not baseline_features[motif_features[0]]:
            raise RuntimeError("no baseline features collected before hazard")

        bins_by_key = {
            "k_axis_bias": args.bins_axis_bias,
            "k_entropy": args.bins_entropy,
        }
        bins_by_key = {k: bins_by_key[k] for k in motif_features}
        bins = build_bins(baseline_features, bins_by_key)
        num_motifs = 1
        for key in bins.edges:
            num_motifs *= bins.bins[key]

        base_state = _clone_state(base_state)

        conditions = ["control", "inject_in", "inject_out"]
        metrics_by_condition: Dict[str, Dict[str, float]] = {}
        ring_pre_hists = _baseline_ring_hists(
            baseline_features,
            motif_features,
            bins,
            ring_np,
            num_motifs,
        )
        ring_pre_mean = ring_pre_hists.mean(axis=0) if ring_pre_hists.size else np.zeros((num_motifs,))
        coverage_pre = top_n_coverage(ring_pre_mean, args.top_n)

        for condition in conditions:
            ring_hists, outside_hists, mismatch_region_vals, accept_vals = _run_condition(
                condition,
                _clone_state(base_state),
                params,
                seed,
                rng_state,
                window_steps,
                args.max_windows,
                args.hazard_start_window,
                args.hazard_duration_windows,
                args.hazard_sigma,
                args.hazard_layers,
                args.hazard_refresh_each_window,
                hazard_flat_idx,
                ring_idx,
                interfaces,
                center,
                1.0,
                interface_idx,
                region_mask,
                ring_np,
                outside_np,
                bins,
                motif_features,
                num_motifs,
                out_dir,
                args.snapshot_every_windows,
                time.monotonic() + args.max_seconds_per_run,
                args.progress,
                rng_state,
                pre_windows,
            )

            ring_arr = np.stack(ring_hists, axis=0) if ring_hists else np.zeros((0, num_motifs))
            outside_arr = np.stack(outside_hists, axis=0) if outside_hists else np.zeros((0, num_motifs))
            hazard_count = min(args.hazard_duration_windows, ring_arr.shape[0])
            ring_hazard = ring_arr[:hazard_count] if hazard_count else np.zeros((0, num_motifs))
            outside_hazard = outside_arr[:hazard_count] if hazard_count else np.zeros((0, num_motifs))
            ring_mean = ring_hazard.mean(axis=0) if ring_hazard.size else np.zeros((num_motifs,))
            outside_mean = outside_hazard.mean(axis=0) if outside_hazard.size else np.zeros((num_motifs,))
            jsd_in_out = jsd(ring_mean, outside_mean)
            dict_eval = motif_dictionary_eval(
                ring_hazard,
                outside_hazard,
                args.shuffle_n,
                np.random.default_rng(seed),
            )

            pre_mean = float(np.mean(pre_mismatch)) if pre_mismatch else 0.0
            peak = float(np.max(mismatch_region_vals)) if mismatch_region_vals else pre_mean
            spike = max(0.0, peak - pre_mean)
            accept_mean = _mean(accept_vals[-5:])

            coverage_haz = top_n_coverage(ring_mean, args.top_n)

            metrics_by_condition[condition] = {
                "spike": spike,
                "jsd_inout": float(jsd_in_out),
                "dict_delta": float(dict_eval["dict_delta"]),
                "dict_p": float(dict_eval["dict_p"]),
                "accept_mean": accept_mean,
                "coverage_pre": coverage_pre,
                "coverage_hazard": coverage_haz,
            }

        control_metrics = metrics_by_condition["control"]
        best_condition = "control"
        best_metrics = control_metrics
        for condition in ["inject_in", "inject_out"]:
            if metrics_by_condition[condition]["dict_delta"] > best_metrics["dict_delta"]:
                best_condition = condition
                best_metrics = metrics_by_condition[condition]

        status = "FAIL"
        if control_metrics["accept_mean"] >= args.accept_min and control_metrics["spike"] >= args.spike_min:
            if (
                best_metrics["jsd_inout"] >= args.jsd_inout_min
                and best_metrics["dict_delta"] >= args.dict_delta_min
                and best_metrics["dict_p"] <= args.p_max
            ):
                status = "PASS"

        agg_row = {
            "seed": seed,
            "status": status,
            "best_condition": best_condition,
            "spike_control": control_metrics["spike"],
            "jsd_inout": best_metrics["jsd_inout"],
            "dict_delta": best_metrics["dict_delta"],
            "dict_p": best_metrics["dict_p"],
            "dict_eval_scope": "hazard_only",
            "dict_shuffle_n": int(args.shuffle_n),
            "dict_shuffle_mode": "label_swap",
            "accept_mean_control": control_metrics["accept_mean"],
            "coverage_pre": control_metrics["coverage_pre"],
            "coverage_hazard": best_metrics["coverage_hazard"],
        }
        _write_row(agg_writer, agg_row)
        report_rows.append(agg_row)

        if seed == seeds[0] and status != "PASS":
            break

    _close_writer(agg_writer)
    report_path = out_dir / "PHASE18_MOTIF_DICTIONARY_REPORT.md"
    _write_report(report_path, report_rows, args)


def _write_report(report_path: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# Phase 18 Motif Dictionary Semantics\n\n"]
    lines.append("## Dictionary evaluation\n\n")
    lines.append(f"- dict_eval_scope: hazard_only\n")
    lines.append(f"- dict_shuffle_n: {args.shuffle_n}\n")
    lines.append("- dict_shuffle_mode: label_swap\n")
    lines.append(f"- hazard_start_window: {args.hazard_start_window}\n")
    lines.append(f"- hazard_duration_windows: {args.hazard_duration_windows}\n\n")
    lines.append("| seed | status | best_condition | spike_control | jsd_inout | dict_delta | dict_p | accept |\n")
    lines.append("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['status']} | {row['best_condition']} | "
            f"{row['spike_control']:.4g} | {row['jsd_inout']:.4g} | "
            f"{row['dict_delta']:.4g} | {row['dict_p']:.4g} | "
            f"{row['accept_mean_control']:.4g} |\n"
        )
    report_path.write_text("".join(lines))


if __name__ == "__main__":
    main()
