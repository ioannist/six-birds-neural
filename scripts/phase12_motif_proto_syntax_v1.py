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

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.interventions import apply_sigma_flip, apply_sigma_randomize, parse_rect
from ratchet_gpu.motifs import (
    build_bins,
    conditional_entropy_from_counts,
    jsd,
    l1_distance,
    motif_entropy,
    motif_hist,
    motif_ids,
    propagation_score,
    split_condition_counts,
    top_n_coverage,
    topk_transition_mass,
    transition_counts_over_time,
    transition_jsd,
)
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim, _cycle_list
from ratchet_gpu.state import State
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


HEAVY_KEYS = {"strobe_current_map_items_window", "strobe_currents_window"}
SUPPORTED_FEATURES = {"k_axis_bias", "k_entropy", "k_r2", "w_mass"}


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


def _parse_keys(value: str) -> List[str]:
    return [k.strip() for k in value.split(",") if k.strip()]


def _validate_hazard_schedule(start: int, duration: int, max_windows: int) -> None:
    if start < 1:
        raise ValueError("hazard_start_window must be >= 1")
    if duration < 1:
        raise ValueError("hazard_duration_windows must be >= 1")
    if start + duration - 1 > max_windows:
        raise ValueError("hazard window must fit within max_windows")


def _validate_features(features: List[str]) -> None:
    if not features:
        raise ValueError("motif_features must not be empty")
    unknown = sorted(set(features) - SUPPORTED_FEATURES)
    if unknown:
        raise ValueError(f"unsupported motif features: {', '.join(unknown)}")


def _validate_bins(bins_axis_bias: int, bins_entropy: int) -> None:
    if bins_axis_bias < 2 or bins_entropy < 2:
        raise ValueError("bins must be >= 2")


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


def _select_map(
    maps_dict: Dict[str, torch.Tensor],
    key: str,
    interface_idx: int,
    layer_idx: int,
) -> Tuple[str, np.ndarray]:
    tensor = maps_dict.get(key)
    if tensor is None:
        raise KeyError(f"map {key} not found")
    if key in {"k_axis_bias", "k_entropy", "k_r2", "mismatch"}:
        if tensor.shape[0] <= interface_idx:
            raise ValueError(f"interface index {interface_idx} out of range for {key}")
        data = tensor[interface_idx]
        name = f"{key}_i{interface_idx}"
    else:
        if tensor.shape[0] <= layer_idx:
            raise ValueError(f"layer index {layer_idx} out of range for {key}")
        data = tensor[layer_idx]
        name = f"{key}_l{layer_idx}"
    return name, data.detach().cpu().numpy()


def _feature_bins(features: List[str], bins_axis_bias: int, bins_entropy: int) -> Dict[str, int]:
    bins: Dict[str, int] = {}
    for feat in features:
        if feat in {"k_axis_bias", "w_axis_bias"}:
            bins[feat] = bins_axis_bias
        elif feat in {"k_entropy", "w_entropy"}:
            bins[feat] = bins_entropy
        else:
            bins[feat] = bins_entropy
    return bins


def run_seed(
    params: Params,
    seed: int,
    out_dir: Path,
    burn_sweeps: float,
    window_sweeps: float,
    max_windows: int,
    snapshot_every: int,
    hazard_start: int,
    hazard_duration: int,
    hazard_rect: str,
    hazard_sigma: str,
    hazard_layers: List[int],
    hazard_refresh_each_window: bool,
    motif_interface: int,
    motif_features: List[str],
    bins_axis_bias: int,
    bins_entropy: int,
    top_n: int,
    top_trans_k: int,
    shift_max: int,
    prop_top_m: int,
    coverage_min: float,
    jsd_min: float,
    jsd_trans_min: float,
    prop_min: float,
    accept_min: float,
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
    cycle: List[str],
    resume: bool,
) -> Dict[str, Any]:
    raw_path = out_dir / "raw.csv"
    agg_path = out_dir / "agg.csv"
    progress_path = out_dir / "progress.csv"
    jsonl_dir = out_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = out_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    if resume and agg_path.exists():
        with agg_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            if str(row.get("seed")) == str(seed) and row.get("status") not in {"RUNNING", ""}:
                return row

    shape = params.shape
    H, W = int(shape[-2]), int(shape[-1])
    mask, flat_idx = parse_rect(hazard_rect, (H, W))
    mask_t = mask.to(device=params.device)

    N = int(np.prod(params.shape))
    expected = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected))
    window_steps = int(math.ceil(window_sweeps * N / expected))

    jsonl_path = jsonl_dir / f"seed{seed}.jsonl"
    jsonl_handle = jsonl_path.open("a", encoding="utf-8")

    if not raw_path.exists():
        with raw_path.open("w", encoding="utf-8") as rh:
            writer = csv.writer(rh)
            writer.writerow(
                [
                    "seed",
                    "window",
                    "hazard_active",
                    "ep_rate",
                    "accept_window",
                    "mismatch_region",
                    "mismatch_outside",
                    "motif_entropy",
                    "topN_coverage",
                    "motif_count",
                ]
            )
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            writer = csv.writer(ph)
            writer.writerow(
                ["seed", "window", "hazard_active", "ep_rate", "accept_window", "mismatch_region"]
            )

    run_start = time.monotonic()
    window_idx = 0
    status = "RUNNING"
    hazard_active_next = False
    sigma_backup: torch.Tensor | None = None

    baseline_features: Dict[str, List[np.ndarray]] = {k: [] for k in motif_features}
    window_records: List[Dict[str, Any]] = []
    accept_vals: List[float] = []

    def _apply_hazard(state: Any) -> None:
        if hazard_sigma == "flip":
            apply_sigma_flip(state, flat_idx, layers=hazard_layers)
        elif hazard_sigma == "random":
            apply_sigma_randomize(state, flat_idx, layers=hazard_layers)
        else:
            raise ValueError(f"Unknown hazard sigma mode: {hazard_sigma}")

    def _backup_sigma(state: Any) -> None:
        nonlocal sigma_backup
        if sigma_backup is None:
            sigma_backup = state.sigma.clone()

    def _restore_sigma(state: Any) -> None:
        nonlocal sigma_backup
        if sigma_backup is not None:
            state.sigma.copy_(sigma_backup)
            sigma_backup = None

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal window_idx, status, hazard_active_next
        now = time.monotonic()
        if now - start_total > max_seconds_total:
            status = "FAIL_TIME"
            return
        if now - run_start > max_seconds_per_run:
            status = "FAIL_TIME"
            return

        window_idx += 1
        is_burn = step <= burn_steps
        if is_burn:
            return

        hazard_active = hazard_start <= window_idx <= (hazard_start + hazard_duration - 1)
        snapshot, _ = compute_snapshot(state, step, ep_ledger, accepted_frac, None)
        slim = _slim_snapshot(snapshot)
        slim["seed"] = seed
        slim["window"] = window_idx
        slim["hazard_active"] = hazard_active

        map_keys = {"sigma", "k_axis_bias", "k_entropy", "mismatch"}
        map_keys.update(motif_features)
        maps_dict = compute_spatial_maps(state, sorted(map_keys))
        ok, _ = finite_check(maps_dict)
        if not ok:
            status = "FAIL_NAN_MAP"
            return

        mismatch_region, mismatch_outside = _region_mean(maps_dict["mismatch"], mask_t)

        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        accept_window = float(snapshot.get("acceptedFracWindow", accepted_frac))

        feature_arrays: Dict[str, np.ndarray] = {}
        for feat in motif_features:
            _name, arr = _select_map(maps_dict, feat, motif_interface, 0)
            feature_arrays[feat] = arr
        if window_idx < hazard_start:
            for feat in motif_features:
                baseline_features[feat].append(feature_arrays[feat])

        npz_path: Path | None = None
        if snapshot_every > 0 and window_idx % snapshot_every == 0:
            npz_payload: Dict[str, np.ndarray] = {}
            for feat in ["k_axis_bias", "k_entropy", "mismatch", "sigma", *motif_features]:
                key, arr = _select_map(maps_dict, feat, motif_interface, 0)
                npz_payload[key] = arr
            npz_path = npz_dir / f"seed{seed}_win{window_idx:04d}.npz"
            np.savez(npz_path, **npz_payload)

        window_records.append(
            {
                "seed": seed,
                "window": window_idx,
                "hazard_active": hazard_active,
                "ep_rate": ep_rate,
                "accept_window": accept_window,
                "mismatch_region": mismatch_region,
                "mismatch_outside": mismatch_outside,
                "features": feature_arrays,
                "npz_path": npz_path,
            }
        )
        accept_vals.append(accept_window)

        jsonl_handle.write(to_json_line(slim) + "\n")
        jsonl_handle.flush()

        with progress_path.open("a", encoding="utf-8") as ph:
            writer = csv.writer(ph)
            writer.writerow(
                [seed, window_idx, hazard_active, ep_rate, accept_window, mismatch_region]
            )

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

    initial_state = State.initialize(params, seed=seed)
    if hazard_start == 1:
        _backup_sigma(initial_state)
        _apply_hazard(initial_state)
        hazard_active_next = True

    run_sim(
        params,
        seed=seed,
        steps=burn_steps + window_steps * max_windows,
        report_every=window_steps,
        report_callback=report_cb,
        stop_callback=stop_cb,
        protocol_cycle=cycle,
        initial_state=initial_state,
    )

    jsonl_handle.close()

    if status == "RUNNING":
        status = "OK"
    if window_idx < max_windows and status == "OK":
        status = "FAIL_TIME"

    if not baseline_features or not baseline_features[motif_features[0]]:
        status = "FAIL_CONFIG"

    bins = build_bins(baseline_features, _feature_bins(motif_features, bins_axis_bias, bins_entropy))
    num_motifs = 1
    for count in bins.bins.values():
        num_motifs *= count

    hazard_maps: List[np.ndarray] = []
    motif_entropy_vals: List[float] = []
    topn_vals: List[float] = []
    histograms: List[np.ndarray] = []
    windows_seen: List[int] = []
    motif_ids_by_window: List[np.ndarray] = []
    motif_windows: List[int] = []

    for record in window_records:
        ids = motif_ids(record["features"], bins)
        p = motif_hist(ids, num_motifs)
        motif_entropy_vals.append(motif_entropy(p))
        topn_vals.append(top_n_coverage(p, top_n))
        histograms.append(p)
        windows_seen.append(int(record["window"]))
        motif_ids_by_window.append(ids)
        motif_windows.append(int(record["window"]))
        if record["hazard_active"]:
            hazard_maps.append(ids)
        record["motif_entropy"] = motif_entropy_vals[-1]
        record["topN_coverage"] = topn_vals[-1]
        record["motif_count"] = int(np.unique(ids).size)

        if record["npz_path"] is not None and record["npz_path"].exists():
            npz = np.load(record["npz_path"])
            payload = {k: npz[k] for k in npz.files}
            payload["motif_ids"] = ids.astype(np.int32)
            np.savez(record["npz_path"], **payload)

    baseline_counts, hazard_counts, post_counts = split_condition_counts(
        windows_seen,
        hazard_start,
        hazard_duration,
        histograms,
    )
    if baseline_counts.sum() > 0:
        baseline_counts /= baseline_counts.sum()
    if hazard_counts.sum() > 0:
        hazard_counts /= hazard_counts.sum()
    if post_counts.sum() > 0:
        post_counts /= post_counts.sum()

    coverage_baseline = top_n_coverage(baseline_counts, top_n)
    coverage_hazard = top_n_coverage(hazard_counts, top_n)
    jsd_pre_hazard = jsd(baseline_counts, hazard_counts)
    l1_pre_hazard = l1_distance(baseline_counts, hazard_counts)

    accept_mean = float(np.mean(accept_vals)) if accept_vals else 0.0

    top_motifs = np.argsort(hazard_counts)[::-1][:prop_top_m].tolist()
    prop_score, best_motif, best_shift = propagation_score(hazard_maps, top_motifs, shift_max)

    hazard_ids: List[np.ndarray] = []
    pre_ids: List[np.ndarray] = []
    for win, ids in zip(motif_windows, motif_ids_by_window):
        if win < hazard_start:
            pre_ids.append(ids)
        elif hazard_start <= win <= (hazard_start + hazard_duration - 1):
            hazard_ids.append(ids)

    C_pre = transition_counts_over_time(pre_ids, num_motifs)
    C_haz = transition_counts_over_time(hazard_ids, num_motifs)
    jsd_trans = transition_jsd(C_pre, C_haz)
    H_pre = conditional_entropy_from_counts(C_pre)
    H_haz = conditional_entropy_from_counts(C_haz)
    delta_H = H_haz - H_pre
    topk_pre = topk_transition_mass(C_pre, top_trans_k)
    topk_haz = topk_transition_mass(C_haz, top_trans_k)

    with raw_path.open("w", encoding="utf-8") as rh:
        writer = csv.DictWriter(
            rh,
            fieldnames=[
                "seed",
                "window",
                "hazard_active",
                "ep_rate",
                "accept_window",
                "mismatch_region",
                "mismatch_outside",
                "motif_entropy",
                "topN_coverage",
                "motif_count",
            ],
        )
        writer.writeheader()
        for record in window_records:
            writer.writerow({k: record.get(k, "") for k in writer.fieldnames})

    result = {
        "seed": seed,
        "status": status,
        "coverage_baseline": coverage_baseline,
        "coverage_hazard": coverage_hazard,
        "jsd_pre_hazard": jsd_pre_hazard,
        "l1_pre_hazard": l1_pre_hazard,
        "jsd_trans_pre_hazard": jsd_trans,
        "H_pre": H_pre,
        "H_hazard": H_haz,
        "delta_H": delta_H,
        "topK_cover_pre": topk_pre,
        "topK_cover_hazard": topk_haz,
        "prop_score_hazard": prop_score,
        "prop_best_motif": best_motif,
        "prop_best_shift": best_shift,
        "accept_mean": accept_mean,
        "motif_features": ",".join(motif_features),
        "bins_axis_bias": bins_axis_bias,
        "bins_entropy": bins_entropy,
        "hazard_start": hazard_start,
        "hazard_duration": hazard_duration,
    }

    pass_flag = (
        coverage_baseline >= coverage_min
        and coverage_hazard >= coverage_min
        and jsd_pre_hazard >= jsd_min
        and jsd_trans >= jsd_trans_min
        and prop_score >= prop_min
        and accept_mean >= accept_min
    )
    result["pass"] = pass_flag

    motifs_path = out_dir / f"motifs_seed{seed}.json"
    top_ids = np.argsort(hazard_counts)[::-1][: max(1, prop_top_m)]
    motifs_payload = {
        "seed": seed,
        "top_motifs": [
            {"id": int(idx), "prob": float(hazard_counts[idx])} for idx in top_ids
        ],
        "pre_hist": baseline_counts.tolist(),
        "hazard_hist": hazard_counts.tolist(),
        "post_hist": post_counts.tolist(),
        "jsd_pre_hazard": jsd_pre_hazard,
        "jsd_trans_pre_hazard": jsd_trans,
    }
    with motifs_path.open("w", encoding="utf-8") as mh:
        json.dump(motifs_payload, mh, indent=2)

    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        writer = csv.DictWriter(ah, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 12 motif proto-syntax")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--preset", type=str, default="scripts/params/meta_null_coupled_eta1.00_layers3.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--hazard-start-window", type=int, default=6)
    parser.add_argument("--hazard-duration-windows", type=int, default=8)
    parser.add_argument("--hazard-rect", type=str, default="8:16,8:16")
    parser.add_argument("--hazard-sigma", type=str, choices=["random", "flip"], default="random")
    parser.add_argument("--hazard-layers", type=str, default="0")
    parser.add_argument("--hazard-refresh-each-window", action="store_true", default=True)

    parser.add_argument("--motif-interface", type=int, default=0)
    parser.add_argument("--motif-features", type=str, default="k_axis_bias,k_entropy")
    parser.add_argument("--bins-axis-bias", type=int, default=7)
    parser.add_argument("--bins-entropy", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-trans-k", type=int, default=10)
    parser.add_argument("--shift-max", type=int, default=2)
    parser.add_argument("--prop-top-m", type=int, default=5)

    parser.add_argument("--coverage-min", type=float, default=0.60)
    parser.add_argument("--jsd-min", type=float, default=0.01)
    parser.add_argument("--jsd-trans-min", type=float, default=0.01)
    parser.add_argument("--prop-min", type=float, default=0.02)

    args = parser.parse_args()

    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)
    motif_features = _parse_keys(args.motif_features)
    _validate_features(motif_features)
    _validate_bins(args.bins_axis_bias, args.bins_entropy)

    preset = _load_preset(Path(args.preset))
    params = _as_params(
        preset,
        {
            "device": torch.device(args.device),
        },
    )
    if args.hazard_layers == "all":
        hazard_layers = list(range(params.layers))
    else:
        hazard_layers = _parse_layers(args.hazard_layers, params.layers)

    cycle = _cycle_list()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    rows: List[Dict[str, Any]] = []
    start_total = time.monotonic()
    for seed in seeds:
        result = run_seed(
            params,
            seed=seed,
            out_dir=out_dir,
            burn_sweeps=args.burn_in_sweeps,
            window_sweeps=args.window_sweeps,
            max_windows=args.max_windows,
            snapshot_every=args.snapshot_every_windows,
            hazard_start=args.hazard_start_window,
            hazard_duration=args.hazard_duration_windows,
            hazard_rect=args.hazard_rect,
            hazard_sigma=args.hazard_sigma,
            hazard_layers=hazard_layers,
            hazard_refresh_each_window=args.hazard_refresh_each_window,
            motif_interface=args.motif_interface,
            motif_features=motif_features,
            bins_axis_bias=args.bins_axis_bias,
            bins_entropy=args.bins_entropy,
            top_n=args.top_n,
            top_trans_k=args.top_trans_k,
            shift_max=args.shift_max,
            prop_top_m=args.prop_top_m,
            coverage_min=args.coverage_min,
            jsd_min=args.jsd_min,
            jsd_trans_min=args.jsd_trans_min,
            prop_min=args.prop_min,
            accept_min=args.accept_min,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
            cycle=cycle,
            resume=args.resume,
        )
        rows.append(result)
        if seed == seeds[0] and not result.get("pass"):
            print("PHASE12_GATE=FAIL")
            break

    agg_path = out_dir / "agg.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        writer = csv.DictWriter(ah, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = out_dir / "PHASE12_MOTIF_SYNTAX_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 12 motif proto-syntax v1\n\n")
        fh.write(
            "| seed | status | coverage_pre | jsd_pre_hazard | jsd_trans_pre_hazard | "
            "prop_score | pass |\n"
        )
        fh.write("| ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            fh.write(
                f"| {row['seed']} | {row['status']} | {row['coverage_baseline']:.6g} | "
                f"{row['jsd_pre_hazard']:.6g} | {row['jsd_trans_pre_hazard']:.6g} | "
                f"{row['prop_score_hazard']:.6g} | {row['pass']} |\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
