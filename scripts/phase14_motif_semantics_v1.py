#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.interventions import apply_sigma_flip, apply_sigma_randomize, parse_rect
from ratchet_gpu.motifs import (
    MotifBins,
    build_bins,
    jsd,
    motif_entropy,
    motif_hist,
    motif_ids,
    propagation_score,
    split_condition_counts,
    top_n_coverage,
)
from ratchet_gpu.params import Params
from ratchet_gpu.semantics import (
    motif_fraction_timeseries,
    predictive_semantic_pvalue,
    ring_hist,
    ring_masks_from_rect,
    semantic_effect,
)
from ratchet_gpu.sim import run_sim, _cycle_list
from ratchet_gpu.state import State
from ratchet_gpu.spatial import compute_spatial_maps, finite_check

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


def _parse_keys(value: str) -> List[str]:
    return [k.strip() for k in value.split(",") if k.strip()]


def _validate_hazard_schedule(start: int, duration: int, max_windows: int) -> None:
    if start < 1:
        raise ValueError("hazard_start_window must be >= 1")
    if duration < 1:
        raise ValueError("hazard_duration_windows must be >= 1")
    if start + duration - 1 > max_windows:
        raise ValueError("hazard window must fit within max_windows")


def _validate_bins(bins_axis_bias: int, bins_entropy: int) -> None:
    if bins_axis_bias < 2:
        raise ValueError("bins_axis_bias must be >= 2")
    if bins_entropy < 2:
        raise ValueError("bins_entropy must be >= 2")


def _validate_features(features: List[str]) -> None:
    allowed = {"k_axis_bias", "k_entropy", "k_r2"}
    unknown = [f for f in features if f not in allowed]
    if unknown:
        raise ValueError(f"unsupported motif features: {', '.join(unknown)}")


def _slim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    slim = dict(snapshot)
    for key in list(slim.keys()):
        if key in HEAVY_KEYS or key.endswith("_items_window"):
            slim.pop(key, None)
    return slim


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
        if feat == "k_axis_bias":
            bins[feat] = bins_axis_bias
        elif feat == "k_entropy":
            bins[feat] = bins_entropy
        else:
            bins[feat] = bins_entropy
    return bins


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


def _region_mean(map_tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    data = map_tensor.to(dtype=torch.float32)
    if data.ndim == 3:
        data = data.mean(dim=0)
    region = data[mask]
    outside = data[~mask]
    region_mean = float(region.mean().item()) if region.numel() else 0.0
    outside_mean = float(outside.mean().item()) if outside.numel() else 0.0
    return region_mean, outside_mean


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
    shift_max: int,
    prop_top_m: int,
    coverage_min: float,
    jsd_min: float,
    prop_min: float,
    spike_min: float,
    semantic_best_max: float,
    z_semantic_max: float,
    semantic_p_max: float,
    semantic_support_min: float,
    semantic_top_k: int,
    shuffle_n: int,
    semantic_shuffle_mode: str,
    semantic_pred_enable: bool,
    semantic_pred_lag: int,
    semantic_pred_metric: str,
    semantic_pred_p_max: float,
    semantic_pred_corr_max: float,
    semantic_pred_shift_n: int,
    semantic_pred_top_k: int,
    accept_min: float,
    ring_width: int,
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
    cycle: List[str],
    resume: bool,
) -> Dict[str, Any]:
    seed_dir = out_dir
    raw_path = seed_dir / "raw.csv"
    agg_path = seed_dir / "agg.csv"
    progress_path = seed_dir / "progress.csv"
    jsonl_dir = seed_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = seed_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    if resume and agg_path.exists():
        with agg_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            if str(row.get("seed")) == str(seed) and row.get("status") not in {"RUNNING", ""}:
                return row

    shape = params.shape
    H, W = int(shape[-2]), int(shape[-1])
    region_mask_t, flat_idx = parse_rect(hazard_rect, (H, W))
    region_mask_t = region_mask_t.to(device=params.device)
    region_mask_np, ring_mask_np, outside_mask_np = ring_masks_from_rect(
        hazard_rect, (H, W), width=ring_width
    )

    N = int(np.prod(params.shape))
    expected = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected))
    window_steps = int(math.ceil(window_sweeps * N / expected))

    seed_dir.mkdir(parents=True, exist_ok=True)
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
    mismatch_region_vals: List[float] = []
    motif_maps: List[np.ndarray] = []
    mismatch_maps: List[np.ndarray] = []
    windows_seen: List[int] = []

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

        map_keys = set(["k_axis_bias", "k_entropy", "mismatch", "sigma"])
        if "k_r2" in motif_features:
            map_keys.add("k_r2")
        maps_dict = compute_spatial_maps(state, list(map_keys))
        ok, _ = finite_check(maps_dict)
        if not ok:
            status = "FAIL_NAN_MAP"
            return

        mismatch_region, mismatch_outside = _region_mean(maps_dict["mismatch"], region_mask_t)

        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        accept_window = float(snapshot.get("acceptedFracWindow", accepted_frac))

        feature_arrays: Dict[str, np.ndarray] = {}
        for feat in motif_features:
            _name, arr = _select_map(maps_dict, feat, motif_interface, 0)
            feature_arrays[feat] = arr
        if window_idx < hazard_start:
            for feat in motif_features:
                baseline_features[feat].append(feature_arrays[feat])

        if snapshot_every > 0 and window_idx % snapshot_every == 0:
            npz_payload: Dict[str, np.ndarray] = {}
            for feat in ["k_axis_bias", "k_entropy", "mismatch", "sigma", *motif_features]:
                key, arr = _select_map(maps_dict, feat, motif_interface, 0)
                npz_payload[key] = arr
            npz_path = npz_dir / f"seed{seed}_win{window_idx:04d}.npz"
            np.savez(npz_path, **npz_payload)

        record = {
            "seed": seed,
            "window": window_idx,
            "hazard_active": hazard_active,
            "ep_rate": ep_rate,
            "accept_window": accept_window,
            "mismatch_region": mismatch_region,
            "mismatch_outside": mismatch_outside,
            "features": feature_arrays,
        }
        window_records.append(record)
        accept_vals.append(accept_window)
        mismatch_region_vals.append(mismatch_region)
        mismatch_maps.append(_select_map(maps_dict, "mismatch", motif_interface, 0)[1])
        windows_seen.append(window_idx)

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

    motif_entropy_vals: List[float] = []
    topn_vals: List[float] = []
    histograms_full: List[np.ndarray] = []
    histograms_ring: List[np.ndarray] = []
    hazard_maps: List[np.ndarray] = []

    for record in window_records:
        ids = motif_ids(record["features"], bins)
        p = motif_hist(ids, num_motifs)
        motif_entropy_vals.append(motif_entropy(p))
        topn_vals.append(top_n_coverage(p, top_n))
        histograms_full.append(p)
        histograms_ring.append(ring_hist(ids, ring_mask_np, num_motifs))
        if record["hazard_active"]:
            hazard_maps.append(ids)
        record["motif_entropy"] = motif_entropy_vals[-1]
        record["topN_coverage"] = topn_vals[-1]
        record["motif_ids"] = ids
        motif_maps.append(ids)

    pre_full, haz_full, post_full = split_condition_counts(
        windows_seen, hazard_start, hazard_duration, histograms_full
    )
    pre_ring, haz_ring, _post_ring = split_condition_counts(
        windows_seen, hazard_start, hazard_duration, histograms_ring
    )
    if pre_full.sum() > 0:
        pre_full /= pre_full.sum()
    if haz_full.sum() > 0:
        haz_full /= haz_full.sum()
    if pre_ring.sum() > 0:
        pre_ring /= pre_ring.sum()
    if haz_ring.sum() > 0:
        haz_ring /= haz_ring.sum()

    coverage_pre = top_n_coverage(pre_full, top_n)
    coverage_hazard = top_n_coverage(haz_full, top_n)
    jsd_ring_pre_hazard = jsd(pre_ring, haz_ring)
    accept_mean = float(np.mean(accept_vals)) if accept_vals else 0.0

    top_motifs = np.argsort(haz_full)[::-1][:prop_top_m].tolist()
    prop_score, best_motif, best_shift = propagation_score(hazard_maps, top_motifs, shift_max)

    semantics = semantic_effect(
        motif_maps,
        mismatch_maps,
        windows_seen,
        ring_mask_np,
        num_motifs,
        hazard_start,
        hazard_duration,
        semantic_support_min,
        shuffle_n,
        shuffle_mode=semantic_shuffle_mode,
        candidate_top_k=semantic_top_k,
        rng=np.random.default_rng(seed),
    )

    semantic_pred_best_corr = 0.0
    semantic_pred_p = 1.0
    semantic_pred_best_motif = -1
    semantic_pred_shuf_mean = 0.0
    semantic_pred_shuf_std = 0.0
    semantic_pred_candidate_ids: List[int] = []
    if semantic_pred_enable:
        hazard_end = hazard_start + hazard_duration - 1
        idxs = [
            idx
            for idx, win in enumerate(windows_seen)
            if win <= hazard_end - semantic_pred_lag
        ]
        if idxs:
            ring_support = np.zeros(num_motifs, dtype=np.float64)
            for idx in idxs:
                ring_support += histograms_ring[idx]
            order = np.argsort(ring_support)[::-1]
            semantic_pred_candidate_ids = [
                int(i)
                for i in order[:semantic_pred_top_k]
                if ring_support[i] > 0
            ]
            x_tm = motif_fraction_timeseries(
                motif_maps,
                ring_mask_np,
                semantic_pred_candidate_ids,
                idxs,
            )
            y_t = np.array(
                [
                    mismatch_region_vals[i + semantic_pred_lag] - mismatch_region_vals[i]
                    for i in idxs
                ],
                dtype=np.float64,
            )
            best_corr, pval, mean_null, std_null, best_idx = predictive_semantic_pvalue(
                x_tm,
                y_t,
                semantic_pred_shift_n,
                np.random.default_rng(seed + 17),
                metric=semantic_pred_metric,
            )
            semantic_pred_best_corr = float(best_corr)
            semantic_pred_p = float(pval)
            semantic_pred_shuf_mean = float(mean_null)
            semantic_pred_shuf_std = float(std_null)
            if semantic_pred_candidate_ids and best_idx >= 0:
                semantic_pred_best_motif = int(semantic_pred_candidate_ids[best_idx])

    pre_vals = [mismatch_region_vals[i - 1] for i in windows_seen if i < hazard_start]
    haz_vals = [mismatch_region_vals[i - 1] for i in windows_seen if hazard_start <= i <= hazard_start + hazard_duration - 1]
    pre_mean = float(np.mean(pre_vals)) if pre_vals else 0.0
    peak = float(np.max(haz_vals)) if haz_vals else pre_mean
    spike = peak - pre_mean

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
            ],
        )
        writer.writeheader()
        for record in window_records:
            writer.writerow({k: record.get(k, "") for k in writer.fieldnames})

    result = {
        "seed": seed,
        "status": status,
        "coverage_pre": coverage_pre,
        "coverage_hazard": coverage_hazard,
        "jsd_ring_pre_hazard": jsd_ring_pre_hazard,
        "prop_score_hazard": prop_score,
        "prop_best_motif": best_motif,
        "prop_best_shift": best_shift,
        "accept_mean": accept_mean,
        "spike": spike,
        "semantic_best": semantics["semantic_best"],
        "semantic_best_motif": semantics["semantic_best_motif"],
        "semantic_support": semantics["semantic_support"],
        "semantic_z": semantics["semantic_z"],
        "semantic_p": semantics["semantic_p"],
        "semantic_shuf_mean": semantics["semantic_shuf_mean"],
        "semantic_shuf_std": semantics["semantic_shuf_std"],
        "semantic_candidate_ids": semantics.get("semantic_candidate_ids", []),
        "semantic_candidate_supports": semantics.get("semantic_candidate_supports", []),
        "semantic_pred_best_corr": semantic_pred_best_corr,
        "semantic_pred_p": semantic_pred_p,
        "semantic_pred_best_motif": semantic_pred_best_motif,
        "semantic_pred_shuf_mean": semantic_pred_shuf_mean,
        "semantic_pred_shuf_std": semantic_pred_shuf_std,
        "semantic_pred_candidate_ids": semantic_pred_candidate_ids,
    }

    context_pass = (
        coverage_pre >= coverage_min
        and coverage_hazard >= coverage_min
        and jsd_ring_pre_hazard >= jsd_min
        and prop_score >= prop_min
        and spike >= spike_min
        and accept_mean >= accept_min
    )
    if semantic_pred_enable:
        semantic_pred_pass = (
            semantic_pred_best_corr <= semantic_pred_corr_max
            and semantic_pred_p <= semantic_pred_p_max
        )
        pass_flag = context_pass and semantic_pred_pass
        result["status_context"] = "PASS" if context_pass else "FAIL"
        result["status_pred"] = "PASS" if semantic_pred_pass else "FAIL"
    else:
        semantic_best_pass = (
            float(semantics["semantic_best"]) <= semantic_best_max
            and float(semantics["semantic_p"]) <= semantic_p_max
        )
        pass_flag = context_pass and semantic_best_pass
        result["status_context"] = "PASS" if context_pass else "FAIL"
        result["status_pred"] = "N/A"
    result["pass"] = pass_flag

    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        writer = csv.DictWriter(ah, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    report_path = seed_dir / "PHASE14_MOTIF_SEMANTICS_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 14 motif semantics v1\n\n")
        fh.write(f"seed={seed}\n\n")
        fh.write("| seed | status | coverage_pre | coverage_hazard | jsd_ring_pre_hazard | prop_score | spike | semantic_best | semantic_p | semantic_z | semantic_pred_best_corr | semantic_pred_p | support |\n")
        fh.write("| ---: | :----- | -----------: | ---------------: | ------------------: | ----------: | -----: | -------------: | ---------: | ---------: | ----------------------: | -------------: | ------: |\n")
        fh.write(
            f"| {seed} | {status} | {coverage_pre:.4g} | {coverage_hazard:.4g} | {jsd_ring_pre_hazard:.4g} | {prop_score:.4g} | {spike:.4g} | {semantics['semantic_best']:.4g} | {semantics['semantic_p']:.4g} | {semantics['semantic_z']:.4g} | {semantic_pred_best_corr:.4g} | {semantic_pred_p:.4g} | {semantics['semantic_support']:.4g} |\n"
        )
        fh.write("\n")
        fh.write(f"motif_features={','.join(motif_features)}\n")
        fh.write(f"bins_axis_bias={bins_axis_bias} bins_entropy={bins_entropy}\n")
        fh.write(
            f"semantic_support_min={semantic_support_min} semantic_top_k={semantic_top_k} shuffle_mode={semantic_shuffle_mode}\n"
        )
        fh.write(f"jsd_ring_pre_hazard={jsd_ring_pre_hazard:.6g}\n")
        fh.write(f"semantic_best={semantics['semantic_best']:.6g}\n")
        fh.write(f"semantic_p={semantics['semantic_p']:.6g} semantic_p_max={semantic_p_max:.6g}\n")
        fh.write(f"semantic_z={semantics['semantic_z']:.6g}\n")
        if semantic_pred_enable:
            fh.write(
                "semantic_pred: "
                f"best_corr={semantic_pred_best_corr:.6g} "
                f"p={semantic_pred_p:.6g} "
                f"p_max={semantic_pred_p_max:.6g} "
                f"corr_max={semantic_pred_corr_max:.6g}\n"
            )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 14 motif semantics")
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
    parser.add_argument("--no-fail-fast", action="store_true")

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
    parser.add_argument("--shift-max", type=int, default=2)
    parser.add_argument("--prop-top-m", type=int, default=5)
    parser.add_argument("--ring-width", type=int, default=1)
    parser.add_argument("--coverage-min", type=float, default=0.60)
    parser.add_argument("--jsd-ring-min", type=float, default=0.01)
    parser.add_argument("--prop-min", type=float, default=0.02)
    parser.add_argument("--spike-min", type=float, default=0.01)
    parser.add_argument("--semantic-best-max", type=float, default=-0.002)
    parser.add_argument("--z-semantic-max", type=float, default=-2.0)
    parser.add_argument("--semantic-p-max", type=float, default=0.05)
    parser.add_argument("--semantic-support-min", type=float, default=0.05)
    parser.add_argument("--semantic-top-k", type=int, default=8)
    parser.add_argument(
        "--semantic-shuffle-mode",
        type=str,
        default="shift",
        choices=["permute", "shift", "block_shift"],
    )
    parser.add_argument("--shuffle-n", type=int, default=200)
    parser.add_argument("--semantic-pred-enable", action="store_true")
    parser.add_argument("--semantic-pred-lag", type=int, default=1)
    parser.add_argument("--semantic-pred-metric", type=str, default="corr")
    parser.add_argument("--semantic-pred-p-max", type=float, default=0.05)
    parser.add_argument("--semantic-pred-corr-max", type=float, default=-0.10)
    parser.add_argument("--semantic-pred-shift-n", type=int, default=200)
    parser.add_argument("--semantic-pred-top-k", type=int, default=8)

    args = parser.parse_args()

    motif_features = _parse_keys(args.motif_features)
    _validate_features(motif_features)
    _validate_bins(args.bins_axis_bias, args.bins_entropy)
    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preset = _load_preset(Path(args.preset))
    params = _as_params(preset, {"device": args.device})

    cycle = list(_cycle_list())
    seeds = _parse_seeds(args.seeds)
    hazard_layers = _parse_layers(args.hazard_layers, params.layers)

    results: List[Dict[str, Any]] = []
    start_total = time.monotonic()
    fail_fast = not args.no_fail_fast
    for idx, seed in enumerate(seeds):
        result = run_seed(
            params=params,
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
            shift_max=args.shift_max,
            prop_top_m=args.prop_top_m,
            coverage_min=args.coverage_min,
            jsd_min=args.jsd_ring_min,
            prop_min=args.prop_min,
            spike_min=args.spike_min,
            semantic_best_max=args.semantic_best_max,
            z_semantic_max=args.z_semantic_max,
            semantic_p_max=args.semantic_p_max,
            semantic_support_min=args.semantic_support_min,
            semantic_top_k=args.semantic_top_k,
            shuffle_n=args.shuffle_n,
            semantic_shuffle_mode=args.semantic_shuffle_mode,
            semantic_pred_enable=args.semantic_pred_enable,
            semantic_pred_lag=args.semantic_pred_lag,
            semantic_pred_metric=args.semantic_pred_metric,
            semantic_pred_p_max=args.semantic_pred_p_max,
            semantic_pred_corr_max=args.semantic_pred_corr_max,
            semantic_pred_shift_n=args.semantic_pred_shift_n,
            semantic_pred_top_k=args.semantic_pred_top_k,
            accept_min=args.accept_min,
            ring_width=args.ring_width,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
            cycle=cycle,
            resume=args.resume,
        )
        results.append(result)
        if args.progress:
            print(f"COMPLETED seed={seed} status={result.get('status')} pass={result.get('pass')}")
        if fail_fast and seed == seeds[0] and not result.get("pass", False):
            break

    agg_path = out_dir / "agg.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        writer = csv.DictWriter(ah, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    report_path = out_dir / "PHASE14_MOTIF_SEMANTICS_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 14 motif semantics v1\n\n")
        fh.write(
            "| seed | status | status_context | status_pred | coverage_pre | coverage_hazard | jsd_ring_pre_hazard | prop_score | spike | semantic_best | semantic_p | semantic_z | semantic_pred_best_corr | semantic_pred_p | pass |\n"
        )
        fh.write(
            "| ---: | :----- | :------------- | :---------- | -----------: | ---------------: | ------------------: | ----------: | -----: | -------------: | ---------: | ---------: | ----------------------: | -------------: | :--- |\n"
        )
        for record in results:
            fh.write(
                f"| {record['seed']} | {record['status']} | {record.get('status_context','')} | {record.get('status_pred','')} | {float(record['coverage_pre']):.4g} | {float(record['coverage_hazard']):.4g} | {float(record['jsd_ring_pre_hazard']):.4g} | {float(record['prop_score_hazard']):.4g} | {float(record['spike']):.4g} | {float(record['semantic_best']):.4g} | {float(record['semantic_p']):.4g} | {float(record['semantic_z']):.4g} | {float(record.get('semantic_pred_best_corr', 0.0)):.4g} | {float(record.get('semantic_pred_p', 0.0)):.4g} | {record['pass']} |\n"
            )
        fh.write("\n")
        fh.write(f"motif_features={','.join(motif_features)}\n")
        fh.write(f"bins_axis_bias={args.bins_axis_bias} bins_entropy={args.bins_entropy}\n")
        fh.write(
            f"semantic_support_min={args.semantic_support_min} semantic_top_k={args.semantic_top_k} shuffle_mode={args.semantic_shuffle_mode}\n"
        )
        fh.write(
            f"semantic_best_max={args.semantic_best_max} semantic_p_max={args.semantic_p_max} z_semantic_max={args.z_semantic_max}\n"
        )
        if args.semantic_pred_enable:
            fh.write(
                f"semantic_pred: lag={args.semantic_pred_lag} metric={args.semantic_pred_metric} p_max={args.semantic_pred_p_max} corr_max={args.semantic_pred_corr_max} shift_n={args.semantic_pred_shift_n} top_k={args.semantic_pred_top_k}\n"
            )
        fh.write(f"coverage_min={args.coverage_min} jsd_ring_min={args.jsd_ring_min}\n")
        fh.write(f"prop_min={args.prop_min} spike_min={args.spike_min}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
