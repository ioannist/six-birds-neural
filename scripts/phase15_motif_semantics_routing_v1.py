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
from ratchet_gpu.motifs import build_bins, jsd, motif_hist, motif_ids, top_n_coverage
from ratchet_gpu.params import Params
from ratchet_gpu.semantics import (
    alignment_delta,
    alignment_shift_null,
    alignment_score,
    hazard_center,
    pref_axis_map,
    ring_hist,
    ring_masks_from_rect,
)
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


def _parse_keys(value: str) -> List[str]:
    return [k.strip() for k in value.split(",") if k.strip()]


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


def _fail_row(seed: int, status: str) -> Dict[str, Any]:
    return {
        "seed": seed,
        "status": status,
        "spike": float("nan"),
        "jsd_ring_pre_hazard": float("nan"),
        "alignment_delta": float("nan"),
        "alignment_p": float("nan"),
        "alignment_null_mean": float("nan"),
        "alignment_null_std": float("nan"),
        "coverage_pre": float("nan"),
        "coverage_hazard": float("nan"),
        "pass": False,
    }


def _fmt(val: Any) -> str:
    try:
        return f"{float(val):.4g}"
    except (TypeError, ValueError):
        return "nan"


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


def _run_case(
    case: str,
    params: Params,
    seed: int,
    out_dir: Path,
    burn_sweeps: float,
    window_sweeps: float,
    max_windows: int,
    snapshot_every: int,
    hazard_on: bool,
    hazard_start: int,
    hazard_duration: int,
    hazard_rect: str,
    hazard_sigma: str,
    hazard_layers: List[int],
    hazard_refresh_each_window: bool,
    ring_mask_np: np.ndarray,
    pref_axis_np: np.ndarray,
    interface_idx: int,
    layer_idx: int,
    cycle: List[str],
    accept_min: float,
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
    resume: bool,
) -> Dict[str, Any]:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = case_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    raw_path = case_dir / "raw.csv"
    agg_path = case_dir / "agg.csv"
    progress_path = case_dir / "progress.csv"

    if resume and agg_path.exists():
        with agg_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            if str(row.get("seed")) == str(seed) and row.get("status") not in {"RUNNING", ""}:
                return row

    raw_fields = [
        "case",
        "seed",
        "window_index",
        "step",
        "hazard_active",
        "accept_window",
        "alignment_score",
        "mismatch_region",
        "mismatch_outside",
        "k_axis_bias_region",
        "k_axis_bias_outside",
        "k_entropy_region",
        "k_entropy_outside",
    ]

    if not raw_path.exists():
        with raw_path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=raw_fields).writeheader()

    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=["case", "seed", "window_index", "step", "hazard_active", "accept_window"]).writeheader()

    rng = torch.Generator(device=params.resolved_device())
    rng.manual_seed(seed + 11)

    hazard_mask_t, hazard_idx = parse_rect(hazard_rect, params.shape)
    hazard_mask_t = hazard_mask_t.to(device=params.resolved_device())

    window_idx = 0
    status = "RUNNING"
    diag_state = None
    hazard_active_next = False
    sigma_backup = None

    axis_bias_by_window: List[np.ndarray] = []
    entropy_by_window: List[np.ndarray] = []
    mismatch_by_window: List[np.ndarray] = []
    accept_by_window: List[float] = []
    alignment_by_window: List[float] = []
    windows_seen: List[int] = []

    run_start = time.monotonic()

    N = int(np.prod(params.shape))
    expected = _expected_proposals_per_step(N, params.resolved_device(), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / max(expected, 1.0)))
    window_steps = int(math.ceil(window_sweeps * N / max(expected, 1.0)))

    def _backup_sigma(state):
        nonlocal sigma_backup
        sigma_backup = state.sigma[hazard_layers][:, hazard_idx].clone()

    def _restore_sigma(state):
        nonlocal sigma_backup
        if sigma_backup is None:
            return
        state.sigma[hazard_layers][:, hazard_idx] = sigma_backup
        sigma_backup = None

    def _apply_hazard(state):
        if hazard_sigma == "flip":
            apply_sigma_flip(state, hazard_idx, layers=hazard_layers)
        elif hazard_sigma == "random":
            apply_sigma_randomize(state, hazard_idx, layers=hazard_layers, rng=rng)
        else:
            raise ValueError(f"Unknown hazard sigma mode: {hazard_sigma}")

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal status, hazard_active_next, diag_state, window_idx, sigma_backup
        if status != "RUNNING" or window_idx >= max_windows:
            return

        snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
        is_burn = step <= burn_steps
        if not is_burn:
            window_idx += 1

        hazard_active = hazard_active_next if not is_burn else False

        window_props = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
        accept_window = float(ep_ledger.get("window_accepted", 0)) / window_props if window_props else 0.0

        maps_dict = compute_spatial_maps(state, ["k_axis_bias", "k_entropy", "mismatch"])
        ok, bad = finite_check(maps_dict)
        if not ok:
            status = f"FAIL_NAN_MAP:{','.join(bad)}"

        name_bias, bias_map = _select_map(maps_dict, "k_axis_bias", interface_idx, layer_idx)
        name_entropy, entropy_map = _select_map(maps_dict, "k_entropy", interface_idx, layer_idx)
        name_mismatch, mismatch_map = _select_map(maps_dict, "mismatch", interface_idx, layer_idx)

        region_mask_np = hazard_mask_t.cpu().numpy()
        outside_mask_np = ~region_mask_np
        mismatch_region = float(np.mean(mismatch_map[region_mask_np])) if region_mask_np.any() else 0.0
        mismatch_outside = float(np.mean(mismatch_map[outside_mask_np])) if outside_mask_np.any() else 0.0
        bias_region = float(np.mean(bias_map[region_mask_np])) if region_mask_np.any() else 0.0
        bias_outside = float(np.mean(bias_map[outside_mask_np])) if outside_mask_np.any() else 0.0
        entropy_region = float(np.mean(entropy_map[region_mask_np])) if region_mask_np.any() else 0.0
        entropy_outside = float(np.mean(entropy_map[outside_mask_np])) if outside_mask_np.any() else 0.0
        align_score = alignment_score(bias_map, pref_axis_np, ring_mask_np)

        if not is_burn:
            axis_bias_by_window.append(bias_map)
            entropy_by_window.append(entropy_map)
            mismatch_by_window.append(mismatch_map)
            accept_by_window.append(accept_window)
            alignment_by_window.append(align_score)
            windows_seen.append(window_idx)

            if snapshot_every > 0 and window_idx % snapshot_every == 0:
                np.savez(
                    npz_dir / f"seed{seed}_win{window_idx:04d}.npz",
                    **{
                        "step": step,
                        "window_index": window_idx,
                        "case": case,
                        "seed": seed,
                        name_bias: bias_map,
                        name_entropy: entropy_map,
                        name_mismatch: mismatch_map,
                    },
                )

            snap = _slim_snapshot(snapshot)
            snap.update(
                {
                    "case": case,
                    "seed": seed,
                    "window_index": window_idx,
                    "hazard_active": hazard_active,
                    "alignment_score": align_score,
                    "mismatch_region": mismatch_region,
                    "mismatch_outside": mismatch_outside,
                    "k_axis_bias_region": bias_region,
                    "k_axis_bias_outside": bias_outside,
                    "k_entropy_region": entropy_region,
                    "k_entropy_outside": entropy_outside,
                }
            )
            with (jsonl_dir / f"seed{seed}.jsonl").open("a", encoding="utf-8") as jh:
                jh.write(to_json_line(snap) + "\n")

            with raw_path.open("a", encoding="utf-8", newline="") as rh:
                writer = csv.DictWriter(rh, fieldnames=raw_fields)
                writer.writerow(
                    {
                        "case": case,
                        "seed": seed,
                        "window_index": window_idx,
                        "step": step,
                        "hazard_active": hazard_active,
                        "accept_window": accept_window,
                        "alignment_score": align_score,
                        "mismatch_region": mismatch_region,
                        "mismatch_outside": mismatch_outside,
                        "k_axis_bias_region": bias_region,
                        "k_axis_bias_outside": bias_outside,
                        "k_entropy_region": entropy_region,
                        "k_entropy_outside": entropy_outside,
                    }
                )

            with progress_path.open("a", encoding="utf-8", newline="") as ph:
                csv.DictWriter(ph, fieldnames=["case", "seed", "window_index", "step", "hazard_active", "accept_window"]).writerow(
                    {
                        "case": case,
                        "seed": seed,
                        "window_index": window_idx,
                        "step": step,
                        "hazard_active": hazard_active,
                        "accept_window": accept_window,
                    }
                )

        if accept_window < accept_min and not is_burn:
            status = "FAIL_ACCEPT"

        if time.monotonic() - run_start > max_seconds_per_run:
            status = "FAIL_TIME"
        if time.monotonic() - start_total > max_seconds_total:
            status = "FAIL_TIME"

        if not is_burn and hazard_on:
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

    if window_idx >= max_windows and status in {"RUNNING", "FAIL_TIME"}:
        status = "OK"
    elif status == "RUNNING":
        status = "FAIL_TIME"

    summary = {
        "case": case,
        "seed": seed,
        "status": status,
        "windows_completed": window_idx,
        "window_steps": window_steps,
        "accept_mean": float(np.mean(accept_by_window[-5:])) if accept_by_window else 0.0,
        "axis_bias_by_window": axis_bias_by_window,
        "entropy_by_window": entropy_by_window,
        "mismatch_by_window": mismatch_by_window,
        "alignment_by_window": alignment_by_window,
        "windows_seen": windows_seen,
    }
    if not agg_path.exists():
        with agg_path.open("w", encoding="utf-8", newline="") as ah:
            csv.DictWriter(ah, fieldnames=list(summary.keys())).writeheader()
    with agg_path.open("a", encoding="utf-8", newline="") as ah:
        csv.DictWriter(ah, fieldnames=list(summary.keys())).writerow({k: v if not isinstance(v, list) else "" for k, v in summary.items()})
    return summary


def _compute_spike(
    mismatch_region: List[float],
    hazard_start: int,
    hazard_duration: int,
) -> float:
    pre_idx = list(range(1, hazard_start))
    haz_idx = list(range(hazard_start, hazard_start + hazard_duration))
    pre_vals = [mismatch_region[i - 1] for i in pre_idx] if pre_idx else []
    haz_vals = [mismatch_region[i - 1] for i in haz_idx] if haz_idx else []
    pre = float(np.mean(pre_vals)) if pre_vals else 0.0
    peak = float(np.max(haz_vals)) if haz_vals else pre
    return peak - pre


def _motif_role_table(
    motif_ids_by_window: List[np.ndarray],
    axis_bias_by_window: List[np.ndarray],
    ring_mask: np.ndarray,
    pref_axis: np.ndarray,
    hazard_idx: List[int],
    top_k: int,
) -> List[Dict[str, float]]:
    if not hazard_idx:
        return []
    counts = None
    for idx in hazard_idx:
        ids = motif_ids_by_window[idx][ring_mask].ravel()
        if ids.size == 0:
            continue
        max_id = int(ids.max()) + 1
        if counts is None or counts.size < max_id:
            new_counts = np.zeros(max_id, dtype=np.float64)
            if counts is not None:
                new_counts[: counts.size] = counts
            counts = new_counts
        counts[: max_id] += np.bincount(ids, minlength=max_id)
    if counts is None or counts.sum() == 0:
        return []
    support = counts / counts.sum()
    order = np.argsort(support)[::-1][: max(1, top_k)]
    table = []
    for mid in order:
        if support[mid] <= 0:
            continue
        contribs = []
        for idx in hazard_idx:
            ids = motif_ids_by_window[idx]
            mask = (ids == mid) & ring_mask
            if not mask.any():
                continue
            contribs.append(float(np.mean(axis_bias_by_window[idx][mask] * pref_axis[mask])))
        mean_align = float(np.mean(contribs)) if contribs else 0.0
        table.append({"motif_id": int(mid), "support": float(support[mid]), "mean_align": mean_align})
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase15 motif semantics routing intent")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--preset",
        default="scripts/params/meta_null_coupled_eta1.00_layers3.json",
    )
    parser.add_argument("--out-dir", default=".tmp/phase15_motif_routing_semantics_v1")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--hazard-start-window", type=int, default=6)
    parser.add_argument("--hazard-duration-windows", type=int, default=8)
    parser.add_argument("--hazard-rect", required=True)
    parser.add_argument("--hazard-sigma", choices=["random", "flip"], default="random")
    parser.add_argument("--hazard-layers", default="0")
    parser.add_argument("--hazard-refresh-each-window", action="store_true")
    parser.add_argument("--motif-interface", type=int, default=0)
    parser.add_argument("--motif-features", default="k_axis_bias,k_entropy")
    parser.add_argument("--bins-axis-bias", type=int, default=7)
    parser.add_argument("--bins-entropy", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--ring-width", type=int, default=1)
    parser.add_argument("--shuffle-n", type=int, default=200)
    parser.add_argument("--spike-min", type=float, default=0.01)
    parser.add_argument("--align-delta-min", type=float, default=0.01)
    parser.add_argument("--align-p-max", type=float, default=0.10)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)
    motif_features = _parse_keys(args.motif_features)
    if not motif_features:
        raise ValueError("motif-features must not be empty")

    preset = _load_preset(Path(args.preset))
    base_params = _as_params(preset, {"device": args.device})
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    region_mask_np, ring_mask_np, outside_mask_np = ring_masks_from_rect(
        args.hazard_rect, base_params.shape, width=args.ring_width
    )
    center = hazard_center(args.hazard_rect, base_params.shape)
    pref_axis_np = pref_axis_map(base_params.shape, center)

    start_total = time.monotonic()
    cycle = _cycle_list()
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        if time.monotonic() - start_total > args.max_seconds_total:
            break
        baseline = _run_case(
            "baseline",
            base_params,
            seed,
            out_dir,
            args.burn_in_sweeps,
            args.window_sweeps,
            args.max_windows,
            args.snapshot_every_windows,
            False,
            args.hazard_start_window,
            args.hazard_duration_windows,
            args.hazard_rect,
            args.hazard_sigma,
            [int(x) for x in args.hazard_layers.split(",") if x.strip()],
            args.hazard_refresh_each_window,
            ring_mask_np,
            pref_axis_np,
            args.motif_interface,
            0,
            cycle,
            args.accept_min,
            args.max_seconds_total,
            args.max_seconds_per_run,
            start_total,
            args.resume,
        )
        if baseline.get("status") != "OK":
            rows.append(_fail_row(seed, str(baseline.get("status"))))
            continue

        pre_idx = [i for i, w in enumerate(baseline["windows_seen"]) if w < args.hazard_start_window]
        baseline_features: Dict[str, List[np.ndarray]] = {feat: [] for feat in motif_features}
        for idx in pre_idx:
            baseline_features["k_axis_bias"].append(baseline["axis_bias_by_window"][idx])
            baseline_features["k_entropy"].append(baseline["entropy_by_window"][idx])
        bins = build_bins(baseline_features, _feature_bins(motif_features, args.bins_axis_bias, args.bins_entropy))
        num_motifs = 1
        for b in bins.bins.values():
            num_motifs *= b

        hazard = _run_case(
            "hazard",
            base_params,
            seed,
            out_dir,
            args.burn_in_sweeps,
            args.window_sweeps,
            args.max_windows,
            args.snapshot_every_windows,
            True,
            args.hazard_start_window,
            args.hazard_duration_windows,
            args.hazard_rect,
            args.hazard_sigma,
            [int(x) for x in args.hazard_layers.split(",") if x.strip()],
            args.hazard_refresh_each_window,
            ring_mask_np,
            pref_axis_np,
            args.motif_interface,
            0,
            cycle,
            args.accept_min,
            args.max_seconds_total,
            args.max_seconds_per_run,
            start_total,
            args.resume,
        )

        if hazard.get("status") != "OK":
            rows.append(_fail_row(seed, str(hazard.get("status"))))
            continue

        motif_ids_by_window: List[np.ndarray] = []
        for bias, entropy in zip(hazard["axis_bias_by_window"], hazard["entropy_by_window"]):
            features = {"k_axis_bias": bias, "k_entropy": entropy}
            motif_ids_by_window.append(motif_ids(features, bins))

        pre_idx = [i for i, w in enumerate(hazard["windows_seen"]) if w < args.hazard_start_window]
        hazard_idx = [
            i
            for i, w in enumerate(hazard["windows_seen"])
            if args.hazard_start_window <= w <= (args.hazard_start_window + args.hazard_duration_windows - 1)
        ]
        hist_pre = motif_hist(motif_ids_by_window[pre_idx[0]], num_motifs) if pre_idx else np.zeros(num_motifs)
        hist_haz = motif_hist(motif_ids_by_window[hazard_idx[0]], num_motifs) if hazard_idx else np.zeros(num_motifs)
        for idx in pre_idx[1:]:
            hist_pre += motif_hist(motif_ids_by_window[idx], num_motifs)
        for idx in hazard_idx[1:]:
            hist_haz += motif_hist(motif_ids_by_window[idx], num_motifs)
        if hist_pre.sum() > 0:
            hist_pre = hist_pre / hist_pre.sum()
        if hist_haz.sum() > 0:
            hist_haz = hist_haz / hist_haz.sum()
        jsd_ring = jsd(hist_pre, hist_haz)

        spike = _compute_spike(
            [float(np.mean(m)) for m in hazard["mismatch_by_window"]],
            args.hazard_start_window,
            args.hazard_duration_windows,
        )

        align_pre = [hazard["alignment_by_window"][i] for i in pre_idx]
        align_haz = [hazard["alignment_by_window"][i] for i in hazard_idx]
        align_delta = alignment_delta(align_pre, align_haz)
        align_p, align_null_mean, align_null_std = alignment_shift_null(
            hazard["axis_bias_by_window"],
            pref_axis_np,
            ring_mask_np,
            pre_idx,
            hazard_idx,
            args.shuffle_n,
            np.random.default_rng(seed + 7),
        )

        role_table = _motif_role_table(
            motif_ids_by_window,
            hazard["axis_bias_by_window"],
            ring_mask_np,
            pref_axis_np,
            hazard_idx,
            args.top_n,
        )
        roles_path = out_dir / "hazard" / f"motif_roles_seed{seed}.csv"
        with roles_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=["motif_id", "support", "mean_align"])
            writer.writeheader()
            writer.writerows(role_table)

        coverage_pre = top_n_coverage(hist_pre, args.top_n)
        coverage_haz = top_n_coverage(hist_haz, args.top_n)

        pass_seed = (
            spike >= args.spike_min
            and align_delta >= args.align_delta_min
            and align_p <= args.align_p_max
            and hazard.get("accept_mean", 0.0) >= args.accept_min
        )

        rows.append(
            {
                "seed": seed,
                "status": hazard.get("status"),
                "spike": spike,
                "jsd_ring_pre_hazard": jsd_ring,
                "alignment_delta": align_delta,
                "alignment_p": align_p,
                "alignment_null_mean": align_null_mean,
                "alignment_null_std": align_null_std,
                "coverage_pre": coverage_pre,
                "coverage_hazard": coverage_haz,
                "pass": pass_seed,
            }
        )

        if seed == seeds[0] and not pass_seed:
            break

    agg_path = out_dir / "agg.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "status",
                "spike",
                "jsd_ring_pre_hazard",
                "alignment_delta",
                "alignment_p",
                "alignment_null_mean",
                "alignment_null_std",
                "coverage_pre",
                "coverage_hazard",
                "pass",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_path = out_dir / "PHASE15_MOTIF_ROUTING_SEMANTICS_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 15 motif routing semantics v1\n\n")
        fh.write("## Summary\n\n")
        fh.write("| seed | status | spike | jsd_ring | alignment_delta | alignment_p | pass |\n")
        fh.write("| ---: | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            fh.write(
                f"| {row['seed']} | {row['status']} | {_fmt(row.get('spike'))} | "
                f"{_fmt(row.get('jsd_ring_pre_hazard'))} | {_fmt(row.get('alignment_delta'))} | "
                f"{_fmt(row.get('alignment_p'))} | {row.get('pass')} |\n"
            )
        fh.write("\n")
        fh.write(f"spike_min={args.spike_min} align_delta_min={args.align_delta_min} align_p_max={args.align_p_max}\n")


if __name__ == "__main__":
    main()
