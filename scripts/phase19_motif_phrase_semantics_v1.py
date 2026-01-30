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
    apply_sigma_flip,
    apply_sigma_randomize,
    check_k_invariants,
    parse_rect,
)
from ratchet_gpu.motifs import (
    MotifBins,
    build_bins,
    dictionary_score,
    dictionary_weights,
    jsd,
    motif_entropy,
    motif_histogram,
    motif_ids,
    symmetric_edges,
)
from ratchet_gpu.params import Params
from ratchet_gpu.semantics import hazard_center, ring_masks_from_rect, shift_null_corr
from ratchet_gpu.sim import run_sim
from ratchet_gpu.spatial import compute_spatial_maps, finite_check, k_radial_focus_grid
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
        if feat not in {"k_axis_bias", "k_entropy", "k_radial_focus"}:
            raise ValueError(f"unsupported motif feature {feat}")
    return feats


def _validate_bins(bins_axis: int, bins_entropy: int, bins_radial: int) -> None:
    if bins_axis < 2:
        raise ValueError("bins_axis_bias must be >= 2")
    if bins_entropy < 2:
        raise ValueError("bins_entropy must be >= 2")
    if bins_radial < 2:
        raise ValueError("bins_radial must be >= 2")


def _validate_ring_thickness(value: int) -> None:
    if value < 1:
        raise ValueError("ring_thickness must be >= 1")


def _validate_phrase_args(
    phrase_mode: str,
    phrase_start: int,
    hazard_duration: int,
    phrase_use_hazard_only: bool,
) -> None:
    if phrase_mode != "alternating":
        raise ValueError("phrase_mode must be alternating")
    if phrase_start not in (0, 1):
        raise ValueError("phrase_start must be 0 or 1")
    if hazard_duration < 2:
        raise ValueError("hazard_duration_windows must be >= 2 for phrase decoding")
    if not phrase_use_hazard_only:
        raise ValueError("phrase_use_hazard_windows_only must be true in v1")


def _normalize_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_config(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(v) for v in value]
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, (np.integer, np.floating)):
        return _normalize_config(value.item())
    return value


def _write_run_config(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_normalize_config(payload), indent=2, sort_keys=True))


def _load_run_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _parse_inject_layers(value: str, total: int) -> List[int]:
    if total <= 0:
        raise ValueError("no interfaces available for injection")
    if value == "all":
        return list(range(total))
    layers = [int(x) for x in value.split(",") if x.strip()]
    if not layers:
        raise ValueError("inject_layers cannot be empty")
    for layer in layers:
        if layer < 0 or layer >= total:
            raise ValueError("inject_layers out of range")
    return layers


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
    case: str,
    token: str,
    state: State,
    params: Params,
    ring_idx: torch.Tensor,
    center: Tuple[float, float],
    interfaces: List[int],
    strength: float,
    rng: torch.Generator,
) -> bool:
    if case == "inject_in" or (case == "phrase_alt" and token == "IN"):
        apply_k_redistribute_radial_inward_in_ring(
            state,
            params,
            ring_idx,
            center=center,
            interfaces=interfaces,
            strength=strength,
            rng=rng,
        )
        return True
    if case == "inject_out" or (case == "phrase_alt" and token == "OUT"):
        apply_k_redistribute_radial_outward_in_ring(
            state,
            params,
            ring_idx,
            center=center,
            interfaces=interfaces,
            strength=strength,
            rng=rng,
        )
        return True
    return False


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
    if key in {"k_axis_bias", "k_entropy", "k_r2", "k_radial_focus", "mismatch"}:
        arr = arr[interface]
    return arr.detach().cpu().numpy()


def _run_pre_windows(
    params: Params,
    state: State,
    seed: int,
    steps: int,
    report_every: int,
    interface: int,
    region_mask: np.ndarray,
    motif_features: List[str],
    center: Tuple[float, float],
    rng_state_in: torch.Tensor | None,
) -> Tuple[State, torch.Tensor | None, Dict[str, List[np.ndarray]], List[float], List[float]]:
    diag_state = None
    feature_store: Dict[str, List[np.ndarray]] = {k: [] for k in motif_features}
    mismatch_vals: List[float] = []
    accept_vals: List[float] = []

    def pre_cb(st: State, step: int, ep_ledger: Dict[str, Any], accepted_frac: float) -> None:
        nonlocal diag_state
        snap, diag_state = compute_snapshot(st, step, ep_ledger, accepted_frac, diag_state)
        maps_dict = compute_spatial_maps(st, ["k_axis_bias", "k_entropy", "mismatch"])
        if "k_radial_focus" in motif_features:
            maps_dict["k_radial_focus"] = k_radial_focus_grid(st, center)
        ok, _ = finite_check(maps_dict)
        if not ok:
            raise RuntimeError("non-finite map during pre window")
        for feat in motif_features:
            feature_store[feat].append(_select_map(maps_dict, feat, interface))
        mismatch = _select_map(maps_dict, "mismatch", interface)
        mismatch_region = float(mismatch[region_mask].mean()) if region_mask.any() else 0.0
        mismatch_vals.append(mismatch_region)
        accept_vals.append(_get_accept(snap))

    seed_for_run = seed if rng_state_in is None else None
    summary = run_sim(
        params,
        seed=seed_for_run,
        steps=steps,
        report_every=report_every,
        report_callback=pre_cb,
        initial_state=state,
        initial_rng_state=rng_state_in,
        return_state=True,
    )
    state = summary["state"]
    rng_state = summary.get("rng_state")
    return state, rng_state, feature_store, mismatch_vals, accept_vals


def _hazard_active_for_window(window_idx: int, hazard_start: int, hazard_duration: int) -> bool:
    if hazard_duration <= 0:
        return False
    return hazard_start <= window_idx <= hazard_start + hazard_duration - 1


def _hazard_applied_windows(
    window_offset: int,
    max_windows: int,
    hazard_start: int,
    hazard_duration: int,
    hazard_refresh_each_window: bool,
) -> List[int]:
    applied: List[int] = []
    last_applied = None
    for window_idx in range(1, max_windows + 1):
        window_idx_global = window_offset + window_idx
        hazard_active = _hazard_active_for_window(window_idx_global, hazard_start, hazard_duration)
        if hazard_active and (hazard_refresh_each_window or last_applied is None):
            applied.append(window_idx_global)
            last_applied = window_idx_global
        elif not hazard_active:
            last_applied = None
    return applied


def _phrase_token_for_window(
    window_idx: int,
    hazard_start: int,
    hazard_duration: int,
    phrase_start: int,
    phrase_use_hazard_only: bool,
) -> str:
    if phrase_use_hazard_only and not _hazard_active_for_window(window_idx, hazard_start, hazard_duration):
        return "none"
    offset = window_idx - hazard_start
    if offset < 0:
        return "none"
    return "OUT" if (offset + phrase_start) % 2 == 0 else "IN"


def _top_n_summary(hist: np.ndarray, top_n: int) -> Tuple[str, str]:
    if hist.size == 0:
        return "", ""
    count = max(1, int(top_n))
    ids = np.argsort(hist)[::-1][:count]
    fracs = hist[ids]
    ids_str = "|".join(str(int(idx)) for idx in ids)
    fracs_str = "|".join(f"{float(val):.6g}" for val in fracs)
    return ids_str, fracs_str


def _run_condition(
    case: str,
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
    inject_interfaces: List[int],
    center: Tuple[float, float],
    strength: float,
    interface: int,
    region_mask: np.ndarray,
    ring_mask: np.ndarray,
    outside_mask: np.ndarray,
    bins: MotifBins,
    motif_features: List[str],
    num_motifs: int,
    out_dir: Path,
    snapshot_every: int,
    max_seconds: float,
    progress: bool,
    window_offset: int,
    phrase_start: int,
    phrase_use_hazard_only: bool,
    top_n: int,
    hazard_mask_np: np.ndarray,
) -> Tuple[List[np.ndarray], List[float], List[float], List[float], List[float], List[str]]:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "jsonl").mkdir(parents=True, exist_ok=True)
    npz_dir = case_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    raw_writer = _build_csv_writer(
        case_dir / "raw.csv",
        [
            "case",
            "seed",
            "window_index",
            "step",
            "hazard_active",
            "hazard_applied",
            "token",
            "acceptedFracWindow",
            "mismatch_region",
            "mismatch_outside",
            "mismatch_abs_mean",
            "k_axis_bias_focus",
            "radial_ring_mean",
            "top_motif_ids",
            "top_motif_fracs",
            "motif_entropy",
            "k_entropy_region",
            "k_entropy_outside",
        ],
    )
    jsonl_path = case_dir / "jsonl" / f"{case}_seed{seed}.jsonl"
    jsonl_handle = jsonl_path.open("w")

    diag_state = None
    window_idx = 0
    hazard_active_next = False
    sigma_backup: torch.Tensor | None = None
    ring_hists: List[np.ndarray] = []
    mismatch_region_vals: List[float] = []
    accept_vals: List[float] = []
    focus_vals: List[float] = []
    radial_ring_vals: List[float] = []
    tokens: List[str] = []
    hazard_applied_windows: set[int] = set()

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
        token = (
            _phrase_token_for_window(
                window_idx_global,
                hazard_start_window,
                hazard_duration,
                phrase_start,
                phrase_use_hazard_only,
            )
            if case == "phrase_alt"
            else "none"
        )
        tokens.append(token)

        snap, diag_state = compute_snapshot(st, step, ep_ledger, accepted_frac, diag_state)
        slim = _slim_snapshot(snap)
        slim.update(
            {
                "seed": seed,
                "case": case,
                "window": window_idx_global,
                "hazard_active": hazard_active,
            }
        )
        jsonl_handle.write(to_json_line(slim) + "\n")
        jsonl_handle.flush()

        maps_dict = compute_spatial_maps(st, ["k_axis_bias", "k_entropy", "mismatch", "sigma"])
        maps_dict["k_radial_focus"] = k_radial_focus_grid(st, center)
        ok, _ = finite_check(maps_dict)
        if not ok:
            raise RuntimeError("non-finite map during hazard run")
        features = {feat: _select_map(maps_dict, feat, interface) for feat in motif_features}
        ids = motif_ids(features, bins)
        ring_hist = motif_histogram(ids, ring_mask, num_motifs)
        ring_hists.append(ring_hist)

        mismatch_map = _select_map(maps_dict, "mismatch", interface)
        mismatch_region = float(mismatch_map[region_mask].mean()) if region_mask.any() else 0.0
        mismatch_outside = float(mismatch_map[outside_mask].mean()) if outside_mask.any() else 0.0
        mismatch_abs_mean = float(mismatch_map.mean()) if mismatch_map.size else 0.0
        mismatch_region_vals.append(mismatch_region)

        k_axis_bias = _select_map(maps_dict, "k_axis_bias", interface)
        ring_mean = float(k_axis_bias[ring_mask].mean()) if ring_mask.any() else 0.0
        outside_mean = float(k_axis_bias[outside_mask].mean()) if outside_mask.any() else 0.0
        focus = ring_mean - outside_mean
        focus_vals.append(focus)

        radial_map = _select_map(maps_dict, "k_radial_focus", interface)
        radial_ring = float(radial_map[ring_mask].mean()) if ring_mask.any() else 0.0
        radial_ring_vals.append(radial_ring)

        k_entropy = _select_map(maps_dict, "k_entropy", interface)
        k_entropy_region = float(k_entropy[region_mask].mean()) if region_mask.any() else 0.0
        k_entropy_outside = float(k_entropy[outside_mask].mean()) if outside_mask.any() else 0.0

        accept_window = _get_accept(snap)
        accept_vals.append(accept_window)

        top_ids, top_fracs = _top_n_summary(ring_hist, top_n)
        entropy_val = motif_entropy(ring_hist)

        _write_row(
            raw_writer,
            {
                "case": case,
                "seed": seed,
                "window_index": window_idx_global,
                "step": step,
                "hazard_active": hazard_active,
                "hazard_applied": window_idx_global in hazard_applied_windows,
                "token": token,
                "acceptedFracWindow": accept_window,
                "mismatch_region": mismatch_region,
                "mismatch_outside": mismatch_outside,
                "mismatch_abs_mean": mismatch_abs_mean,
                "k_axis_bias_focus": focus,
                "radial_ring_mean": radial_ring,
                "top_motif_ids": top_ids,
                "top_motif_fracs": top_fracs,
                "motif_entropy": entropy_val,
                "k_entropy_region": k_entropy_region,
                "k_entropy_outside": k_entropy_outside,
            },
        )

        if progress:
            print(
                f"{case} seed={seed} win={window_idx_global} hazard={int(hazard_active)} "
                f"accept={accept_window:.3g} mismatch={mismatch_region:.3g}",
                flush=True,
            )

        if snapshot_every > 0 and window_idx % snapshot_every == 0:
            sigma_map = maps_dict["sigma"][0].detach().cpu().numpy()
            payload = {
                "k_axis_bias_i0": k_axis_bias,
                "k_entropy_i0": k_entropy,
                "k_radial_focus_i0": radial_map,
                "sigma_l0": sigma_map,
                "mismatch_i0": mismatch_map,
                "hazard_mask": hazard_mask_np,
                "ring_mask": ring_mask,
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
                hazard_applied_windows.add(next_window_idx_global)
                token_next = (
                    _phrase_token_for_window(
                        next_window_idx_global,
                        hazard_start_window,
                        hazard_duration,
                        phrase_start,
                        phrase_use_hazard_only,
                    )
                    if case == "phrase_alt"
                    else "none"
                )
                applied = _apply_intervention(
                    case,
                    token_next,
                    st,
                    params,
                    ring_idx,
                    center,
                    inject_interfaces,
                    strength,
                    torch.Generator(device=st.device),
                )
                if applied:
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
        hazard_applied_windows.add(window_offset + 1)
        token_first = (
            _phrase_token_for_window(
                window_offset + 1,
                hazard_start_window,
                hazard_duration,
                phrase_start,
                phrase_use_hazard_only,
            )
            if case == "phrase_alt"
            else "none"
        )
        applied = _apply_intervention(
            case,
            token_first,
            state,
            params,
            ring_idx,
            center,
            inject_interfaces,
            strength,
            torch.Generator(device=state.device),
        )
        if applied:
            ok, msg = check_k_invariants(state, params)
            if not ok:
                raise RuntimeError(f"K invariants failed: {msg}")
        hazard_active_next = True

    seed_for_run = seed if rng_state is None else None
    run_sim(
        params,
        seed=seed_for_run,
        steps=window_steps * max_windows,
        report_every=window_steps,
        report_callback=report_cb,
        stop_callback=stop_cb,
        initial_state=state,
        initial_rng_state=rng_state,
        return_state=True,
    )
    jsonl_handle.close()
    _close_writer(raw_writer)
    return ring_hists, mismatch_region_vals, accept_vals, focus_vals, radial_ring_vals, tokens


def _hazard_slice(values: List[float], hazard_duration: int) -> List[float]:
    if not values:
        return []
    return values[: max(0, min(hazard_duration, len(values)))]


def _write_report(report_path: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# Phase 19 Motif Phrase Semantics\n\n"]
    lines.append("## Phrase decoding\n\n")
    lines.append(f"- hazard_start_window: {args.hazard_start_window}\n")
    lines.append(f"- hazard_duration_windows: {args.hazard_duration_windows}\n")
    lines.append(f"- shuffle_n: {args.shuffle_n}\n")
    lines.append(f"- alignment_min: {args.alignment_min}\n")
    lines.append(f"- p_max: {args.p_max}\n\n")
    lines.append(
        "| seed | status | fail_reason | spike_control | jsd_out_in | alignment | alignment_p | accept_control | accept_phrase |\n"
    )
    lines.append("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['status']} | {row['fail_reason']} | "
            f"{row['spike_control']:.4g} | {row['jsd_out_in']:.4g} | "
            f"{row['alignment']:.4g} | {row['alignment_p']:.4g} | "
            f"{row['accept_mean_control']:.4g} | {row['accept_mean_phrase']:.4g} |\n"
        )
    report_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 motif phrase semantics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=int, default=150)
    parser.add_argument("--window-sweeps", type=int, default=80)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")

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
    parser.add_argument("--bins-radial", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)

    parser.add_argument("--phrase-mode", default="alternating")
    parser.add_argument("--phrase-start", type=int, default=0)
    parser.add_argument("--phrase-use-hazard-windows-only", action="store_true", default=True)

    parser.add_argument("--intervention-strength", type=float, default=1.0)
    parser.add_argument("--ablate-frac", type=float, default=1.0)
    parser.add_argument("--inject-layers", default="0")

    parser.add_argument("--shuffle-n", type=int, default=200)
    parser.add_argument("--spike-min", type=float, default=0.01)
    parser.add_argument("--jsd-out-in-min", type=float, default=0.01)
    parser.add_argument("--alignment-min", type=float, default=0.10)
    parser.add_argument("--p-max", type=float, default=0.10)

    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("no seeds specified")
    _validate_hazard_schedule(args.hazard_start_window, args.hazard_duration_windows, args.max_windows)
    _validate_ring_thickness(args.ring_thickness)
    _validate_bins(args.bins_axis_bias, args.bins_entropy, args.bins_radial)
    motif_features = _parse_motif_features(args.motif_features)
    _validate_phrase_args(
        args.phrase_mode,
        args.phrase_start,
        args.hazard_duration_windows,
        args.phrase_use_hazard_windows_only,
    )
    if not args.hazard_refresh_each_window:
        raise ValueError("hazard_refresh_each_window is required for phase19")

    preset = _load_preset(Path(args.preset))
    params = _as_params(preset, {"device": args.device})
    params = Params(**{**params.__dict__, "p3_on": False, "p6_on": False})
    device = params.resolved_device()
    device_str = str(device)

    if len(params.shape) != 2:
        raise ValueError("Phase19 expects 2D lattice shape")
    shape = params.shape
    region_np, ring_np, outside_np = ring_masks_from_rect(args.hazard_rect, shape, args.ring_thickness)
    center = hazard_center(args.hazard_rect, shape)
    hazard_mask, hazard_flat_idx = parse_rect(args.hazard_rect, shape)
    ring_idx = torch.as_tensor(np.flatnonzero(ring_np), dtype=torch.long, device=device)

    interface_idx = int(args.motif_interface)
    total_interfaces = max(0, params.layers - 1)
    if interface_idx < 0 or interface_idx >= total_interfaces:
        raise ValueError("motif_interface out of range")
    inject_interfaces = _parse_inject_layers(args.inject_layers, total_interfaces)

    N = math.prod(params.shape)
    expected = _expected_proposals_per_step(N, device_str, params.kernel_weights)
    burn_steps = int(math.ceil(args.burn_in_sweeps * N / expected))
    window_steps = int(math.ceil(args.window_sweeps * N / expected))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"
    config_payload = {
        "args": vars(args),
        "derived": {
            "device_resolved": device_str,
            "shape": list(shape),
            "burn_steps": burn_steps,
            "window_steps": window_steps,
            "expected_proposals_per_step": expected,
            "motif_features": motif_features,
            "inject_interfaces": inject_interfaces,
            "interface_idx": interface_idx,
            "hazard_center": [float(center[0]), float(center[1])],
            "seeds": seeds,
        },
    }
    if args.resume:
        if not config_path.exists():
            raise ValueError("resume requested but run_config.json is missing; use a new out-dir")
        existing = _load_run_config(config_path)
        if _normalize_config(existing) != _normalize_config(config_payload):
            raise ValueError("resume config mismatch; use a new out-dir or remove --resume")
    else:
        _write_run_config(config_path, config_payload)
    agg_path = out_dir / "agg.csv"
    agg_fields = [
        "seed",
        "status",
        "fail_reason",
        "spike_control",
        "jsd_out_in",
        "alignment",
        "alignment_p",
        "focus_alignment",
        "focus_alignment_p",
        "radial_ring_mean_out",
        "radial_ring_mean_in",
        "radial_ring_mean_out_minus_in",
        "accept_mean_control",
        "accept_mean_phrase",
    ]
    agg_writer = _build_csv_writer(agg_path, agg_fields)

    report_rows: List[Dict[str, Any]] = []
    total_start = time.monotonic()

    for seed in seeds:
        if time.monotonic() - total_start > args.max_seconds_total:
            break

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

        pre_windows = max(0, args.hazard_start_window - 1)
        if pre_windows > 0:
            pre_steps = window_steps * pre_windows
            base_state, rng_state, baseline_features, pre_mismatch, _pre_accept = _run_pre_windows(
                params,
                base_state,
                seed,
                pre_steps,
                window_steps,
                interface_idx,
                region_np,
                motif_features,
                center,
                rng_state,
            )
        else:
            baseline_features = {k: [] for k in motif_features}
            pre_mismatch = []

        if motif_features and not baseline_features[motif_features[0]]:
            raise RuntimeError("no baseline features collected before hazard")

        bins_by_key = {
            "k_axis_bias": args.bins_axis_bias,
            "k_entropy": args.bins_entropy,
            "k_radial_focus": args.bins_radial,
        }
        bins_by_key = {k: bins_by_key[k] for k in motif_features}
        edges_by_key: Dict[str, np.ndarray] = {}
        if "k_radial_focus" in bins_by_key:
            values = np.concatenate(
                [arr.ravel() for arr in baseline_features["k_radial_focus"]], axis=0
            )
            edges_by_key["k_radial_focus"] = symmetric_edges(values, bins_by_key["k_radial_focus"])
        bins = build_bins(baseline_features, bins_by_key, edges_by_key=edges_by_key)
        num_motifs = 1
        for key in bins.edges:
            num_motifs *= bins.bins[key]

        base_state = _clone_state(base_state)

        cases = ["control", "inject_out", "inject_in", "phrase_alt"]
        case_data: Dict[str, Dict[str, Any]] = {}

        for case in cases:
            ring_hists, mismatch_vals, accept_vals, focus_vals, radial_ring_vals, tokens = _run_condition(
                case,
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
                inject_interfaces,
                center,
                float(args.intervention_strength),
                interface_idx,
                region_np,
                ring_np,
                outside_np,
                bins,
                motif_features,
                num_motifs,
                out_dir,
                args.snapshot_every_windows,
                time.monotonic() + args.max_seconds_per_run,
                args.progress,
                pre_windows,
                args.phrase_start,
                args.phrase_use_hazard_windows_only,
                args.top_n,
                hazard_mask.cpu().numpy(),
            )
            case_data[case] = {
                "ring_hists": ring_hists,
                "mismatch_vals": mismatch_vals,
                "accept_vals": accept_vals,
                "focus_vals": focus_vals,
                "radial_ring_vals": radial_ring_vals,
                "tokens": tokens,
            }

        control_vals = case_data["control"]
        pre_mean = float(np.mean(pre_mismatch)) if pre_mismatch else 0.0
        control_hazard = _hazard_slice(control_vals["mismatch_vals"], args.hazard_duration_windows)
        peak = float(np.max(control_hazard)) if control_hazard else pre_mean
        spike_control = max(0.0, peak - pre_mean)
        accept_control = _hazard_slice(control_vals["accept_vals"], args.hazard_duration_windows)
        accept_mean_control = _mean(accept_control)

        hazard_count = max(0, min(args.hazard_duration_windows, len(case_data["inject_out"]["ring_hists"])))
        out_hists = np.stack(case_data["inject_out"]["ring_hists"], axis=0) if hazard_count else np.zeros((0, num_motifs))
        in_hists = np.stack(case_data["inject_in"]["ring_hists"], axis=0) if hazard_count else np.zeros((0, num_motifs))
        out_hazard = out_hists[:hazard_count] if hazard_count else np.zeros((0, num_motifs))
        in_hazard = in_hists[:hazard_count] if hazard_count else np.zeros((0, num_motifs))
        p_out = out_hazard.mean(axis=0) if out_hazard.size else np.zeros((num_motifs,))
        p_in = in_hazard.mean(axis=0) if in_hazard.size else np.zeros((num_motifs,))
        jsd_out_in = float(jsd(p_out, p_in)) if p_out.size else 0.0

        radial_out = _hazard_slice(case_data["inject_out"]["radial_ring_vals"], args.hazard_duration_windows)
        radial_in = _hazard_slice(case_data["inject_in"]["radial_ring_vals"], args.hazard_duration_windows)
        radial_ring_mean_out = _mean(radial_out)
        radial_ring_mean_in = _mean(radial_in)
        radial_ring_mean_out_minus_in = radial_ring_mean_out - radial_ring_mean_in

        weights = dictionary_weights(p_out, p_in, eps=1e-9) if p_out.size else np.zeros((num_motifs,))
        phrase_hists = np.stack(case_data["phrase_alt"]["ring_hists"], axis=0) if case_data["phrase_alt"]["ring_hists"] else np.zeros((0, num_motifs))
        phrase_hazard = phrase_hists[:hazard_count] if hazard_count else np.zeros((0, num_motifs))
        scores = np.array([
            dictionary_score(hist, weights) for hist in phrase_hazard
        ], dtype=np.float64)
        signs = np.array([
            1.0 if (i + args.phrase_start) % 2 == 0 else -1.0 for i in range(len(scores))
        ], dtype=np.float64)

        rng = np.random.default_rng(seed)
        alignment, alignment_p, _align_null_mean, _align_null_std = shift_null_corr(
            scores,
            signs,
            args.shuffle_n,
            rng,
        )

        focus_vals = np.array(_hazard_slice(case_data["phrase_alt"]["focus_vals"], args.hazard_duration_windows), dtype=np.float64)
        focus_alignment, focus_alignment_p, _focus_null_mean, _focus_null_std = shift_null_corr(
            focus_vals,
            signs,
            args.shuffle_n,
            rng,
        )

        accept_phrase = _hazard_slice(case_data["phrase_alt"]["accept_vals"], args.hazard_duration_windows)
        accept_mean_phrase = _mean(accept_phrase)

        status = "PASS"
        fail_reason = ""
        if spike_control < args.spike_min:
            status = "FAIL"
            fail_reason = "NO_SPIKE"
        elif jsd_out_in < args.jsd_out_in_min:
            status = "FAIL"
            fail_reason = "NO_DICT_SEPARATION"
        elif alignment < args.alignment_min or alignment_p > args.p_max:
            status = "FAIL"
            fail_reason = "NO_PHRASE_DECODE"

        agg_row = {
            "seed": seed,
            "status": status,
            "fail_reason": fail_reason,
            "spike_control": spike_control,
            "jsd_out_in": jsd_out_in,
            "alignment": alignment,
            "alignment_p": alignment_p,
            "focus_alignment": focus_alignment,
            "focus_alignment_p": focus_alignment_p,
            "radial_ring_mean_out": radial_ring_mean_out,
            "radial_ring_mean_in": radial_ring_mean_in,
            "radial_ring_mean_out_minus_in": radial_ring_mean_out_minus_in,
            "accept_mean_control": accept_mean_control,
            "accept_mean_phrase": accept_mean_phrase,
        }
        _write_row(agg_writer, agg_row)
        report_rows.append(agg_row)

        print(
            f"PHASE19_SEED={seed} status={status} fail_reason={fail_reason} "
            f"jsd_out_in={jsd_out_in:.4g} alignment={alignment:.4g} p={alignment_p:.4g} spike={spike_control:.4g}",
            flush=True,
        )

        if seed == seeds[0] and status != "PASS":
            break

    _close_writer(agg_writer)
    report_path = out_dir / "PHASE19_MOTIF_PHRASE_REPORT.md"
    _write_report(report_path, report_rows, args)


if __name__ == "__main__":
    main()
