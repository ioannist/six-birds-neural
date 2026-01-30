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
from ratchet_gpu.interventions import (
    apply_sigma_flip,
    apply_sigma_randomize,
    apply_w_lesion_redistribute,
    check_w_invariants,
    parse_rect,
)
from ratchet_gpu.params import Params
from ratchet_gpu.setpoints import (
    block_mask,
    block_mean,
    masked_distance,
    pre_peak_post,
    target_distance,
)
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


def _validate_injury_schedule(start: int, duration: int, max_windows: int) -> None:
    if start < 1:
        raise ValueError("injury_window must be >= 1")
    if duration < 1:
        raise ValueError("injury_duration_windows must be >= 1")
    if start + duration - 1 > max_windows:
        raise ValueError("injury window must fit within max_windows")


def _injury_active_window(window_idx: int, injury_window: int, injury_duration: int) -> bool:
    return injury_window <= window_idx <= (injury_window + injury_duration - 1)


def _injury_apply_between_windows(injury_window: int, injury_duration: int) -> List[int]:
    # Apply between window (w-1) and w so that window w reflects the injury.
    return [injury_window - 1 + i for i in range(injury_duration)]


def _pass_recovery(
    spike_c: float,
    rec_c: float,
    adv: float,
    post_adv: float,
    spike_min: float,
    recovery_min: float,
    adv_min: float,
    post_delta_min: float,
) -> bool:
    return (
        spike_c >= spike_min
        and rec_c >= recovery_min
        and (adv >= adv_min or post_adv >= post_delta_min)
    )


def _pass_suppression(
    spike_c: float,
    spike_d: float,
    damage_c: float,
    damage_d: float,
    suppression_frac_max: float,
    damage_max: float,
    damage_adv_min: float,
) -> bool:
    return (
        spike_c <= spike_d * suppression_frac_max
        and damage_c <= damage_max
        and damage_c <= damage_d - damage_adv_min
    )


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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _case_params(base: Params, case: str, eta: float, b_w_scale: float, b_k_scale: float) -> Params:
    if case == "coupled":
        overrides = {"eta": eta, "p3_on": False, "p6_on": False}
    elif case == "decoupled":
        overrides = {"eta": 0.0, "p3_on": False, "p6_on": False}
    else:
        raise ValueError(f"Unknown case: {case}")
    if b_w_scale != 1.0:
        overrides["B_w"] = max(1, int(round(base.B_w * b_w_scale)))
    if b_k_scale != 1.0:
        overrides["B_k"] = max(1, int(round(base.B_k * b_k_scale)))
    return Params.from_dict(base, overrides)


def _apply_injury_sigma(state: Any, flat_idx: torch.Tensor, layers: List[int], mode: str) -> None:
    if mode == "none":
        return
    if mode == "flip":
        apply_sigma_flip(state, flat_idx, layers=layers)
    elif mode == "random":
        apply_sigma_randomize(state, flat_idx, layers=layers)
    else:
        raise ValueError(f"Unknown injury sigma mode: {mode}")


def run_case(
    params: Params,
    seed: int,
    out_dir: Path,
    case: str,
    burn_sweeps: float,
    window_sweeps: float,
    max_windows: int,
    last_m: int,
    injury_window: int,
    injury_duration: int,
    injury_rect: str,
    injury_mode: str,
    injury_layers: List[int],
    injury_refresh_each_window: bool,
    injury_w_frac: float,
    injury_w_redistribute: bool,
    block: int,
    accept_min: float,
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
    cycle: List[str],
    resume: bool,
) -> Dict[str, Any]:
    case_dir = out_dir / case
    _ensure_dir(case_dir)
    jsonl_dir = case_dir / "jsonl"
    npz_dir = case_dir / "npz"
    targets_dir = case_dir / "targets"
    _ensure_dir(jsonl_dir)
    _ensure_dir(npz_dir)
    _ensure_dir(targets_dir)

    raw_path = case_dir / "raw.csv"
    agg_path = case_dir / "agg.csv"
    progress_path = case_dir / "progress.csv"

    if resume and agg_path.exists():
        with agg_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            if str(row.get("seed")) == str(seed) and row.get("status") not in {"RUNNING", ""}:
                return row

    shape = params.shape
    H, W = int(shape[-2]), int(shape[-1])
    mask, flat_idx = parse_rect(injury_rect, (H, W))
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
                    "injury_active",
                    "injury_applied",
                    "ep_rate",
                    "accept_window",
                    "mismatch_region",
                    "mismatch_outside",
                    "dist_target_sigma_coarse",
                    "corr_target_sigma_coarse",
                ]
            )
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            writer = csv.writer(ph)
            writer.writerow(
                ["seed", "window", "injury_active", "accept_window", "dist_target_sigma_coarse"]
            )

    run_start = time.monotonic()
    window_idx = 0
    status = "RUNNING"
    injury_active_next = False
    sigma_backup: torch.Tensor | None = None
    injury_applied_window = -1
    injury_applied_any = False

    window_records: List[Dict[str, Any]] = []
    sigma_coarse: List[torch.Tensor] = []
    accept_vals: List[float] = []

    def _backup_sigma(state: Any) -> None:
        nonlocal sigma_backup
        if sigma_backup is None:
            sigma_backup = state.sigma.clone()

    def _restore_sigma(state: Any) -> None:
        nonlocal sigma_backup
        if sigma_backup is not None:
            state.sigma.copy_(sigma_backup)
            sigma_backup = None

    def _apply_injury(state: Any) -> None:
        _backup_sigma(state)
        _apply_injury_sigma(state, flat_idx, injury_layers, injury_mode)
        if injury_w_redistribute and injury_w_frac > 0:
            info = apply_w_lesion_redistribute(
                state,
                params,
                flat_idx,
                layers=injury_layers if injury_layers else "all",
                frac=injury_w_frac,
            )
            ok, msg = check_w_invariants(state, params)
            if not ok:
                raise RuntimeError(f"W invariant failed after lesion: {msg}")
            return info
        return None

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal window_idx, status, injury_active_next, injury_applied_window, injury_applied_any
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

        injury_active = injury_active_next
        snapshot, _ = compute_snapshot(state, step, ep_ledger, accepted_frac, None)
        slim = _slim_snapshot(snapshot)
        slim["seed"] = seed
        slim["window"] = window_idx
        slim["injury_active"] = injury_active

        maps_dict = compute_spatial_maps(state, ["sigma", "mismatch", "k_entropy", "w_mass"])
        ok, _ = finite_check(maps_dict)
        if not ok:
            status = "FAIL_NAN_MAP"
            return

        sigma_l0 = maps_dict["sigma"][0]
        coarse = block_mean(sigma_l0.to(dtype=torch.float64), block)
        sigma_coarse.append(coarse)

        mismatch_region, mismatch_outside = _region_mean(maps_dict["mismatch"], mask_t)

        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        accept_window = float(snapshot.get("acceptedFracWindow", accepted_frac))
        accept_vals.append(accept_window)

        injury_applied = injury_applied_window == window_idx

        window_records.append(
            {
                "seed": seed,
                "window": window_idx,
                "injury_active": injury_active,
                "injury_applied": injury_applied,
                "ep_rate": ep_rate,
                "accept_window": accept_window,
                "mismatch_region": mismatch_region,
                "mismatch_outside": mismatch_outside,
            }
        )

        jsonl_handle.write(to_json_line(slim) + "\n")
        jsonl_handle.flush()

        npz_payload: Dict[str, np.ndarray] = {}
        for key in ["sigma", "mismatch", "k_entropy", "w_mass"]:
            grid = maps_dict[key]
            if key in {"k_entropy"}:
                npz_payload[f"{key}_i0"] = grid[0].detach().cpu().numpy()
            elif key in {"mismatch"}:
                npz_payload[f"{key}_i0"] = grid[0].detach().cpu().numpy()
            else:
                npz_payload[f"{key}_l0"] = grid[0].detach().cpu().numpy()
        npz_path = npz_dir / f"seed{seed}_win{window_idx:04d}.npz"
        np.savez(npz_path, **npz_payload)

        with progress_path.open("a", encoding="utf-8") as ph:
            writer = csv.writer(ph)
            writer.writerow([seed, window_idx, injury_active, accept_window, ""])

        next_window_idx = window_idx + 1
        injury_active_next_new = _injury_active_window(
            next_window_idx, injury_window, injury_duration
        )
        if injury_active_next_new:
            if injury_refresh_each_window or injury_applied_window < 0:
                try:
                    _apply_injury(state)
                    injury_applied_window = next_window_idx
                    injury_applied_any = True
                except RuntimeError:
                    status = "FAIL_BUDGET"
                    return
        elif injury_active_next:
            _restore_sigma(state)
        injury_active_next = injury_active_next_new

    def stop_cb(*_args) -> bool:
        return status != "RUNNING" or window_idx >= max_windows

    initial_state = State.initialize(params, seed=seed)
    if injury_window == 1:
        _apply_injury(initial_state)
        injury_active_next = True
        injury_applied_window = 1
        injury_applied_any = True

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

    if not sigma_coarse:
        status = "FAIL_CONFIG"

    pre_target_windows = [i for i in range(1, injury_window) if i >= 1]
    target = torch.stack([sigma_coarse[i - 1] for i in pre_target_windows]).mean(dim=0)
    target_path = targets_dir / f"seed{seed}_target.npz"
    np.savez(target_path, target_sigma_coarse=target.detach().cpu().numpy(), block=block)

    mask_coarse = block_mask(mask_t, block, threshold=0.0)
    dist_region_vals: List[float] = []
    dist_outside_vals: List[float] = []
    dist_global_vals: List[float] = []
    corr_vals: List[float] = []
    for coarse in sigma_coarse:
        dist_region_vals.append(masked_distance(coarse, target, mask_coarse))
        dist_outside_vals.append(masked_distance(coarse, target, ~mask_coarse))
        dist_global_vals.append(target_distance(coarse, target))
        a = coarse.flatten().to(dtype=torch.float64)
        b = target.flatten().to(dtype=torch.float64)
        denom = float(torch.linalg.norm(a) * torch.linalg.norm(b))
        corr = float(torch.dot(a, b).item() / denom) if denom > 0 else 0.0
        corr_vals.append(corr)

    for record, dist_g, dist_r, dist_o, corr in zip(
        window_records, dist_global_vals, dist_region_vals, dist_outside_vals, corr_vals
    ):
        record["dist_target_sigma_coarse"] = dist_g
        record["dist_target_sigma_region"] = dist_r
        record["dist_target_sigma_outside"] = dist_o
        record["corr_target_sigma_coarse"] = corr

    pre, peak, post = pre_peak_post(
        dist_region_vals, injury_window, injury_duration, last_m
    )
    raw_spike = peak - pre
    spike = max(0.0, raw_spike)
    denom = max(spike, 1e-12)
    recovery = (peak - post) / denom if spike > 0 else 0.0
    damage = post - pre
    accept_mean = float(np.mean(accept_vals[-last_m:])) if accept_vals else 0.0

    with raw_path.open("w", encoding="utf-8") as rh:
        writer = csv.DictWriter(
            rh,
            fieldnames=[
                "seed",
                "window",
                "injury_active",
                "injury_applied",
                "ep_rate",
                "accept_window",
                "mismatch_region",
                "mismatch_outside",
                "dist_target_sigma_coarse",
                "dist_target_sigma_region",
                "dist_target_sigma_outside",
                "corr_target_sigma_coarse",
            ],
        )
        writer.writeheader()
        for record in window_records:
            writer.writerow({k: record.get(k, "") for k in writer.fieldnames})

    result = {
        "seed": seed,
        "status": status,
        "pre": pre,
        "peak": peak,
        "post": post,
        "raw_spike": raw_spike,
        "spike": spike,
        "recovery_frac": recovery,
        "damage": damage,
        "accept_mean": accept_mean,
        "injury_window": injury_window,
        "injury_duration": injury_duration,
        "injury_mode": injury_mode,
        "injury_rect": injury_rect,
        "injury_applied_any": injury_applied_any,
    }

    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        writer = csv.DictWriter(ah, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13 pattern memory setpoints")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--preset", type=str, default="scripts/params/meta_null_coupled_eta1.00_layers3.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--b-w-scale", type=float, default=1.0)
    parser.add_argument("--b-k-scale", type=float, default=1.0)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--injury-window", type=int, default=8)
    parser.add_argument("--injury-duration-windows", type=int, default=1)
    parser.add_argument("--injury-rect", type=str, default="8:16,8:16")
    parser.add_argument("--injury-sigma", type=str, choices=["random", "flip", "none"], default="random")
    parser.add_argument("--injury-mode", type=str, choices=["random", "flip", "none"], default=None)
    parser.add_argument("--injury-layers", type=str, default="0")
    parser.add_argument("--injury-refresh-each-window", action="store_true")
    parser.add_argument("--injury-w-frac", type=float, default=0.0)
    parser.add_argument("--injury-w-redistribute", action="store_true")

    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--spike-min", type=float, default=0.02)
    parser.add_argument("--recovery-min", type=float, default=0.20)
    parser.add_argument("--suppression-frac-max", type=float, default=0.50)
    parser.add_argument("--damage-adv-min", type=float, default=0.005)
    parser.add_argument("--damage-max", type=float, default=0.0)
    parser.add_argument("--adv-min", type=float, default=0.10)
    parser.add_argument("--post-delta-min", type=float, default=0.01)

    args = parser.parse_args()

    _validate_injury_schedule(args.injury_window, args.injury_duration_windows, args.max_windows)
    if args.block_size <= 0:
        raise ValueError("block-size must be > 0")

    preset = _load_preset(Path(args.preset))
    base_params = _as_params(preset, {"device": torch.device(args.device)})
    if base_params.layers < 2:
        raise ValueError("layers must be >= 2")
    if args.injury_layers == "all":
        injury_layers = list(range(base_params.layers))
    else:
        injury_layers = _parse_layers(args.injury_layers, base_params.layers)
    injury_mode = args.injury_mode if args.injury_mode is not None else args.injury_sigma

    eta_coupled = base_params.eta if args.eta is None else args.eta
    coupled_params = _case_params(base_params, "coupled", eta_coupled, args.b_w_scale, args.b_k_scale)
    decoupled_params = _case_params(base_params, "decoupled", eta_coupled, args.b_w_scale, args.b_k_scale)

    cycle = _cycle_list()
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    start_total = time.monotonic()
    seeds = _parse_seeds(args.seeds)

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        coupled = run_case(
            coupled_params,
            seed=seed,
            out_dir=out_dir,
            case="coupled",
            burn_sweeps=args.burn_in_sweeps,
            window_sweeps=args.window_sweeps,
            max_windows=args.max_windows,
            last_m=args.last_m,
            injury_window=args.injury_window,
            injury_duration=args.injury_duration_windows,
            injury_rect=args.injury_rect,
            injury_mode=injury_mode,
            injury_layers=injury_layers,
            injury_refresh_each_window=args.injury_refresh_each_window,
            injury_w_frac=args.injury_w_frac,
            injury_w_redistribute=args.injury_w_redistribute,
            block=args.block_size,
            accept_min=args.accept_min,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
            cycle=cycle,
            resume=args.resume,
        )
        decoupled = run_case(
            decoupled_params,
            seed=seed,
            out_dir=out_dir,
            case="decoupled",
            burn_sweeps=args.burn_in_sweeps,
            window_sweeps=args.window_sweeps,
            max_windows=args.max_windows,
            last_m=args.last_m,
            injury_window=args.injury_window,
            injury_duration=args.injury_duration_windows,
            injury_rect=args.injury_rect,
            injury_mode=injury_mode,
            injury_layers=injury_layers,
            injury_refresh_each_window=args.injury_refresh_each_window,
            injury_w_frac=args.injury_w_frac,
            injury_w_redistribute=args.injury_w_redistribute,
            block=args.block_size,
            accept_min=args.accept_min,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
            cycle=cycle,
            resume=args.resume,
        )

        spike_c = coupled["spike"]
        spike_d = decoupled["spike"]
        raw_spike_c = coupled["raw_spike"]
        raw_spike_d = decoupled["raw_spike"]
        rec_c = coupled["recovery_frac"]
        rec_d = decoupled["recovery_frac"]
        damage_c = coupled["damage"]
        damage_d = decoupled["damage"]
        adv = rec_c - rec_d
        post_adv = decoupled["post"] - coupled["post"]
        injury_gate = coupled.get("injury_applied_any") and decoupled.get("injury_applied_any")
        spike_gate = spike_d >= args.spike_min
        recovery_ok = _pass_recovery(
            spike_c,
            rec_c,
            adv,
            post_adv,
            args.spike_min,
            args.recovery_min,
            args.adv_min,
            args.post_delta_min,
        )
        suppression_ok = _pass_suppression(
            spike_c,
            spike_d,
            damage_c,
            damage_d,
            args.suppression_frac_max,
            args.damage_max,
            args.damage_adv_min,
        ) and injury_gate
        pass_flag = (
            spike_gate
            and (recovery_ok or suppression_ok)
            and coupled["accept_mean"] >= args.accept_min
            and injury_gate
        )
        if pass_flag:
            fail_reason = "OK_RECOVERY" if recovery_ok else "OK_SUPPRESSION"
        elif not injury_gate:
            fail_reason = "NO_INJURY"
        elif not spike_gate:
            fail_reason = "NO_DAMAGE"
        elif spike_c < args.spike_min or rec_c < args.recovery_min:
            fail_reason = "RECOVERY_FAIL"
        elif spike_c > spike_d * args.suppression_frac_max:
            fail_reason = "SUPPRESSION_FAIL"
        elif damage_c > args.damage_max or damage_c > damage_d - args.damage_adv_min:
            fail_reason = "NO_ADVANTAGE"
        else:
            fail_reason = "NO_ADVANTAGE"
        pass_path = "RECOVERY" if recovery_ok else "SUPPRESSION" if suppression_ok else "FAIL"
        results.append(
            {
                "seed": seed,
                "status": "PASS" if pass_flag else "FAIL",
                "fail_reason": fail_reason,
                "pass_path": pass_path,
                "raw_spike_coupled": raw_spike_c,
                "spike_coupled": spike_c,
                "recovery_frac_coupled": rec_c,
                "damage_coupled": damage_c,
                "raw_spike_decoupled": raw_spike_d,
                "spike_decoupled": spike_d,
                "recovery_frac_decoupled": rec_d,
                "damage_decoupled": damage_d,
                "advantage": adv,
                "post_delta": post_adv,
                "accept_mean_coupled": coupled["accept_mean"],
                "accept_mean_decoupled": decoupled["accept_mean"],
            }
        )
        if seed == seeds[0] and not pass_flag:
            print("PHASE13_GATE=FAIL")
            break

    agg_path = out_dir / "agg.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as ah:
        writer = csv.DictWriter(ah, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    report_path = out_dir / "PHASE13_SETPOINT_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 13 pattern memory setpoints v1\n\n")
        fh.write("| seed | status | fail_reason | pass_path | spike_c | recovery_c | damage_c | spike_d | recovery_d | damage_d | advantage |\n")
        fh.write("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in results:
            fh.write(
                f"| {row['seed']} | {row['status']} | {row['fail_reason']} | {row['pass_path']} | "
                f"{row['spike_coupled']:.6g} | {row['recovery_frac_coupled']:.6g} | "
                f"{row['damage_coupled']:.6g} | {row['spike_decoupled']:.6g} | "
                f"{row['recovery_frac_decoupled']:.6g} | {row['damage_decoupled']:.6g} | "
                f"{row['advantage']:.6g} |\n"
            )
        fh.write("\nDefinitions:\n")
        fh.write("- raw_spike = peak - pre\n")
        fh.write("- spike = max(raw_spike, 0)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
