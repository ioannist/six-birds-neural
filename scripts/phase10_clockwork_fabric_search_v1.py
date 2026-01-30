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

from ratchet_gpu.clockwork import fabric_scores
from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.params import Params
from ratchet_gpu.sim import _cycle_list, run_sim
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
    with path.open() as fh:
        return json.load(fh)


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


def _match_cycle_weights(kw: Dict[str, float], cycle: List[str]) -> Dict[str, float]:
    matched = {k: (1.0 if k in cycle else 0.0) for k in kw.keys()}
    for name in cycle:
        matched.setdefault(name, 1.0)
    return matched


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


def _slim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    slim = dict(snapshot)
    for key in list(slim.keys()):
        if key in HEAVY_KEYS or key.endswith("_items_window"):
            slim.pop(key, None)
    return slim


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


def _summarize_tail(vals: List[float], last_m: int) -> Tuple[float, float]:
    tail = vals[-last_m:] if vals else [0.0]
    mean_val = sum(tail) / len(tail)
    var = sum((v - mean_val) ** 2 for v in tail) / max(1, len(tail) - 1)
    ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
    return mean_val, ci_half


def _fmt(val: Any) -> str:
    try:
        return f"{float(val):.6g}"
    except (TypeError, ValueError):
        return "nan"


def _write_report(rows: List[Dict[str, Any]], report_path: Path, command: str) -> None:
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 10 clockwork fabric search v1\n\n")
        fh.write("## Command\n\n")
        fh.write(f"`{command}`\n\n")
        fh.write("## Summary\n\n")
        fh.write(
            "| case | seed | status | control_best | protocol_best | delta | best_key | best_metric |\n"
        )
        fh.write("| --- | ---: | --- | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            fh.write(
                f"| {row['case']} | {row['seed']} | {row['status']} | {_fmt(row['control_best'])} | "
                f"{_fmt(row['protocol_best'])} | {_fmt(row['delta'])} | {row['best_key']} | "
                f"{row['best_metric']} |\n"
            )


def _best_from_frames(
    frames: Dict[str, List[np.ndarray]],
    last_m: int,
    omega_min: float,
    r2_min: float,
) -> Tuple[str, str, float, Dict[str, float]]:
    best_key = ""
    best_metric = ""
    best_score = 0.0
    best_detail: Dict[str, float] = {}
    for key, series in frames.items():
        if len(series) < last_m:
            continue
        arr = np.stack(series[-last_m:], axis=0)
        scores = fabric_scores(arr, omega_min=omega_min)
        metric = "travel" if scores["travel_score"] >= scores["phase_score"] else "phase"
        score = scores["fabric_score"]
        if metric == "travel":
            if abs(scores["travel_omega"]) < omega_min or scores["travel_r2"] < r2_min:
                continue
        if score >= best_score:
            best_score = score
            best_key = key
            best_metric = metric
            best_detail = scores
    return best_key, best_metric, best_score, best_detail


def run_case(
    case: str,
    params: Params,
    seed: int,
    out_dir: Path,
    window_steps: int,
    max_windows: int,
    snapshot_every: int,
    keys: List[str],
    accept_min: float,
    interface_idx: int,
    layer_idx: int,
    omega_min: float,
    r2_min: float,
    last_m: int,
    report_every: int,
    cycle: List[str],
    resume: bool,
    initial_state: Any,
    initial_rng_state: torch.Tensor,
    max_seconds_total: float,
    max_seconds_per_run: float,
    start_total: float,
) -> Dict[str, Any]:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.csv"
    agg_path = case_dir / "agg.csv"
    progress_path = case_dir / "progress.csv"
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = case_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    if resume and agg_path.exists():
        with agg_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            if str(row.get("seed")) == str(seed) and row.get("status") not in {"RUNNING", ""}:
                return row

    if not raw_path.exists():
        with raw_path.open("w", encoding="utf-8") as fh:
            headers = [
                "case",
                "seed",
                "window_index",
                "step",
                "ep_rate",
                "accept_window",
                "mismatch_abs_mean",
                "best_key",
                "best_metric",
                "best_score",
            ]
            for key in keys:
                headers.append(f"fabric_score_{key}")
            fh.write(",".join(headers) + "\n")
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as fh:
            fh.write("case,seed,window_index,step,ep_rate,accept_window\n")

    frames: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    window_ep: List[float] = []
    window_accept: List[float] = []
    window_mismatch: List[float] = []
    window_step_vals: List[int] = []
    window_idx = 0
    status = "RUNNING"
    diag_state = None
    run_start = time.monotonic()
    jsonl_path = jsonl_dir / f"seed{seed}.jsonl"
    jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    progress_handle = progress_path.open("a", encoding="utf-8")

    def report_cb(state, step, ep_ledger, accepted_frac):
        nonlocal window_idx, status, diag_state
        if status != "RUNNING":
            return
        snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
        window_idx += 1

        window_props = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
        accept_window = float(ep_ledger.get("window_accepted", 0)) / window_props if window_props else 0.0
        ep_rate = float(snapshot.get("ep_rate_exact_window", 0.0))
        mismatch_abs = snapshot.get("mismatch_abs_mean")

        maps_dict = compute_spatial_maps(state, keys)
        ok, bad = finite_check(maps_dict)
        if not ok:
            status = f"FAIL_NAN_MAP:{','.join(bad)}"
            return

        for key in keys:
            name, arr = _select_map(maps_dict, key, interface_idx, layer_idx)
            frames[key].append(arr)
        window_ep.append(ep_rate)
        window_accept.append(accept_window)
        window_mismatch.append(float(mismatch_abs) if mismatch_abs is not None else 0.0)
        window_step_vals.append(step)

        if window_idx % snapshot_every == 0:
            npz_payload = {name: arr for name, arr in (_select_map(maps_dict, k, interface_idx, layer_idx) for k in keys)}
            np.savez(npz_dir / f"seed{seed}_win{window_idx:04d}.npz", **npz_payload)

        slim = _slim_snapshot(snapshot)
        slim.update(
            {
                "case": case,
                "seed": seed,
                "window_index": window_idx,
                "ep_rate_exact_window": ep_rate,
                "acceptedFracWindow": accept_window,
            }
        )
        jsonl_handle.write(to_json_line(slim) + "\n")
        jsonl_handle.flush()

        progress_handle.write(f"{case},{seed},{window_idx},{step},{ep_rate},{accept_window}\n")
        progress_handle.flush()

        if accept_window < accept_min:
            status = "FAIL_ACCEPT_COLLAPSE"
        if time.monotonic() - run_start > max_seconds_per_run:
            status = "FAIL_TIME"
        if time.monotonic() - start_total > max_seconds_total:
            status = "FAIL_TIME"
        if window_idx >= max_windows:
            status = "OK"

    def stop_cb(*_args) -> bool:
        return status != "RUNNING"

    run_sim(
        params,
        seed=seed,
        steps=window_steps * max_windows,
        report_every=report_every,
        report_callback=report_cb,
        stop_callback=stop_cb,
        protocol_cycle=cycle,
        initial_state=initial_state,
        initial_rng_state=initial_rng_state,
    )

    jsonl_handle.close()
    progress_handle.close()

    if status == "RUNNING":
        status = "OK"

    window_rows = []
    for idx in range(window_idx):
        row = {
            "case": case,
            "seed": seed,
            "window_index": idx + 1,
            "step": window_step_vals[idx],
            "ep_rate": window_ep[idx],
            "accept_window": window_accept[idx],
            "mismatch_abs_mean": window_mismatch[idx],
        }
        best_score = 0.0
        best_key = ""
        best_metric = ""
        for key in keys:
            series = np.stack(frames[key][: idx + 1], axis=0)
            scores = fabric_scores(series, omega_min=omega_min)
            row[f"fabric_score_{key}"] = scores["fabric_score"]
            if scores["fabric_score"] > best_score:
                best_score = scores["fabric_score"]
                best_key = key
                best_metric = "travel" if scores["travel_score"] >= scores["phase_score"] else "phase"
        row.update({"best_key": best_key, "best_metric": best_metric, "best_score": best_score})
        window_rows.append(row)

    with raw_path.open("a", encoding="utf-8") as fh:
        for row in window_rows:
            values = [
                row["case"],
                row["seed"],
                row["window_index"],
                row["step"],
                row["ep_rate"],
                row["accept_window"],
                row["mismatch_abs_mean"],
                row["best_key"],
                row["best_metric"],
                row["best_score"],
            ]
            for key in keys:
                values.append(row.get(f"fabric_score_{key}", 0.0))
            fh.write(",".join(str(v) for v in values) + "\n")

    best_key, best_metric, best_score, best_detail = _best_from_frames(
        frames, last_m, omega_min, r2_min
    )
    agg = {
        "case": case,
        "seed": seed,
        "status": status,
        "best_key": best_key,
        "best_metric": best_metric,
        "best_score": best_score,
        "accept_mean_last_m": float(np.mean(window_accept[-last_m:])) if window_accept else 0.0,
        "fabric_score": best_detail.get("fabric_score", 0.0),
        "travel_score": best_detail.get("travel_score", 0.0),
        "phase_score": best_detail.get("phase_score", 0.0),
        "travel_omega": best_detail.get("travel_omega", 0.0),
        "travel_r2": best_detail.get("travel_r2", 0.0),
    }
    with agg_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(agg.keys()))
        writer.writeheader()
        writer.writerow(agg)

    return {
        "status": status,
        "frames": frames,
        "accept": window_accept,
        "mismatch": window_mismatch,
        "best_key": best_key,
        "best_metric": best_metric,
        "best_score": best_score,
        "best_detail": best_detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 clockwork fabric search v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", default="scripts/params/meta_null_coupled_eta1.00_layers3.json")
    parser.add_argument("--out-dir", default=".tmp/phase10_clockwork_v1")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--max-windows", type=int, default=60)
    parser.add_argument("--last-m", type=int, default=20)
    parser.add_argument("--snapshot-every-windows", type=int, default=1)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--match-control-cycle-weights", action="store_true")
    parser.add_argument("--analysis-keys", default="k_axis_bias,k_entropy,sigma,mismatch")
    parser.add_argument("--analysis-interface", type=int, default=0)
    parser.add_argument("--analysis-layer", type=int, default=0)
    parser.add_argument("--fabric-min", type=float, default=0.05)
    parser.add_argument("--delta-min", type=float, default=0.02)
    parser.add_argument("--r2-min", type=float, default=0.60)
    parser.add_argument("--omega-min", type=float, default=0.10)
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=1800)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preset = _load_preset(Path(args.preset))
    base_overrides: Dict[str, Any] = {"device": args.device}
    base_params = _as_params(preset, base_overrides)

    if len(base_params.shape) != 2:
        raise ValueError("Phase 10 requires a 2D lattice shape")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    keys = _parse_keys(args.analysis_keys)
    cycle = _cycle_list()

    start_total = time.monotonic()
    report_rows: List[Dict[str, Any]] = []
    command = " ".join([str(x) for x in sys.argv])

    for seed in seeds:
        if time.monotonic() - start_total > args.max_seconds_total:
            break
        N = math.prod(base_params.shape)
        expected_props = _expected_proposals_per_step(N, str(base_params.device), base_params.kernel_weights)
        burn_steps = int(math.ceil(args.burn_in_sweeps * N / expected_props))
        window_steps = int(math.ceil(args.window_sweeps * N / expected_props))

        burn_summary = run_sim(base_params, seed=seed, steps=burn_steps, report_every=burn_steps, return_state=True)
        state0 = burn_summary.get("state")
        rng_state = burn_summary.get("rng_state")
        if state0 is None or rng_state is None:
            raise RuntimeError("Failed to capture burn-in state")

        control_params = Params.from_dict(base_params, {"p3_on": False})
        if args.match_control_cycle_weights:
            matched = _match_cycle_weights(control_params.kernel_weights, cycle)
            control_params = Params.from_dict(control_params, {"kernel_weights": matched})
        protocol_params = Params.from_dict(base_params, {"p3_on": True})

        control = run_case(
            "control_p3_matched",
            control_params,
            seed,
            out_dir,
            window_steps,
            args.max_windows,
            args.snapshot_every_windows,
            keys,
            args.accept_min,
            args.analysis_interface,
            args.analysis_layer,
            args.omega_min,
            args.r2_min,
            args.last_m,
            window_steps,
            cycle,
            args.resume,
            initial_state=_clone_state(state0),
            initial_rng_state=rng_state.clone(),
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
        )
        if control["status"] != "OK":
            report_rows.append(
                {
                    "case": "control_p3_matched",
                    "seed": seed,
                    "status": control["status"],
                    "control_best": 0.0,
                    "protocol_best": 0.0,
                    "delta": 0.0,
                    "best_key": "",
                    "best_metric": "",
                }
            )
            _write_report(report_rows, out_dir / "PHASE10_CLOCKWORK_REPORT.md", command)
            break

        protocol = run_case(
            "protocol_p3_on",
            protocol_params,
            seed,
            out_dir,
            window_steps,
            args.max_windows,
            args.snapshot_every_windows,
            keys,
            args.accept_min,
            args.analysis_interface,
            args.analysis_layer,
            args.omega_min,
            args.r2_min,
            args.last_m,
            window_steps,
            cycle,
            args.resume,
            initial_state=_clone_state(state0),
            initial_rng_state=rng_state.clone(),
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            start_total=start_total,
        )

        control_best = float(control["best_score"])
        protocol_best = float(protocol["best_score"])
        delta = protocol_best - control_best

        status = "PASS" if (protocol_best >= args.fabric_min and delta >= args.delta_min) else "FAIL"
        report_rows.append(
            {
                "case": "protocol_p3_on",
                "seed": seed,
                "status": status,
                "control_best": control_best,
                "protocol_best": protocol_best,
                "delta": delta,
                "best_key": protocol["best_key"],
                "best_metric": protocol["best_metric"],
            }
        )

        _write_report(report_rows, out_dir / "PHASE10_CLOCKWORK_REPORT.md", command)

        if args.progress:
            print(
                f"PHASE10_SEED={seed} control_best={control_best:.6g} "
                f"protocol_best={protocol_best:.6g} delta={delta:.6g} status={status}"
            )

        if status != "PASS":
            break

    if args.progress:
        print(f"PHASE10_REPORT={out_dir / 'PHASE10_CLOCKWORK_REPORT.md'}")


if __name__ == "__main__":
    main()
