#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, TextIO

from ratchet_gpu.lattice import Lattice, generate_stencil
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
}


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [int(item) for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must be comma-separated ints")
    return tuple(parts)


def _parse_list(value: str, cast=float) -> list[Any]:
    if not value:
        return []
    return [cast(item) for item in value.split(",") if str(item).strip()]

ACCEPT_MIN = 1e-4


def _ci_half(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, float("inf")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, float("inf")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    df = n - 1
    tcrit = _T_CRIT_95.get(df, 1.96)
    return mean, std, tcrit * se


def _rate_micro(delta_total: float, window_steps: int) -> float:
    return delta_total / max(1, window_steps)


def _kernel_weights(w_neighbor_weight: float) -> dict[str, float]:
    weights = {name: 0.0 for name in Params(shape=(1,), layers=2).kernel_weights}
    weights["spin_flip_color0"] = 1.0
    weights["spin_flip_color1"] = 1.0
    weights["w_local"] = 1.0
    weights["w_neighbor"] = w_neighbor_weight
    return weights


def _build_params(
    shape: tuple[int, ...],
    layers: int,
    beta: float,
    J: float,
    radius_w: int,
    w_fill: float,
    l_w: int,
    w_neighbor_weight: float,
    device: str,
) -> tuple[Params, int, int]:
    lattice = Lattice(shape)
    R_W = generate_stencil(
        d=lattice.d,
        policy="l1_ball_odd",
        radius=radius_w,
        bipartite=True,
        shape=shape,
    )
    K_W = int(R_W.shape[0])
    capacity = l_w * layers * lattice.N * K_W
    B_w = int(round(w_fill * capacity))

    params = Params(
        shape=shape,
        layers=layers,
        p3_on=False,
        p6_on=False,
        beta=beta,
        J=J,
        kappa_T=1.0,
        eta=0.0,
        eta_drive=0.0,
        l_s=0,
        l_w=l_w,
        l_k=1,
        B_w=B_w,
        B_k=0,
        radius_w=radius_w,
        radius_k=0,
        stencil_policy_w="l1_ball_odd",
        stencil_policy_k="l1_ball_even",
        kernel_weights=_kernel_weights(w_neighbor_weight),
        report_every=1000,
        device=device,
    )
    return params, K_W, B_w


def _run_early_stop(
    params: Params,
    seed: int,
    burn_in_steps: int,
    window_steps: int,
    min_windows: int,
    max_windows: int,
    last_m: int,
    mean_thresh: float,
    ci_thresh: float,
    fail_fast_mean: bool,
    fail_fast_ci: bool,
    out_path: Path,
    progress_label: str | None,
    progress_handle: TextIO | None,
    run_start_time: float,
    max_seconds_per_run: float,
) -> dict[str, Any]:
    prev_total = 0.0
    prev_step = 0
    window_rates: list[float] = []
    window_accepts: list[float] = []
    ci_history: list[float] = []
    best_ci_half = float("inf")
    stop_flag = {"value": False}
    status = "FAIL_MAX_WINDOWS"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", encoding="utf-8")

    def _report_callback(state, step, ep_ledger, accepted_frac):
        nonlocal prev_total, prev_step, best_ci_half, status
        if len(window_rates) >= max_windows:
            status = "FAIL_MAX_WINDOWS"
            stop_flag["value"] = True
            return
        total = float(ep_ledger.get("ep_total_exact", 0.0))
        window_steps = int(ep_ledger.get("window_steps", 0))
        delta_total = total - prev_total
        rate_micro = _rate_micro(delta_total, window_steps)
        prev_total = total
        prev_step = step
        if step <= burn_in_steps:
            return

        window_rates.append(rate_micro)
        accept_window = float(ep_ledger.get("window_accept_frac", accepted_frac))
        window_accepts.append(accept_window)

        tail = window_rates[-last_m:]
        mean_ep, std_ep, ci_half = _ci_half(tail)
        best_ci_half = min(best_ci_half, ci_half)
        ci_history.append(ci_half)

        pass_now = False
        if len(window_rates) >= min_windows:
            pass_now = (
                abs(mean_ep) <= mean_thresh
                and ci_half <= ci_thresh
                and accept_window >= ACCEPT_MIN
            )
            if pass_now:
                stop_flag["value"] = True
                status = "PASS_EARLY"
            elif (
                accept_window < ACCEPT_MIN
                and len(window_accepts) >= 2
                and window_accepts[-2] < ACCEPT_MIN
            ):
                stop_flag["value"] = True
                status = "FAIL_ACCEPT_LOW"
            elif fail_fast_mean and ci_half <= ci_thresh and abs(mean_ep) > mean_thresh:
                stop_flag["value"] = True
                status = "FAIL_MEAN_EARLY"
            elif (
                fail_fast_ci
                and len(window_rates) >= min_windows + 3
                and len(ci_history) >= 3
                and ci_half > 2 * ci_thresh
                and ci_half >= 0.9 * ci_history[-3]
            ):
                stop_flag["value"] = True
                status = "FAIL_CI_STUCK"

        record = {
            "step": int(step),
            "window_index": len(window_rates),
            "ep_rate_exact_window": rate_micro,
            "mean_ep_last_m": mean_ep,
            "ci_half": ci_half,
            "acceptedFracWindow": accept_window,
            "window_steps": window_steps,
            "window_proposals": int(ep_ledger.get("window_proposals", 0)),
            "window_accepted": int(ep_ledger.get("window_accepted", 0)),
            "pass": pass_now,
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        if progress_handle is not None:
            progress_row = {
                "timestamp": time.time(),
                "config_id": progress_label or "",
                "seed": seed,
                "step": int(step),
                "window_index": len(window_rates),
                "ep_rate_exact_window": rate_micro,
                "mean_ep_last_m": mean_ep,
                "ci_half": ci_half,
                "acceptedFracWindow": accept_window,
                "window_steps": window_steps,
                "window_proposals": int(ep_ledger.get("window_proposals", 0)),
                "window_accepted": int(ep_ledger.get("window_accepted", 0)),
                "pass": pass_now,
            }
            progress_handle.write(
                ",".join(str(progress_row[k]) for k in progress_row) + "\n"
            )
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
        if progress_label:
            percent = 100.0 * len(window_rates) / max_windows
            print(
                f"[{progress_label}] window {len(window_rates)}/{max_windows} "
                f"({percent:.1f}%)",
                flush=True,
        )
        if (
            not stop_flag["value"]
            and not pass_now
            and time.monotonic() - run_start_time > max_seconds_per_run
        ):
            status = "FAIL_TIME"
            stop_flag["value"] = True

    def _stop_callback(state, step, ep_ledger, accepted_frac):
        return stop_flag["value"]

    max_steps = burn_in_steps + max_windows * window_steps
    run_sim(
        params,
        seed=seed,
        steps=max_steps,
        report_every=window_steps,
        device=params.device,
        report_callback=_report_callback,
        stop_callback=_stop_callback,
    )
    handle.close()

    tail = window_rates[-last_m:] if window_rates else []
    mean_ep, std_ep, ci_half = _ci_half(tail)
    accepted_last = window_accepts[-1] if window_accepts else 0.0
    pass_final = (
        len(window_rates) >= min_windows
        and abs(mean_ep) <= mean_thresh
        and ci_half <= ci_thresh
        and accepted_last >= ACCEPT_MIN
    )
    if status == "FAIL_MAX_WINDOWS" and pass_final:
        status = "PASS_EARLY"

    return {
        "pass": pass_final,
        "windows_used": len(window_rates),
        "mean_ep": mean_ep,
        "ci_half": ci_half,
        "acceptedFracWindow": accepted_last,
        "best_ci_half": best_ci_half,
        "status": status,
    }


def _config_id(beta: float, J: float, w_fill: float, w_neighbor_weight: float) -> str:
    wf_int = int(round(w_fill * 1000))
    wn_int = int(round(w_neighbor_weight * 100))
    return f"rw3_b{beta:.2f}_J{J:.2f}_wf{wf_int:03d}_wn{wn_int:03d}"


def _expected_proposals_per_step(
    N: int, device: str, kernel_weights: dict[str, float]
) -> float:
    weights = {k: max(0.0, float(v)) for k, v in kernel_weights.items()}
    if all(v == 0.0 for v in weights.values()):
        weights = {k: 1.0 for k in weights}
    total = sum(weights.values())
    exp = 0.0
    for name, w in weights.items():
        proposals = 1.0
        if name == "w_neighbor" and device == "cuda":
            proposals = N / 2.0
        exp += (w / total) * proposals
    return max(1.0, exp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 null screen v4 (early stop)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--shape", type=_parse_shape, default=(24, 24))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--betas", default="0.25,0.5")
    parser.add_argument("--Js", default="0.5,1.0")
    parser.add_argument("--w-fills", default="0.03,0.05")
    parser.add_argument("--radius-w", type=int, default=3)
    parser.add_argument("--w-neighbor-weights", default="0.25,1.0")
    parser.add_argument("--l-w", type=int, default=4)
    parser.add_argument("--burn-in-sweeps", type=float, default=200.0)
    parser.add_argument("--window-sweeps", type=float, default=100.0)
    parser.add_argument("--min-windows", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--out-dir", default=".tmp/phase1_null_v4")
    parser.add_argument("--mean-thresh", type=float, default=3e-4)
    parser.add_argument("--ci-thresh", type=float, default=8e-4)
    parser.add_argument("--max-seconds-total", type=float, default=5400.0)
    parser.add_argument("--max-seconds-per-run", type=float, default=1200.0)
    parser.add_argument(
        "--progress-csv",
        default="",
        help="Path to append window progress (default: <out-dir>/screen_progress.csv)",
    )
    parser.add_argument(
        "--preset-out",
        default="scripts/params/phase1_null_balanced_quick_v2.json",
        help="Path to write selected preset JSON.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-window progress for each config/seed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have a terminal status recorded in screen_raw.csv",
    )
    parser.add_argument(
        "--fail-fast-mean",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable early failure on mean threshold (default: 1)",
    )
    parser.add_argument(
        "--fail-fast-ci",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable early failure on CI stuck logic (default: 1)",
    )
    args = parser.parse_args()

    betas = _parse_list(args.betas, float)
    Js = _parse_list(args.Js, float)
    w_fills = _parse_list(args.w_fills, float)
    w_neighbor_weights = _parse_list(args.w_neighbor_weights, float)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    total_runs = (
        len(betas)
        * len(Js)
        * len(w_fills)
        * len(w_neighbor_weights)
        * len(seeds)
    )
    completed_runs = 0

    N = math.prod(args.shape)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "screen_raw.csv"
    agg_path = out_dir / "screen_agg.csv"
    jsonl_dir = out_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    progress_path = (
        Path(args.progress_csv)
        if args.progress_csv
        else out_dir / "screen_progress.csv"
    )
    start_time = time.monotonic()

    raw_fields = [
        "config_id",
        "seed",
        "beta",
        "J",
        "w_fill",
        "w_neighbor_weight",
        "radius_w",
        "K_W",
        "l_w",
        "B_w",
        "burn_in_sweeps",
        "window_sweeps",
        "min_windows",
        "max_windows",
        "last_m",
        "windows_used",
        "mean_ep",
        "ci_half",
        "acceptedFracWindow",
        "best_ci_half",
        "seconds_used",
        "status",
        "pass",
    ]

    per_config: dict[str, list[dict[str, Any]]] = {}
    completed: set[tuple[str, int]] = set()
    if args.resume and raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            for row in reader:
                per_config.setdefault(row["config_id"], []).append(row)
                if row.get("status") in {
                    "PASS_EARLY",
                    "FAIL_MEAN_EARLY",
                    "FAIL_CI_STUCK",
                    "FAIL_ACCEPT_LOW",
                    "FAIL_BUDGET",
                    "FAIL_TIME",
                    "FAIL_MAX_WINDOWS",
                }:
                    completed.add((row["config_id"], int(row["seed"])))
        completed_runs = len(completed)

    mode = "a" if args.resume and raw_path.exists() else "w"
    write_header = True
    if raw_path.exists() and raw_path.stat().st_size > 0 and mode == "a":
        write_header = False

    with raw_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        if write_header:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        for beta in betas:
            for J in Js:
                for w_fill in w_fills:
                    for w_neighbor_weight in w_neighbor_weights:
                        config_id = _config_id(beta, J, w_fill, w_neighbor_weight)
                        params, K_W, B_w = _build_params(
                            shape=args.shape,
                            layers=args.layers,
                            beta=beta,
                            J=J,
                            radius_w=args.radius_w,
                            w_fill=w_fill,
                            l_w=args.l_w,
                            w_neighbor_weight=w_neighbor_weight,
                            device=args.device,
                        )
                        expected_props = _expected_proposals_per_step(
                            N, args.device, params.kernel_weights
                        )
                        burn_in_steps = int(
                            math.ceil(args.burn_in_sweeps * N / expected_props)
                        )
                        window_steps = int(
                            math.ceil(args.window_sweeps * N / expected_props)
                        )
                        for seed in seeds:
                            if time.monotonic() - start_time > args.max_seconds_total:
                                print("Total time budget exceeded; stopping.")
                                break
                            if (config_id, seed) in completed:
                                completed_runs += 1
                                print(
                                    f"SKIP {completed_runs}/{total_runs} {config_id} seed={seed} (resume)",
                                    flush=True,
                                )
                                continue
                            out_path = jsonl_dir / f"{config_id}_seed{seed}.jsonl"
                            label = None
                            if args.progress:
                                label = f"{config_id} seed={seed}"
                            progress_handle = progress_path.open("a", encoding="utf-8")
                            if progress_path.stat().st_size == 0:
                                progress_handle.write(
                                    "timestamp,config_id,seed,step,window_index,"
                                    "ep_rate_exact_window,mean_ep_last_m,ci_half,"
                                    "acceptedFracWindow,window_steps,window_proposals,"
                                    "window_accepted,pass\n"
                                )
                                progress_handle.flush()
                                os.fsync(progress_handle.fileno())
                            run_start = time.monotonic()
                            try:
                                result = _run_early_stop(
                                    params=params,
                                    seed=seed,
                                    burn_in_steps=burn_in_steps,
                                    window_steps=window_steps,
                                    min_windows=args.min_windows,
                                    max_windows=args.max_windows,
                                    last_m=args.last_m,
                                    mean_thresh=args.mean_thresh,
                                    ci_thresh=args.ci_thresh,
                                    fail_fast_mean=bool(args.fail_fast_mean),
                                    fail_fast_ci=bool(args.fail_fast_ci),
                                    out_path=out_path,
                                    progress_label=label,
                                    progress_handle=progress_handle,
                                    run_start_time=run_start,
                                    max_seconds_per_run=args.max_seconds_per_run,
                                )
                            except Exception as exc:
                                print(f"ERROR during run {config_id} seed={seed}: {exc}")
                                result = {
                                    "pass": False,
                                    "windows_used": 0,
                                    "mean_ep": 0.0,
                                    "ci_half": float("inf"),
                                    "acceptedFracWindow": 0.0,
                                    "best_ci_half": float("inf"),
                                    "status": "ERROR",
                                }
                            seconds_used = time.monotonic() - run_start
                            progress_handle.close()
                            row = {
                                "config_id": config_id,
                                "seed": seed,
                                "beta": beta,
                                "J": J,
                                "w_fill": w_fill,
                                "w_neighbor_weight": w_neighbor_weight,
                                "radius_w": args.radius_w,
                                "K_W": K_W,
                                "l_w": args.l_w,
                                "B_w": B_w,
                                "burn_in_sweeps": args.burn_in_sweeps,
                                "window_sweeps": args.window_sweeps,
                                "min_windows": args.min_windows,
                                "max_windows": args.max_windows,
                                "last_m": args.last_m,
                                "windows_used": result["windows_used"],
                                "mean_ep": result["mean_ep"],
                                "ci_half": result["ci_half"],
                                "acceptedFracWindow": result["acceptedFracWindow"],
                                "best_ci_half": result["best_ci_half"],
                                "seconds_used": round(seconds_used, 3),
                                "status": result["status"],
                                "pass": str(result["pass"]).lower(),
                            }
                            writer.writerow(row)
                            handle.flush()
                            os.fsync(handle.fileno())
                            per_config.setdefault(config_id, []).append(row)
                            completed_runs += 1
                            print(
                                f"SUMMARY seed={seed} status={row['status']} "
                                f"windows={row['windows_used']} "
                                f"lastm_mean={row['mean_ep']} "
                                f"lastm_ci={row['ci_half']} "
                                f"lastm_accept={row['acceptedFracWindow']}"
                            )
                            print(
                                f"COMPLETED {completed_runs}/{total_runs} "
                                f"{config_id} seed={seed} status={row['status']}",
                                flush=True,
                            )
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break

    agg_fields = [
        "config_id",
        "beta",
        "J",
        "w_fill",
        "w_neighbor_weight",
        "pass_count",
        "total",
        "pass_rate",
        "ci_half_mean",
        "windows_used_mean",
        "acceptedFracWindowMean",
        "ep_mean_mean",
    ]

    summaries = []
    with agg_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fields)
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        for config_id, rows in per_config.items():
            pass_count = sum(1 for r in rows if r["pass"] == "true")
            total = len(rows)
            ci_half_mean = sum(float(r["ci_half"]) for r in rows) / total
            windows_used_mean = sum(float(r["windows_used"]) for r in rows) / total
            accept_mean = sum(float(r["acceptedFracWindow"]) for r in rows) / total
            ep_mean_mean = sum(float(r["mean_ep"]) for r in rows) / total
            entry = {
                "config_id": config_id,
                "beta": rows[0]["beta"],
                "J": rows[0]["J"],
                "w_fill": rows[0]["w_fill"],
                "w_neighbor_weight": rows[0]["w_neighbor_weight"],
                "pass_count": pass_count,
                "total": total,
                "pass_rate": pass_count / total,
                "ci_half_mean": ci_half_mean,
                "windows_used_mean": windows_used_mean,
                "acceptedFracWindowMean": accept_mean,
                "ep_mean_mean": ep_mean_mean,
            }
            writer.writerow(entry)
            handle.flush()
            os.fsync(handle.fileno())
            summaries.append(entry)

    if not summaries:
        print("No configs to summarize.")
        return

    pass_candidates = [entry for entry in summaries if entry["pass_rate"] > 0]
    if pass_candidates:
        pass_candidates.sort(
            key=lambda e: (
                -e["pass_rate"],
                e["ci_half_mean"],
                e["windows_used_mean"],
                abs(e["acceptedFracWindowMean"] - 0.35),
            )
        )
        best = pass_candidates[0]
        preset_reason = "PASS_FILTER"
        preset_pass = True
    else:
        def _fallback_key(entry: dict[str, Any]) -> tuple[float, float, float, float]:
            return (
                abs(entry["ep_mean_mean"]),
                entry["ci_half_mean"],
                abs(entry["acceptedFracWindowMean"] - 0.35),
                entry["windows_used_mean"],
            )

        summaries.sort(key=_fallback_key)
        best = summaries[0]
        preset_reason = "NO_PASS_BEST_CANDIDATE"
        preset_pass = False

    params, K_W, B_w = _build_params(
        shape=args.shape,
        layers=args.layers,
        beta=float(best["beta"]),
        J=float(best["J"]),
        radius_w=args.radius_w,
        w_fill=float(best["w_fill"]),
        l_w=args.l_w,
        w_neighbor_weight=float(best["w_neighbor_weight"]),
        device=args.device,
    )

    preset = {
        "config_id": best["config_id"],
        "shape": list(args.shape),
        "layers": args.layers,
        "p3_on": False,
        "p6_on": False,
        "beta": params.beta,
        "J": params.J,
        "eta": 0.0,
        "eta_drive": 0.0,
        "radius_w": args.radius_w,
        "radius_k": 0,
        "l_w": args.l_w,
        "l_k": 1,
        "l_s": 0,
        "B_w": B_w,
        "B_k": 0,
        "w_neighbor_weight": float(best["w_neighbor_weight"]),
        "kernel_weights": params.kernel_weights,
        "pass": preset_pass,
    }
    if not preset_pass:
        preset["note"] = "NO_PASS_BEST_CANDIDATE"

    params_dir = Path("scripts/params")
    params_dir.mkdir(parents=True, exist_ok=True)
    preset_path = Path(args.preset_out)
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    with preset_path.open("w", encoding="utf-8") as handle:
        json.dump(preset, handle, indent=2, sort_keys=True)

    print(
        f"PRESET_SELECTED config_id={best['config_id']} "
        f"pass={'true' if preset_pass else 'false'} reason={preset_reason}"
    )

    # Emit status counts
    from collections import Counter

    all_rows = [r for rows in per_config.values() for r in rows]
    status_counts = Counter(r["status"] for r in all_rows)
    print(f"STATUS COUNTS: {dict(status_counts)}")

    # Duplicate files under validate_* names for downstream consumers expecting that naming.
    validate_raw = out_dir / "validate_raw.csv"
    validate_agg = out_dir / "validate_agg.csv"
    validate_progress = out_dir / "validate_progress.csv"
    for src, dst in [
        (raw_path, validate_raw),
        (agg_path, validate_agg),
        (progress_path, validate_progress),
    ]:
        try:
            dst.write_bytes(src.read_bytes())
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
