#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.lattice import Lattice, generate_stencil
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim


def _parse_list(value: str, cast=float) -> List[Any]:
    if not value:
        return []
    return [cast(item) for item in value.split(",") if str(item).strip()]


def _parse_shape(value: str) -> Tuple[int, ...]:
    parts = [int(item) for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must be comma-separated ints")
    return tuple(parts)


def _ensure_dirs(base: Path) -> None:
    (base / "jsonl").mkdir(parents=True, exist_ok=True)


def _w_metrics(state: Any) -> Dict[str, float]:
    W = state.W.to(dtype=torch.float64)
    l_w = state.params.l_w

    w_zero_frac = float((W == 0).float().mean().item())
    w_cap_frac = float((W.mean().item()) / float(l_w))

    sum_w = W.sum(dim=-1)
    active_deg = (W > 0).sum(dim=-1).to(dtype=torch.float64)
    w_active_deg_mean = float(active_deg.mean().item())

    eps = 1e-12
    denom = sum_w.unsqueeze(-1)
    probs = torch.where(denom > 0, W / denom, torch.zeros_like(W))
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    w_entropy_mean = float(entropy.mean().item())

    r2 = (state.R_W.to(dtype=torch.float64) ** 2).sum(dim=-1)
    r2_vals = (probs * r2).sum(dim=-1)
    w_r2_mean = float(r2_vals.mean().item())

    sigma = state.sigma.to(dtype=torch.float64)
    align_vals = []
    for layer in range(state.layers):
        sigma_layer = sigma[layer]
        neighbors = torch.zeros((state.N, state.K_W), device=state.device, dtype=torch.float64)
        if state.K_W > 0:
            neighbors = torch.stack(
                [
                    torch.roll(sigma_layer.view(*state.lattice.shape),
                               shifts=tuple(-int(s) for s in offset),
                               dims=tuple(range(state.lattice.d)))
                    .reshape(state.N)
                    for offset in state.R_W.tolist()
                ],
                dim=1,
            )
        numerator = (W[layer] * sigma_layer[:, None] * neighbors).sum(dim=-1)
        align = torch.where(sum_w[layer] > 0, numerator / sum_w[layer], torch.zeros_like(numerator))
        align_vals.append(align)
    w_align_mean = float(torch.stack(align_vals).mean().item())

    return {
        "w_zero_frac": w_zero_frac,
        "w_cap_frac": w_cap_frac,
        "w_active_deg_mean": w_active_deg_mean,
        "w_entropy_mean": w_entropy_mean,
        "w_r2_mean": w_r2_mean,
        "w_align_mean": w_align_mean,
    }


def _mag_abs(state: Any) -> float:
    sigma = state.sigma.to(dtype=torch.float64)
    return float(sigma.mean().abs().item())


def _compute_k_w(shape: Tuple[int, ...], radius_w: int, policy: str) -> int:
    lattice = Lattice(shape)
    offsets = generate_stencil(
        d=lattice.d,
        policy=policy,
        radius=radius_w,
        bipartite=True,
        shape=shape,
    )
    return int(offsets.shape[0])


def _mean_ci(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    ci = 1.96 * math.sqrt(var) / math.sqrt(n)
    return mean, ci


def _config_id(
    radius_w: int, beta: float, J: float, w_fill: float, w_neighbor_weight: float
) -> str:
    return (
        f"rw{radius_w}_b{beta:.2f}_J{J:.2f}_wf{w_fill:.2f}"
        f"_wn{w_neighbor_weight:.2f}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _autocorr_lag1(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    denom = sum((v - mean) ** 2 for v in values)
    if denom == 0.0:
        return 0.0
    num = sum((values[i] - mean) * (values[i - 1] - mean) for i in range(1, n))
    return num / denom


def _ess(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    rho1 = _autocorr_lag1(values)
    if rho1 is None:
        return None
    rho1 = max(-0.99, min(0.99, rho1))
    return n * (1.0 - rho1) / (1.0 + rho1)


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
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _ci_half_95(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, float("inf")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, float("inf")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    df = n - 1
    tcrit = _T_CRIT_95.get(df, 1.96)
    ci_half = tcrit * se
    return mean, se, ci_half


def _run_config(
    base_params: Params,
    config_overrides: Dict[str, Any],
    config_id: str,
    seed: int,
    out_dir: Path,
    steps: int,
    report_every: int,
    device: str,
    burn_in_steps: int,
    resume: bool,
    progress: bool,
) -> Dict[str, Any]:
    params = Params.from_dict(base_params, config_overrides)
    jsonl_path = out_dir / "jsonl" / f"{config_id}_seed{seed}.jsonl"

    diag_state = None
    last_snapshot = None
    ep_rates = []

    records = []
    if resume and jsonl_path.exists():
        records = _read_jsonl(jsonl_path)
        if records and records[-1].get("step", 0) >= steps:
            for record in records:
                step = int(record.get("step", 0))
                if step > burn_in_steps:
                    ep_rates.append(record.get("ep_rate_exact_window", 0.0))
            last_snapshot = records[-1]
        else:
            records = []

    if not records:
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            def _report_callback(state, step, ep_ledger, accepted_frac):
                nonlocal diag_state, last_snapshot
                snapshot, diag_state = compute_snapshot(
                    state, step, ep_ledger, accepted_frac, diag_state
                )
                if step > burn_in_steps:
                    ep_rates.append(snapshot.get("ep_rate_exact_window", 0.0))
                snapshot.update(_w_metrics(state))
                snapshot["mag_abs"] = _mag_abs(state)
                last_snapshot = snapshot
                handle.write(to_json_line(snapshot) + "\n")
                if progress:
                    percent = 100.0 * float(step) / float(max(1, steps))
                    print(
                        f"[{config_id} seed={seed}] step {step}/{steps} ({percent:.1f}%)",
                        flush=True,
                    )

            summary = run_sim(
                params,
                seed=seed,
                steps=steps,
                report_every=report_every,
                device=device,
                report_callback=_report_callback,
            )

        if last_snapshot is None:
            last_snapshot = {
                "step": steps,
                "ep_rate_exact_window": summary.get("epMicroRateWindowLast", 0.0),
                "acceptedFrac": summary.get("acceptedFrac", 0.0),
            }
        last_snapshot["epExactRateWindowLast"] = summary.get("epMicroRateWindowLast", 0.0)
        last_snapshot["acceptedFrac"] = summary.get("acceptedFrac", 0.0)
        for key, value in summary.items():
            if key.startswith("proposalsWindow_") or key.startswith("acceptedWindow_") or key.startswith("acceptWindow_"):
                last_snapshot[key] = value

    if last_snapshot is None:
        last_snapshot = {
            "step": steps,
            "ep_rate_exact_window": 0.0,
            "acceptedFrac": 0.0,
        }
    last_snapshot["ep_rate_series"] = ep_rates

    return last_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 null calibration sweep")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--shape", type=_parse_shape, default=(24, 24))
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--report-every", type=int, default=10000)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--radius-ws", default="1,3")
    parser.add_argument("--betas", default="0.5,1.0")
    parser.add_argument("--w-fills", default="0.10,0.25,0.40")
    parser.add_argument("--l-w", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--Js", default="")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--stencil-policy-w", default="l1_ball_odd")
    parser.add_argument("--w-neighbor-weight", type=float, default=1.0)
    parser.add_argument("--w-neighbor-weights", default="")
    parser.add_argument("--out-dir", default=".tmp/phase1_null")
    parser.add_argument("--burn-in-sweeps", type=float, default=0.0)
    parser.add_argument("--measure-sweeps", type=float, default=0.0)
    parser.add_argument("--window-sweeps", type=float, default=50.0)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print percent progress at each report interval.",
    )

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    radius_ws = _parse_list(args.radius_ws, int)
    betas = _parse_list(args.betas, float)
    w_fills = _parse_list(args.w_fills, float)
    Js = _parse_list(args.Js, float) if args.Js else [float(args.J)]
    w_neighbor_weights = (
        _parse_list(args.w_neighbor_weights, float)
        if args.w_neighbor_weights
        else [float(args.w_neighbor_weight)]
    )

    out_dir = Path(args.out_dir)
    _ensure_dirs(out_dir)
    if args.resume and (out_dir / "jsonl").exists():
        available = set()
        for path in (out_dir / "jsonl").glob("*_seed*.jsonl"):
            try:
                _, seed_part = path.stem.rsplit("_seed", 1)
                available.add(int(seed_part))
            except ValueError:
                continue
        if available:
            seeds = sorted(available)
    N_sites = math.prod(args.shape)

    burn_in_steps = int(args.burn_in_sweeps * N_sites)
    measure_steps = int(args.measure_sweeps * N_sites)
    if args.burn_in_sweeps or args.measure_sweeps:
        if args.measure_sweeps <= 0:
            raise ValueError("measure-sweeps must be > 0 when using sweeps mode")
        args.steps = burn_in_steps + measure_steps
        args.report_every = max(1, int(args.window_sweeps * N_sites))

    default_params = Params(shape=args.shape, layers=args.layers)
    base_kernel_weights = {name: 0.0 for name in default_params.kernel_weights}
    for name in base_kernel_weights:
        if name.startswith("spin_flip_color"):
            base_kernel_weights[name] = 1.0
    for name in ("w_local", "w_neighbor"):
        if name in base_kernel_weights:
            base_kernel_weights[name] = 1.0
    if "w_neighbor" in base_kernel_weights:
        base_kernel_weights["w_neighbor"] = 1.0
    kernel_names = sorted(base_kernel_weights.keys())

    base_params = Params(
        shape=args.shape,
        layers=args.layers,
        p3_on=False,
        p6_on=False,
        beta=1.0,
        J=args.J,
        eta=0.0,
        eta_drive=0.0,
        l_w=args.l_w,
        l_k=1,
        l_s=0,
        B_w=0,
        B_k=0,
        radius_w=1,
        radius_k=0,
        stencil_policy_w=args.stencil_policy_w,
        stencil_policy_k="l1_ball_even",
        kernel_weights=base_kernel_weights,
        report_every=args.report_every,
        device=args.device,
    )

    runs_path = out_dir / "phase1_null_runs.csv"
    agg_path = out_dir / "phase1_null_agg.csv"

    run_fields = [
        "config_id",
        "seed",
        "device",
        "shape",
        "meta_layers",
        "beta",
        "J",
        "radius_w",
        "K_W",
        "l_w",
        "B_w",
        "w_fill",
        "w_neighbor_weight",
        "steps",
        "burn_in_sweeps",
        "measure_sweeps",
        "window_sweeps",
        "last_m",
        "epExactRateWindowLast",
        "epLastMMean",
        "epLastMStd",
        "epLastMN",
        "epAutocorrLag1",
        "epEss",
        "acceptedFrac",
        "mag_abs",
        "w_zero_frac",
        "w_cap_frac",
        "w_active_deg_mean",
        "w_entropy_mean",
        "w_r2_mean",
        "w_align_mean",
    ]
    for name in kernel_names:
        run_fields.append(f"proposalsWindow_{name}")
        run_fields.append(f"acceptedWindow_{name}")
        run_fields.append(f"acceptWindow_{name}")

    with open(runs_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()

        per_config: Dict[str, List[Dict[str, Any]]] = {}

        for radius_w in radius_ws:
            K_W = _compute_k_w(args.shape, radius_w, args.stencil_policy_w)
            for beta in betas:
                for J_val in Js:
                    for w_fill in w_fills:
                        for w_neighbor_weight in w_neighbor_weights:
                            config_id = _config_id(
                                radius_w, beta, J_val, w_fill, w_neighbor_weight
                            )
                            capacity = args.l_w * args.layers * math.prod(args.shape) * K_W
                            B_w = int(round(w_fill * capacity))
                            kernel_weights = dict(base_kernel_weights)
                            if "w_neighbor" in kernel_weights:
                                kernel_weights["w_neighbor"] = w_neighbor_weight

                            overrides = {
                                "beta": beta,
                                "J": J_val,
                                "radius_w": radius_w,
                                "B_w": B_w,
                                "kernel_weights": kernel_weights,
                            }

                            if args.resume:
                                seed_paths = sorted((out_dir / "jsonl").glob(f"{config_id}_seed*.jsonl"))
                                config_seeds = []
                                for path in seed_paths:
                                    try:
                                        _, seed_part = path.stem.rsplit("_seed", 1)
                                        config_seeds.append(int(seed_part))
                                    except ValueError:
                                        continue
                                if config_seeds:
                                    config_seeds = sorted(set(config_seeds))
                                else:
                                    config_seeds = list(seeds)
                            else:
                                config_seeds = list(seeds)

                            for seed in config_seeds:
                                snapshot = _run_config(
                                    base_params,
                                    overrides,
                                    config_id,
                                    seed,
                                    out_dir,
                                    args.steps,
                                    args.report_every,
                                    args.device,
                                    burn_in_steps,
                                    args.resume,
                                    args.progress,
                                )

                                ep_rate_last = snapshot.get("epExactRateWindowLast", 0.0)
                                ep_rates = snapshot.get("ep_rate_series", [])
                                if ep_rates:
                                    ep_rate_last = ep_rates[-1]
                                tail = ep_rates[-args.last_m :] if ep_rates else []
                                ep_last_m_mean = None
                                ep_last_m_std = None
                                ep_last_m_n = len(tail)
                                if tail:
                                    mean = sum(tail) / len(tail)
                                    if len(tail) > 1:
                                        var = sum((v - mean) ** 2 for v in tail) / (len(tail) - 1)
                                        ep_last_m_std = math.sqrt(var)
                                    else:
                                        ep_last_m_std = 0.0
                                    ep_last_m_mean = mean

                                ep_autocorr = _autocorr_lag1(ep_rates)
                                ep_ess = _ess(ep_rates)

                                w_cap_frac_value = 0.0
                                if K_W > 0:
                                    w_cap_frac_value = float(B_w) / float(
                                        args.l_w * args.layers * math.prod(args.shape) * K_W
                                    )

                                row = {
                                    "config_id": config_id,
                                    "seed": seed,
                                    "device": args.device,
                                    "shape": str(args.shape),
                                    "meta_layers": args.layers,
                                    "beta": beta,
                                    "J": J_val,
                                    "radius_w": radius_w,
                                    "K_W": K_W,
                                    "l_w": args.l_w,
                                    "B_w": B_w,
                                    "w_fill": w_fill,
                                    "w_neighbor_weight": w_neighbor_weight,
                                    "steps": args.steps,
                                    "burn_in_sweeps": args.burn_in_sweeps,
                                    "measure_sweeps": args.measure_sweeps,
                                    "window_sweeps": args.window_sweeps,
                                    "last_m": args.last_m,
                                    "epExactRateWindowLast": ep_rate_last,
                                    "epLastMMean": ep_last_m_mean,
                                    "epLastMStd": ep_last_m_std,
                                    "epLastMN": ep_last_m_n,
                                    "epAutocorrLag1": ep_autocorr,
                                    "epEss": ep_ess,
                                    "acceptedFrac": snapshot.get("acceptedFrac", 0.0),
                                    "mag_abs": snapshot.get("mag_abs", 0.0),
                                    "w_zero_frac": snapshot.get("w_zero_frac", 0.0),
                                    "w_cap_frac": w_cap_frac_value,
                                    "w_active_deg_mean": snapshot.get("w_active_deg_mean", 0.0),
                                    "w_entropy_mean": snapshot.get("w_entropy_mean", 0.0),
                                    "w_r2_mean": snapshot.get("w_r2_mean", 0.0),
                                    "w_align_mean": snapshot.get("w_align_mean", 0.0),
                                }
                                row["_ep_last_m_values"] = tail
                                for name in kernel_names:
                                    row[f"proposalsWindow_{name}"] = snapshot.get(
                                        f"proposalsWindow_{name}", 0
                                    )
                                    row[f"acceptedWindow_{name}"] = snapshot.get(
                                        f"acceptedWindow_{name}", 0
                                    )
                                    row[f"acceptWindow_{name}"] = snapshot.get(
                                        f"acceptWindow_{name}", 0.0
                                    )
                                writer.writerow({key: row.get(key, "") for key in run_fields})
                                per_config.setdefault(config_id, []).append(row)

    agg_fields = [
        "config_id",
        "radius_w",
        "w_fill",
        "beta",
        "J",
        "w_neighbor_weight",
        "acceptedFracMean",
        "acceptedFracCI",
        "magAbsMean",
        "magAbsCI",
        "wEntropyMean",
        "wEntropyCI",
        "wR2Mean",
        "wR2CI",
        "epMean",
        "epSE",
        "epCIHalf",
        "epN",
        "epAutocorrLag1Mean",
        "epEssMean",
        "wZeroFracMean",
        "wCapFracMean",
        "wActiveDegMean",
        "pass",
    ]

    config_scores = []

    with open(agg_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fields)
        writer.writeheader()

        for config_id, rows in per_config.items():
            first = rows[0]
            radius_w = int(first["radius_w"])
            w_fill = float(first["w_fill"])
            beta = float(first["beta"])
            J_val = float(first["J"])
            w_neighbor_weight = float(first["w_neighbor_weight"])

            acc_mean, acc_ci = _mean_ci([r["acceptedFrac"] for r in rows])
            mag_mean, mag_ci = _mean_ci([r["mag_abs"] for r in rows])
            ent_mean, ent_ci = _mean_ci([r["w_entropy_mean"] for r in rows])
            r2_mean, r2_ci = _mean_ci([r["w_r2_mean"] for r in rows])
            ep_pool = []
            for row in rows:
                values = row.get("_ep_last_m_values") or []
                if values:
                    ep_pool.extend(values)
                else:
                    mean = row.get("epLastMMean")
                    std = row.get("epLastMStd")
                    n = int(row.get("epLastMN") or 0)
                    if mean is not None and std is not None and n > 1:
                        ep_pool.extend([mean] * n)
                    elif mean is not None:
                        ep_pool.append(mean)

            ep_mean, ep_se, ep_ci_half = _ci_half_95(ep_pool)
            ep_n = len(ep_pool)
            autocorrs = [r["epAutocorrLag1"] for r in rows if r.get("epAutocorrLag1") is not None]
            ess_vals = [r["epEss"] for r in rows if r.get("epEss") is not None]
            ep_autocorr_mean = sum(autocorrs) / len(autocorrs) if autocorrs else 0.0
            ep_ess_mean = sum(ess_vals) / len(ess_vals) if ess_vals else 0.0
            wz_mean, _ = _mean_ci([r["w_zero_frac"] for r in rows])
            wc_mean, _ = _mean_ci([r["w_cap_frac"] for r in rows])
            wd_mean, _ = _mean_ci([r["w_active_deg_mean"] for r in rows])

            pass_flag = (
                abs(ep_mean) <= 2e-4
                and ep_ci_half <= 5e-4
                and 0.10 <= acc_mean <= 0.85
                and (w_fill - 0.03) <= wc_mean <= (w_fill + 0.03)
                and 0.50 <= wz_mean <= 0.99
            )

            writer.writerow(
                {
                    "config_id": config_id,
                    "radius_w": radius_w,
                    "w_fill": w_fill,
                    "beta": beta,
                    "J": J_val,
                    "w_neighbor_weight": w_neighbor_weight,
                    "acceptedFracMean": acc_mean,
                    "acceptedFracCI": acc_ci,
                    "magAbsMean": mag_mean,
                    "magAbsCI": mag_ci,
                    "wEntropyMean": ent_mean,
                    "wEntropyCI": ent_ci,
                    "wR2Mean": r2_mean,
                    "wR2CI": r2_ci,
                    "epMean": ep_mean,
                    "epSE": ep_se,
                    "epCIHalf": ep_ci_half,
                    "epN": ep_n,
                    "epAutocorrLag1Mean": ep_autocorr_mean,
                    "epEssMean": ep_ess_mean,
                    "wZeroFracMean": wz_mean,
                    "wCapFracMean": wc_mean,
                    "wActiveDegMean": wd_mean,
                    "pass": str(pass_flag).lower(),
                }
            )

            config_scores.append(
                {
                    "config_id": config_id,
                    "radius_w": radius_w,
                    "w_fill": w_fill,
                    "beta": beta,
                    "J": J_val,
                    "w_neighbor_weight": w_neighbor_weight,
                    "acceptedFracMean": acc_mean,
                    "magAbsMean": mag_mean,
                    "wEntropyMean": ent_mean,
                    "wR2Mean": r2_mean,
                    "epMean": ep_mean,
                    "epCIHalf": ep_ci_half,
                    "pass": pass_flag,
                    "rows": rows,
                }
            )

    # Selection rule
    candidates = []
    for entry in config_scores:
        if entry["pass"]:
            candidates.append(entry)

    if not candidates:
        print("No configs satisfy hard constraints; see CSV for details.")
        return

    r2_vals = [c["wR2Mean"] for c in candidates]
    median_r2 = sorted(r2_vals)[len(r2_vals) // 2]

    def _score(entry: Dict[str, Any]) -> float:
        return entry["wEntropyMean"] - 0.1 * abs(entry["wR2Mean"] - median_r2)

    candidates.sort(key=_score, reverse=True)
    selected = candidates[0]

    selected_config_id = selected["config_id"]
    selected_row = selected["rows"][0]

    preset_kernel_weights = dict(base_kernel_weights)
    if "w_neighbor" in preset_kernel_weights:
        preset_kernel_weights["w_neighbor"] = selected["w_neighbor_weight"]

    preset = {
        "config_id": selected_config_id,
        "shape": args.shape,
        "layers": args.layers,
        "p3_on": False,
        "p6_on": False,
        "beta": selected["beta"],
        "J": selected["J"],
        "eta": 0.0,
        "eta_drive": 0.0,
        "radius_w": selected["radius_w"],
        "radius_k": 0,
        "l_w": args.l_w,
        "l_k": 1,
        "l_s": 0,
        "B_w": int(selected_row["B_w"]),
        "B_k": 0,
        "w_neighbor_weight": selected["w_neighbor_weight"],
        "kernel_weights": preset_kernel_weights,
    }

    params_dir = Path("scripts/params")
    params_dir.mkdir(parents=True, exist_ok=True)
    preset_path = params_dir / "phase1_null_balanced_v3.json"
    with open(preset_path, "w", encoding="utf-8") as handle:
        json.dump(preset, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
