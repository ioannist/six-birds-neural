from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim
from ratchet_gpu.diagnostics import compute_snapshot, to_json_line


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [int(item) for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must be comma-separated ints")
    return tuple(parts)


def _parse_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not value:
        return weights
    for item in value.split(","):
        if not item.strip():
            continue
        name, raw = item.split(":")
        weights[name] = float(raw)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="ratchet-gpu CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a simulation")
    run.add_argument("--shape", type=_parse_shape, default=(6, 6))
    run.add_argument("--layers", type=int, default=2)
    run.add_argument("--meta-layers", dest="layers", type=int)
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    run.add_argument("--p3-on", type=int, default=0)
    run.add_argument("--p6-on", type=int, default=0)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--beta", type=float, default=1.0)
    run.add_argument("--J", type=float, default=1.0)
    run.add_argument("--kappa-T", type=float, default=1.0)
    run.add_argument("--eta", type=float, default=0.2)
    run.add_argument("--eta-drive", type=float, default=0.0)
    run.add_argument("--l-s", type=int, default=1)
    run.add_argument("--l-w", type=int, default=3)
    run.add_argument("--l-k", type=int, default=3)
    run.add_argument("--B-w", type=int, default=2)
    run.add_argument("--B-k", type=int, default=2)
    run.add_argument("--radius-w", type=int, default=1)
    run.add_argument("--radius-k", type=int, default=2)
    run.add_argument("--report-every", type=int, default=1000)
    run.add_argument(
        "--kernel-weights",
        default="",
        help="comma-separated name:weight pairs",
    )
    run.add_argument("--out", default="", help="write JSONL diagnostics to path")
    run.add_argument("--summary", default="", help="write CSV summary to path")

    args = parser.parse_args()

    weights = _parse_weights(args.kernel_weights)

    kwargs = dict(
        shape=args.shape,
        layers=args.layers,
        p3_on=bool(args.p3_on),
        p6_on=bool(args.p6_on),
        beta=args.beta,
        J=args.J,
        kappa_T=args.kappa_T,
        eta=args.eta,
        eta_drive=args.eta_drive,
        l_s=args.l_s,
        l_w=args.l_w,
        l_k=args.l_k,
        B_w=args.B_w,
        B_k=args.B_k,
        radius_w=args.radius_w,
        radius_k=args.radius_k,
        report_every=args.report_every,
        device=args.device,
    )
    if weights:
        kwargs["kernel_weights"] = weights

    params = Params(**kwargs)

    diag_state = None
    last_snapshot = None
    out_path = args.out.strip()
    summary_path = args.summary.strip()

    out_handle = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_handle = open(out_path, "w", encoding="utf-8")

    def _report_callback(state, step, ep_ledger, accepted_frac):
        nonlocal diag_state, last_snapshot
        snapshot, diag_state = compute_snapshot(
            state, step, ep_ledger, accepted_frac, diag_state
        )
        last_snapshot = snapshot
        if out_handle is not None:
            out_handle.write(to_json_line(snapshot) + "\n")

    summary = run_sim(
        params,
        seed=args.seed,
        steps=args.steps,
        report_every=args.report_every,
        report_callback=_report_callback if out_handle or summary_path else None,
    )

    if out_handle is not None:
        out_handle.close()

    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        if last_snapshot is None:
            last_snapshot = {
                "step": args.steps,
                "ep_total_exact": 0.0,
                "ep_rate_exact_window": 0.0,
                "mismatch_abs_mean": None,
                "k_entropy_mean": None,
                "k_r2_mean": None,
                "k_coh_mean": None,
                "acceptedFrac": summary.get("acceptedFrac", 0.0),
            }
        fields = [
            "step",
            "ep_total_exact",
            "ep_rate_exact_window",
            "mismatch_abs_mean",
            "k_entropy_mean",
            "k_r2_mean",
            "k_coh_mean",
            "acceptedFrac",
        ]
        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            row = {}
            for key in fields:
                value = last_snapshot.get(key, None)
                row[key] = "" if value is None else value
            writer.writerow(row)

    print("summary")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
