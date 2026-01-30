#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_null


def _parse_seeds(value: str) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [int(item) for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must be comma-separated ints")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Null-regime epExact check")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--report-every", type=int, default=100000)
    parser.add_argument("--shape", type=_parse_shape, default="6,6")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--kappa-t", type=float, default=1.0)
    parser.add_argument("--l-k", type=int, default=3)
    parser.add_argument("--l-s", type=int, default=1)
    parser.add_argument("--B-k", type=int, default=2)
    parser.add_argument("--radius-w", type=int, default=1)
    parser.add_argument("--radius-k", type=int, default=2)
    parser.add_argument("--stencil-policy-w", default="l1_ball_odd")
    parser.add_argument("--stencil-policy-k", default="l1_ball_even")

    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)

    params = Params(
        shape=args.shape,
        layers=args.layers,
        beta=args.beta,
        eta=args.eta,
        kappa_T=args.kappa_t,
        l_k=args.l_k,
        l_s=args.l_s,
        B_k=args.B_k,
        stencil_policy_w=args.stencil_policy_w,
        stencil_policy_k=args.stencil_policy_k,
        radius_w=args.radius_w,
        radius_k=args.radius_k,
        device=args.device,
    )

    header = f"{'seed':>6}  {'epExactRateWindowLast':>22}  {'acceptedFrac':>13}  {'pass':>5}"
    print(header)
    passed = 0

    for seed in seeds:
        summary = run_null(
            params,
            seed=seed,
            steps=args.steps,
            report_every=args.report_every,
            device=args.device,
        )
        rate = summary["epExactRateWindowLast"]
        accepted = summary["acceptedFrac"]
        ok = abs(rate) <= 2e-4
        passed += int(ok)
        print(f"{seed:6d}  {rate:22.6e}  {accepted:13.6f}  {str(ok).lower():>5}")

    total = len(seeds)
    print(f"passed {passed}/{total}")
    if passed == total:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
