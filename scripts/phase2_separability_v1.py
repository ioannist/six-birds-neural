#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import sys

from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim

# ensure sibling import works even when run as script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
from phase1_null_screen_v4 import (  # type: ignore
    _expected_proposals_per_step,
    _rate_micro,
)


def _load_preset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with path.open() as f:
        return json.load(f)


def _build_params_from_preset(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Params:
    data = {k: v for k, v in preset.items() if k not in {"config_id", "pass", "note"}}
    data.update(overrides)
    if isinstance(data.get("shape"), list):
        data["shape"] = tuple(data["shape"])
    if isinstance(data.get("kernel_weights"), dict):
        data["kernel_weights"] = dict(data["kernel_weights"])
    data.pop("w_neighbor_weight", None)
    return Params(**data)


def _early_stop(
    params: Params,
    seed: int,
    burn_steps: int,
    window_steps: int,
    min_windows: int,
    max_windows: int,
    last_m: int,
    pass_rule,
    fail_rules,
    out_path: Path,
    progress_path: Path,
    label: str,
    max_seconds_per_run: float,
) -> Dict[str, Any]:
    prev_total = 0.0
    window_rates: List[float] = []
    window_accepts: List[float] = []
    mismatch_first = None
    mismatch_last = None
    status = "RUNNING"
    start = time.monotonic()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    prog_handle = progress_path.open("a", encoding="utf-8")
    if progress_path.stat().st_size == 0:
        prog_handle.write(
            "case,seed,step,window_index,ep_rate_exact_window,mean_ep_last_m,ci_half,"
            "acceptedFracWindow,window_steps,window_proposals,window_accepted,pass\n"
        )
        prog_handle.flush()
        os.fsync(prog_handle.fileno())
    out_handle = out_path.open("w", encoding="utf-8")

    def _report(state, step, ep_ledger, accepted_frac):
        nonlocal prev_total, status, mismatch_first, mismatch_last
        if len(window_rates) >= max_windows:
            status = "FAIL_MAX_WINDOWS"
            return
        total = float(ep_ledger.get("ep_total_exact", 0.0))
        delta = total - prev_total
        prev_total = total
        window_steps_led = int(ep_ledger.get("window_steps", 0))
        rate = _rate_micro(delta, window_steps_led)
        if step <= burn_steps:
            return
        window_rates.append(rate)
        accept_window = float(ep_ledger.get("window_accept_frac", accepted_frac))
        window_accepts.append(accept_window)
        mismatch = ep_ledger.get("mismatch_abs_mean", None)
        if mismatch_first is None:
            mismatch_first = mismatch
        mismatch_last = mismatch
        tail = window_rates[-last_m:]
        mean_ep = sum(tail) / len(tail)
        var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
        std = math.sqrt(var)
        ci_half = 1.96 * std / math.sqrt(len(tail))
        pass_now = pass_rule(mean_ep, ci_half, accept_window, mismatch_first, mismatch_last)
        fail_now = fail_rules(mean_ep, ci_half, accept_window, len(window_rates))
        record = {
            "step": int(step),
            "window_index": len(window_rates),
            "ep_rate_exact_window": rate,
            "mean_ep_last_m": mean_ep,
            "ci_half": ci_half,
            "acceptedFracWindow": accept_window,
            "window_steps": window_steps_led,
            "window_proposals": int(ep_ledger.get("window_proposals", 0)),
            "window_accepted": int(ep_ledger.get("window_accepted", 0)),
            "mismatch_abs_mean": mismatch_last,
            "pass": pass_now,
        }
        out_handle.write(json.dumps(record) + "\n")
        out_handle.flush()
        prog_handle.write(
            f"{label},{seed},{record['step']},{record['window_index']},"
            f"{record['ep_rate_exact_window']},{record['mean_ep_last_m']},"
            f"{record['ci_half']},{record['acceptedFracWindow']},"
            f"{record['window_steps']},{record['window_proposals']},"
            f"{record['window_accepted']},{record['pass']}\n"
        )
        prog_handle.flush()
        os.fsync(prog_handle.fileno())
        nonlocal_status = None
        if pass_now:
            nonlocal_status = "PASS_EARLY"
        elif fail_now:
            nonlocal_status = fail_now
        elif time.monotonic() - start > max_seconds_per_run:
            nonlocal_status = "FAIL_TIME"
        if nonlocal_status:
            status = nonlocal_status

    def _stop(state, step, ep_ledger, accepted_frac):
        return status != "RUNNING" or len(window_rates) >= max_windows

    max_steps = burn_steps + max_windows * window_steps
    run_sim(
        params,
        seed=seed,
        steps=max_steps,
        report_every=window_steps,
        device=params.device,
        report_callback=_report,
        stop_callback=_stop,
    )
    out_handle.close()
    prog_handle.close()
    if status == "RUNNING":
        status = "FAIL_MAX_WINDOWS"
    tail = window_rates[-last_m:] if window_rates else [0.0]
    mean_ep = sum(tail) / len(tail)
    var = sum((v - mean_ep) ** 2 for v in tail) / max(1, len(tail) - 1)
    ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
    accepted_last = window_accepts[-1] if window_accepts else 0.0
    return {
        "status": status,
        "windows_used": len(window_rates),
        "mean_ep": mean_ep,
        "ci_half": ci_half,
        "acceptedFracWindow": accepted_last,
        "mismatch_first": mismatch_first,
        "mismatch_last": mismatch_last,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 separability suite")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out-dir", default=".tmp/phase2_separability_v1")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preset-out", default="scripts/params/phase1_null_balanced_quick_v8_24x24.json")
    args = parser.parse_args()

    preset_path = Path(args.preset)
    preset = _load_preset(preset_path)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [1, 2, 3]

    cases = {
        "meta_null": {
            "p3_on": False,
            "p6_on": False,
            "B_k": 2,
            "radius_k": 2,
            "l_k": 1,
            "eta": 0.0,
            "eta_drive": 0.0,
            "kernel_weights": {
                "spin_flip_color0": 1.0,
                "spin_flip_color1": 1.0,
                "k_local": 1.0,
                "k_neighbor_trade": 1.0,
                "k_p5_exchange": 0.0,
                "w_local": 1.0,
                "w_neighbor": preset.get("w_neighbor_weight", 0.25),
                "n_flip": 0.0,
                "s_step": 0.0,
            },
        },
        "p6_drive": {
            "p3_on": False,
            "p6_on": True,
            "eta": 0.0,
            "eta_drive": 0.5,
        },
        "p3_protocol": {
            "p3_on": True,
            "p6_on": False,
            "eta": 0.0,
            "eta_drive": 0.0,
        },
    }

    plan = {"seeds": seeds, "cases": list(cases.keys())}
    if args.dry_run:
        plan_path = out_root / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2))
        print(f"Wrote plan to {plan_path}")
        return

    burn_sweeps = 300
    window_sweeps = 80
    min_windows = 20
    max_windows = 120
    last_m = 10
    max_seconds_total = 5400
    max_seconds_per_run = 600
    start = time.monotonic()

    raw_rows = []
    completed = set()
    raw_rows = []
    for case_name, override in cases.items():
        case_dir = out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        raw_path = case_dir / "raw.csv"
        progress_path = case_dir / "progress.csv"
        jsonl_dir = case_dir / "jsonl"
        jsonl_dir.mkdir(parents=True, exist_ok=True)

        if args.resume and raw_path.exists():
            import csv as _csv
            with raw_path.open("r", encoding="utf-8") as rh:
                reader = _csv.DictReader(rh)
                for row in reader:
                    completed.add((case_name, int(row["seed"])))
                    raw_rows.append(row)

        # init progress header
        if not progress_path.exists():
            with progress_path.open("w", encoding="utf-8") as ph:
                ph.write(
                    "case,seed,step,window_index,ep_rate_exact_window,mean_ep_last_m,ci_half,"
                    "acceptedFracWindow,window_steps,window_proposals,window_accepted,pass\n"
                )

        for seed in seeds:
            if time.monotonic() - start > max_seconds_total:
                print("TOTAL TIME CAP HIT")
                break
            if (case_name, seed) in completed:
                print(f"SKIP {case_name} seed={seed} (resume)")
                continue
            overrides = dict(override)
            overrides.setdefault("kernel_weights", preset.get("kernel_weights", {}))
            overrides["device"] = args.device
            params = _build_params_from_preset(preset, overrides)
            N = math.prod(params.shape)
            expected_props = _expected_proposals_per_step(N, args.device, params.kernel_weights)
            burn_steps = int(math.ceil(burn_sweeps * N / expected_props))
            win_steps = int(math.ceil(window_sweeps * N / expected_props))

            def pass_rule(mean_ep, ci_half, accept, m_first, m_last):
                if case_name == "meta_null":
                    return abs(mean_ep) <= 2e-4 and ci_half <= 2e-4 and accept >= 0.005
                if case_name == "p6_drive":
                    ok_mismatch = True
                    if m_first is not None and m_last is not None:
                        ok_mismatch = (m_last - m_first) <= -0.01
                    return mean_ep >= 2e-4 and ci_half <= 5e-4 and ok_mismatch
                if case_name == "p3_protocol":
                    # no strobe metric available; never auto-pass
                    return False
                return False

            def fail_rules(mean_ep, ci_half, accept, widx):
                if case_name == "meta_null":
                    if abs(mean_ep) > 5e-4:
                        return "FAIL_MEAN_EARLY"
                    if ci_half > 8e-4:
                        return "FAIL_CI_STUCK"
                    if widx >= 3 and accept < 0.002:
                        return "FAIL_ACCEPT_STUCK"
                if case_name == "p6_drive":
                    if widx >= min_windows + 5 and mean_ep < 0:
                        return "FAIL_WRONG_SIGN"
                    if ci_half > 1e-3:
                        return "FAIL_CI_STUCK"
                return False

            jsonl_path = jsonl_dir / f"{case_name}_seed{seed}.jsonl"
            result = _early_stop(
                params=params,
                seed=seed,
                burn_steps=burn_steps,
                window_steps=win_steps,
                min_windows=min_windows,
                max_windows=max_windows,
                last_m=last_m,
                pass_rule=pass_rule,
                fail_rules=fail_rules,
                out_path=jsonl_path,
                progress_path=progress_path,
                label=f"{case_name}",
                max_seconds_per_run=max_seconds_per_run,
            )
            raw_rows.append(
                {
                    "case": case_name,
                    "seed": seed,
                    "status": result["status"],
                    "windows_used": result["windows_used"],
                    "mean_ep": result["mean_ep"],
                    "ci_half": result["ci_half"],
                    "acceptedFracWindow": result["acceptedFracWindow"],
                    "mismatch_first": result["mismatch_first"],
                    "mismatch_last": result["mismatch_last"],
                }
            )
            print(
                f"SUMMARY seed={seed} case={case_name} status={result['status']} "
                f"windows={result['windows_used']} lastm_mean={result['mean_ep']} "
                f"lastm_ci={result['ci_half']} lastm_accept={result['acceptedFracWindow']}"
            )
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            fieldnames = list(raw_rows[0].keys()) if raw_rows else []
            writer = csv.DictWriter(rh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([r for r in raw_rows if r["case"] == case_name])
    # aggregate
    from collections import Counter, defaultdict

    counts = defaultdict(Counter)
    for r in raw_rows:
        counts[r["case"]][r["status"]] += 1
    print(f"STATUS COUNTS: {dict((k, dict(v)) for k, v in counts.items())}")

    preset_out = Path(args.preset_out)
    preset_out.parent.mkdir(parents=True, exist_ok=True)
    preset_out.write_text(json.dumps(preset, indent=2))
    print("PRESET_SELECTED:", preset_out)


if __name__ == "__main__":
    main()
