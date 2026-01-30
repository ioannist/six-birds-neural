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

from ratchet_gpu.diagnostics import compute_snapshot, to_json_line
from ratchet_gpu.params import Params
from ratchet_gpu.sim import run_sim, _cycle_list

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
try:
    from phase1_null_screen_v4 import _expected_proposals_per_step  # type: ignore
except Exception:  # pragma: no cover
    def _expected_proposals_per_step(N: int, device: str, kernel_weights: Dict[str, float]) -> float:
        return float(N)


def _load_preset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with path.open() as f:
        return json.load(f)


def _as_params(preset: Dict[str, Any], overrides: Dict[str, Any]) -> Params:
    data = {k: v for k, v in preset.items() if k not in {"config_id", "pass", "note"}}
    data.update(overrides)
    if isinstance(data.get("shape"), list):
        data["shape"] = tuple(data["shape"])
    if isinstance(data.get("kernel_weights"), dict):
        data["kernel_weights"] = dict(data["kernel_weights"])
    data.pop("w_neighbor_weight", None)
    return Params(**data)


def _ensure_k_weights(kw: Dict[str, float]) -> Dict[str, float]:
    kw = dict(kw)
    kw["k_local"] = max(float(kw.get("k_local", 0.0) or 0.0), 0.25)
    kw["k_neighbor_trade"] = max(float(kw.get("k_neighbor_trade", 0.0) or 0.0), 0.25)
    return kw


def _match_cycle_weights(kw: Dict[str, float], cycle: List[str]) -> Dict[str, float]:
    matched = {k: (1.0 if k in cycle else 0.0) for k in kw.keys()}
    for name in cycle:
        matched.setdefault(name, 1.0)
    return matched


def _case_params(base: Params, case: str, eta: float, strobe_sig: str) -> Params:
    kw = _ensure_k_weights(base.kernel_weights)
    overrides = {
        "p6_on": False,
        "eta_drive": 0.0,
        "eta": eta,
        "strobe_on": True,
        "strobe_signature": strobe_sig,
        "B_k": max(2, base.B_k),
        "radius_k": max(2, base.radius_k),
        "l_k": max(2, base.l_k),
        "kernel_weights": kw,
    }
    overrides["p3_on"] = case in {"protocol_p3_fwd", "protocol_p3_rev", "protocol_p3_on"}
    params = Params.from_dict(base, overrides)
    if params.eta <= 0 or params.B_k <= 0 or params.radius_k <= 0 or params.l_k <= 0:
        raise ValueError("Phase3 requires eta>0 and K coupling enabled")
    if kw["k_local"] <= 0 or kw["k_neighbor_trade"] <= 0:
        raise ValueError("Phase3 requires K kernels positive")
    return params


def _protocol_cycle(reverse: bool = False) -> List[str]:
    cycle = _cycle_list()
    if not reverse or not cycle:
        return cycle
    return list(reversed(cycle))


def _summarize_tail(vals: List[float], last_m: int) -> tuple[float, float]:
    tail = vals[-last_m:] if vals else [0.0]
    mean_val = sum(tail) / len(tail)
    var = sum((v - mean_val) ** 2 for v in tail) / max(1, len(tail) - 1)
    ci_half = 1.96 * math.sqrt(var) / math.sqrt(len(tail))
    return mean_val, ci_half


def _strobe_rate_per_proposal(snapshot: Dict[str, Any]) -> float:
    raw = float(snapshot.get("strobe_rate_window", 0.0))
    transitions = int(snapshot.get("strobe_transitions_window", 0))
    proposals = int(snapshot.get("window_proposals", snapshot.get("window_steps", 0)))
    if proposals <= 0 or transitions <= 0:
        return 0.0
    return raw * transitions / proposals


def _effective_min_strobe_transitions(requested: int, window_steps: int, cycle_len: int) -> int:
    if requested <= 0:
        return 0
    if cycle_len <= 0:
        cycle_len = 1
    max_obs = window_steps // cycle_len
    max_transitions = max(0, max_obs - 1)
    return min(requested, max_transitions)


def _edge_key(entry: Dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(entry["a"]), tuple(entry["b"])


def _aggregate_top_edge(top_lists: List[List[Dict[str, Any]]]) -> tuple[str, float, Dict]:
    totals: Dict[tuple[tuple[int, ...], tuple[int, ...]], Dict[str, float]] = {}
    for window_list in top_lists:
        for entry in window_list:
            edge = _edge_key(entry)
            stats = totals.setdefault(edge, {"sum": 0.0, "count": 0.0})
            stats["sum"] += float(entry.get("j", 0.0))
            stats["count"] += 1.0
    if not totals:
        return "", 0.0, {}
    def mean_j(item: tuple[tuple[tuple[int, ...], tuple[int, ...]], Dict[str, float]]) -> float:
        stats = item[1]
        return stats["sum"] / max(1.0, stats["count"])
    edge, stats = max(totals.items(), key=lambda kv: abs(mean_j(kv)))
    mean_val = mean_j((edge, stats))
    edge_str = json.dumps([list(edge[0]), list(edge[1])])
    return edge_str, float(mean_val), totals


def _accumulate_current_map(items_lists: List[List[Dict[str, Any]]]) -> Dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    current_map: Dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for items in items_lists:
        for entry in items:
            u = tuple(entry.get("u") or entry.get("a") or [])
            v = tuple(entry.get("v") or entry.get("b") or [])
            if not u or not v:
                continue
            edge = tuple(sorted((u, v)))
            current_map[edge] = current_map.get(edge, 0.0) + float(entry.get("j", 0.0))
    return current_map


def _current_overlap(
    fwd: Dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    rev: Dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    keys = set(fwd) | set(rev)
    norm_f = math.sqrt(sum((fwd.get(k, 0.0) ** 2) for k in keys))
    norm_r = math.sqrt(sum((rev.get(k, 0.0) ** 2) for k in keys))
    dot = sum(fwd.get(k, 0.0) * rev.get(k, 0.0) for k in keys)
    overlap = dot / (norm_f * norm_r + eps)
    rev_error = math.sqrt(
        sum(((fwd.get(k, 0.0) + rev.get(k, 0.0)) ** 2) for k in keys)
    ) / (norm_f + norm_r + eps)
    return norm_f, norm_r, overlap, rev_error


def _should_check_reversal(windows_used: int, threshold: int) -> bool:
    return windows_used >= threshold


def run_case(
    case: str,
    params: Params,
    seeds: List[int],
    out_dir: Path,
    burn_sweeps: int,
    window_sweeps: int,
    min_windows: int,
    max_windows: int,
    last_m: int,
    accept_min: float,
    mean_thresh: float,
    diff_thresh: float,
    ci_thresh: float,
    min_strobe_transitions: int,
    min_strobe_unique: int,
    min_strobe_bidir: int,
    cycle: List[str],
    control_stats: Dict[int, Dict[str, float]] | None,
    start_total: float,
    max_seconds_total: float,
    max_seconds_per_run: float,
    weights_mode: str,
    reversal_check_after_windows: int,
    metric_mode: str,
) -> List[Dict[str, Any]]:
    N = math.prod(params.shape)
    expected_props = _expected_proposals_per_step(N, str(params.device), params.kernel_weights)
    burn_steps = int(math.ceil(burn_sweeps * N / expected_props))
    window_steps = int(math.ceil(window_sweeps * N / expected_props))
    cycle_len = len(cycle) if cycle else 1
    min_transitions_used = _effective_min_strobe_transitions(
        min_strobe_transitions, window_steps, cycle_len
    )

    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = case_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    progress_path = case_dir / "progress.csv"
    raw_path = case_dir / "raw.csv"
    if not progress_path.exists():
        with progress_path.open("w", encoding="utf-8") as ph:
            ph.write(
                "case,seed,weights_mode,step,window_index,strobe_rate_raw,strobe_rate_per_proposal,"
                "strobe_symgap,strobe_current_l2,strobe_currents_count,strobe_mean_last_m,strobe_ci_half,"
                "acceptedFracWindow,window_proposals,strobe_transitions,min_strobe_transitions_used,"
                "strobe_unique,strobe_edges,pass\n"
            )

    raw_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        if time.monotonic() - start_total > max_seconds_total:
            print("TOTAL TIME CAP HIT")
            break
        diag_state = None
        strobe_rates: List[float] = []
        symgap_rates: List[float] = []
        current_l2_rates: List[float] = []
        currents_windows: List[List[Dict[str, Any]]] = []
        accepts: List[float] = []
        status = "RUNNING"
        print(
            "STROBE_TRANSITIONS_THRESH "
            f"case={case} seed={seed} requested={min_strobe_transitions} "
            f"used={min_transitions_used} window_steps={window_steps} cycle_len={cycle_len}"
        )
        jsonl_path = jsonl_dir / f"{case}_seed{seed}.jsonl"
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        progress_handle = progress_path.open("a", encoding="utf-8")
        run_start = time.monotonic()

        def report_cb(state, step, ep_ledger, accepted_frac):
            nonlocal diag_state, status
            if len(strobe_rates) >= max_windows or status != "RUNNING":
                return
            snapshot, diag_state = compute_snapshot(state, step, ep_ledger, accepted_frac, diag_state)
            strobe_rate_raw = float(snapshot.get("strobe_rate_window", 0.0))
            strobe_rate = _strobe_rate_per_proposal(snapshot)
            transitions = int(snapshot.get("strobe_transitions_window", 0))
            uniq = int(snapshot.get("strobe_unique_states_window", 0))
            edges = int(snapshot.get("strobe_bidirectional_edges_window", 0))
            symgap = float(snapshot.get("strobe_symgap_window", 0.0))
            current_l2 = float(snapshot.get("strobe_current_l2_window", 0.0))
            currents = snapshot.get(
                "strobe_current_map_items_window",
                snapshot.get("strobe_currents_window", []),
            )
            currents_count = int(
                snapshot.get(
                    "strobe_current_map_items_count_window",
                    snapshot.get("strobe_currents_count_window", 0),
                )
            )
            snapshot["strobe_rate_raw_window"] = strobe_rate_raw
            snapshot["strobe_rate_per_proposal_window"] = strobe_rate
            snapshot["min_strobe_transitions_used"] = min_transitions_used
            jsonl_handle.write(to_json_line(snapshot) + "\n")
            jsonl_handle.flush()
            progress_handle.write(
                f"{case},{seed},{weights_mode},{snapshot['step']},{len(strobe_rates)+1},"
                f"{strobe_rate_raw},{strobe_rate},{symgap},{current_l2},{currents_count},0.0,0.0,"
                f"{snapshot.get('acceptedFrac', 0.0)},{snapshot.get('window_proposals', 0)},"
                f"{transitions},{min_transitions_used},{uniq},{edges},{step <= burn_steps}\n"
            )
            progress_handle.flush()
            os.fsync(progress_handle.fileno())

            if step <= burn_steps:
                return

            strobe_rates.append(strobe_rate)
            symgap_rates.append(symgap)
            current_l2_rates.append(current_l2)
            if isinstance(currents, list):
                currents_windows.append(currents)
            accepts.append(float(snapshot.get("acceptedFrac", 0.0)))

            if (
                uniq < min_strobe_unique
                or edges < min_strobe_bidir
                or transitions < min_transitions_used
            ):
                status = "FAIL_STROBE_SPARSE"
                debug_path = case_dir / f"{case}_seed{seed}_strobe_debug.json"
                debug_payload = {
                    "signature": snapshot.get("strobe_signature", ""),
                    "unique_states": uniq,
                    "bidirectional_edges": edges,
                    "transitions": transitions,
                    "top_states": snapshot.get("strobe_top_states_window", []),
                    "strobe_cycle_len": snapshot.get("strobe_cycle_len", 0),
                    "strobe_stride": snapshot.get("strobe_stride", 0),
                }
                debug_path.write_text(json.dumps(debug_payload, indent=2))
                return

            if len(strobe_rates) >= min_windows:
                mean_val, ci_half = _summarize_tail(strobe_rates, last_m)
                accept_mean, _ = _summarize_tail(accepts, last_m)
                if accept_mean < accept_min:
                    status = "FAIL_ACCEPT"
                    return
                if case == "control_p3_off":
                    if metric_mode == "diff_vs_control":
                        status = "PASS_EARLY"
                    else:
                        if abs(mean_val) <= mean_thresh and ci_half <= ci_thresh:
                            status = "PASS_EARLY"
                        elif ci_half <= ci_thresh and abs(mean_val) > mean_thresh:
                            status = "FAIL_MEAN_EARLY"
                else:
                    if metric_mode == "diff_vs_control":
                        if not control_stats or seed not in control_stats:
                            status = "FAIL_CONTROL_MISSING"
                            return
                        control_l2 = float(
                            control_stats[seed].get("current_l2_mean_last_m", 0.0)
                        )
                        current_l2_mean, _ = _summarize_tail(current_l2_rates, last_m)
                        rel_change = abs(current_l2_mean - control_l2) / max(
                            abs(control_l2), 1e-9
                        )
                        if rel_change >= diff_thresh:
                            status = "PASS_EARLY"
                        elif rel_change < 0.05:
                            status = "FAIL_NO_EFFECT"
                            return
                    else:
                        if len(strobe_rates) >= reversal_check_after_windows:
                            status = "DONE_REVERSAL_WINDOW"
                            return

            if status == "RUNNING" and time.monotonic() - run_start > max_seconds_per_run:
                status = "FAIL_TIME"

        def stop_cb(state, step, ep_ledger, accepted_frac):
            return status != "RUNNING" or len(strobe_rates) >= max_windows

        max_steps = burn_steps + max_windows * window_steps
        run_sim(
            params,
            seed=seed,
            steps=max_steps,
            report_every=window_steps,
            device=params.device,
            report_callback=report_cb,
            stop_callback=stop_cb,
            protocol_cycle=cycle,
        )
        jsonl_handle.close()
        progress_handle.close()
        if status == "RUNNING":
            status = "FAIL_MAX_WINDOWS"
        mean_val, ci_half = _summarize_tail(strobe_rates, last_m)
        symgap_mean, symgap_ci = _summarize_tail(symgap_rates, last_m)
        current_l2_mean, current_l2_ci = _summarize_tail(current_l2_rates, last_m)
        accept_mean, _ = _summarize_tail(accepts, last_m)
        current_map = _accumulate_current_map(currents_windows[-last_m:])
        control_l2_ref = None
        rel_change = 0.0
        if metric_mode == "diff_vs_control" and control_stats and seed in control_stats:
            control_l2_ref = float(control_stats[seed].get("current_l2_mean_last_m", 0.0))
            rel_change = abs(current_l2_mean - control_l2_ref) / max(
                abs(control_l2_ref), 1e-9
            )
        top_edge = ""
        top_j = 0.0
        if current_map:
            edge, val = max(current_map.items(), key=lambda kv: abs(kv[1]))
            top_edge = json.dumps([list(edge[0]), list(edge[1])])
            top_j = float(val)
        raw_rows.append(
            {
                "case": case,
                "seed": seed,
                "status": status,
                "windows_used": len(strobe_rates),
                "strobe_rate_metric": "per_proposal",
                "strobe_mean_last_m": mean_val,
                "strobe_ci_half": ci_half,
                "strobe_symgap_mean_last_m": symgap_mean,
                "strobe_symgap_ci_half": symgap_ci,
                "strobe_current_l2_mean_last_m": current_l2_mean,
                "strobe_current_l2_ci_half": current_l2_ci,
                "acceptedFracWindowMean": accept_mean,
                "weights_mode": weights_mode,
                "control_current_l2_ref": control_l2_ref,
                "rel_change": rel_change,
                "top_edge": top_edge,
                "top_edge_j": top_j,
                "edges_fwd": len(current_map),
                "edges_rev": 0,
                "shared_edges": 0,
                "shared_ratio": 0.0,
                "overlap_neg": 0.0,
                "_current_map": current_map,
            }
        )
        print(
            f"SUMMARY case={case} seed={seed} status={status} windows={len(strobe_rates)} "
            f"mean={mean_val} ci={ci_half} accept={accept_mean}"
        )
        if case == "control_p3_off" and status != "PASS_EARLY":
            break

    if raw_rows:
        fieldnames = [k for k in raw_rows[0].keys() if not k.startswith("_")]
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        counts: Dict[str, int] = {}
        for r in raw_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        with (case_dir / "status_counts.json").open("w", encoding="utf-8") as sh:
            json.dump(counts, sh, indent=2)
        pass_count = sum(1 for r in raw_rows if r["status"] == "PASS_EARLY")
        with (case_dir / "agg.csv").open("w", encoding="utf-8", newline="") as ah:
            writer = csv.DictWriter(
                ah, fieldnames=["case", "pass_count", "total", "pass_rate"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "case": case,
                    "pass_count": pass_count,
                    "total": len(raw_rows),
                    "pass_rate": pass_count / max(1, len(raw_rows)),
                }
            )
    return raw_rows


def main():
    parser = argparse.ArgumentParser(description="Phase 3 P3 pumping v4")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--preset", default="scripts/params/phase2_drive_k_balanced_v6.json")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--out-dir", default=".tmp/phase3_p3_pumping_v4")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--burn-in-sweeps", type=float, default=150)
    parser.add_argument("--window-sweeps", type=float, default=80)
    parser.add_argument("--min-windows", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--last-m", type=int, default=5)
    parser.add_argument("--accept-min", type=float, default=0.005)
    parser.add_argument("--mean-thresh-control", type=float, default=5e-4)
    parser.add_argument("--diff-thresh", type=float, default=1e-4)
    parser.add_argument("--ci-thresh", type=float, default=1e-3)
    parser.add_argument("--min-strobe-transitions", type=int, default=200)
    parser.add_argument("--min-strobe-unique", type=int, default=3)
    parser.add_argument("--min-strobe-bidir", type=int, default=1, dest="min_strobe_bidir")
    parser.add_argument("--min-strobe-bidirectional-edges", type=int, dest="min_strobe_bidir")
    parser.add_argument("--strobe-signature", default="mag_wmass")
    parser.add_argument("--reversal-check-after-windows", type=int, default=0)
    parser.add_argument("--overlap-thresh", type=float, default=-0.2)
    parser.add_argument("--rev-error-thresh", type=float, default=0.8)
    parser.add_argument("--min-current-l2", type=float, default=1e-3)
    parser.add_argument(
        "--metric-mode",
        choices=["diff_vs_control", "reversal_overlap"],
        default="diff_vs_control",
    )
    parser.add_argument("--max-seconds-total", type=float, default=5400)
    parser.add_argument("--max-seconds-per-run", type=float, default=900)
    parser.add_argument(
        "--match-control-cycle-weights",
        dest="match_control_cycle_weights",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-match-control-cycle-weights",
        dest="match_control_cycle_weights",
        action="store_false",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preset = _load_preset(Path(args.preset))
    base = _as_params(preset, {"device": args.device})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.monotonic()
    cycle_fwd = _protocol_cycle(False)
    cycle_rev = _protocol_cycle(True)
    print(f"PROTOCOL_CYCLE fwd={cycle_fwd}")
    print(f"PROTOCOL_CYCLE rev={cycle_rev}")
    reversal_check_after = args.reversal_check_after_windows or args.min_windows

    control_params = _case_params(base, "control_p3_off", eta=args.eta, strobe_sig=args.strobe_signature)
    weights_mode = "preset"
    if args.match_control_cycle_weights:
        matched = _match_cycle_weights(control_params.kernel_weights, cycle_fwd)
        matched = _ensure_k_weights(matched)
        control_params = Params.from_dict(control_params, {"kernel_weights": matched})
        weights_mode = "matched_cycle"
    control_rows = run_case(
        case="control_p3_off",
        params=control_params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        accept_min=args.accept_min,
        mean_thresh=args.mean_thresh_control,
        diff_thresh=args.diff_thresh,
        ci_thresh=args.ci_thresh,
        min_strobe_transitions=args.min_strobe_transitions,
        min_strobe_unique=args.min_strobe_unique,
        min_strobe_bidir=args.min_strobe_bidir,
        cycle=cycle_fwd,
        control_stats=None,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        weights_mode=weights_mode,
        reversal_check_after_windows=reversal_check_after,
        metric_mode=args.metric_mode,
    )

    control_stats: Dict[int, Dict[str, float]] = {}
    for r in control_rows:
        control_stats[r["seed"]] = {
            "mean": float(r.get("strobe_mean_last_m", 0.0)),
            "ci_half": float(r.get("strobe_ci_half", 0.0)),
            "current_l2_mean_last_m": float(r.get("strobe_current_l2_mean_last_m", 0.0)),
            "symgap_mean_last_m": float(r.get("strobe_symgap_mean_last_m", 0.0)),
            "acceptedFracWindowMean": float(r.get("acceptedFracWindowMean", 0.0)),
        }
    if not control_rows or any(r["status"] != "PASS_EARLY" for r in control_rows):
        print("STATUS_COUNTS control_p3_off:", {r["status"]: 1 for r in control_rows})
        for r in control_rows:
            if r["status"] == "FAIL_STROBE_SPARSE":
                debug_path = out_dir / "control_p3_off" / f"control_p3_off_seed{r['seed']}_strobe_debug.json"
                if debug_path.exists():
                    info = json.loads(debug_path.read_text())
                    jsonl_path = out_dir / "control_p3_off" / "jsonl" / f"control_p3_off_seed{r['seed']}.jsonl"
                    print(
                        "STOP_REASON=CONTROL_STROBE_DEGENERATE "
                        f"seed={r['seed']} unique_states={info.get('unique_states')} "
                        f"bidirectional_edges={info.get('bidirectional_edges')} "
                        f"transitions={info.get('transitions')} jsonl={jsonl_path}"
                    )
                break
        return

    # Protocol run
    proto_case = "protocol_p3_fwd" if args.metric_mode == "reversal_overlap" else "protocol_p3_on"
    proto_fwd_params = _case_params(base, proto_case, eta=args.eta, strobe_sig=args.strobe_signature)
    proto_fwd_rows = run_case(
        case=proto_case,
        params=proto_fwd_params,
        seeds=seeds,
        out_dir=out_dir,
        burn_sweeps=args.burn_in_sweeps,
        window_sweeps=args.window_sweeps,
        min_windows=args.min_windows,
        max_windows=args.max_windows,
        last_m=args.last_m,
        accept_min=args.accept_min,
        mean_thresh=args.mean_thresh_control,
        diff_thresh=args.diff_thresh,
        ci_thresh=args.ci_thresh,
        min_strobe_transitions=args.min_strobe_transitions,
        min_strobe_unique=args.min_strobe_unique,
        min_strobe_bidir=args.min_strobe_bidir,
        cycle=cycle_fwd,
        control_stats=control_stats,
        start_total=start_total,
        max_seconds_total=args.max_seconds_total,
        max_seconds_per_run=args.max_seconds_per_run,
        weights_mode="protocol_cycle",
        reversal_check_after_windows=reversal_check_after,
        metric_mode=args.metric_mode,
    )

    proto_rev_rows: List[Dict[str, Any]] = []
    if args.metric_mode == "reversal_overlap":
        proto_rev_params = _case_params(base, "protocol_p3_rev", eta=args.eta, strobe_sig=args.strobe_signature)
        proto_rev_rows = run_case(
            case="protocol_p3_rev",
            params=proto_rev_params,
            seeds=seeds,
            out_dir=out_dir,
            burn_sweeps=args.burn_in_sweeps,
            window_sweeps=args.window_sweeps,
            min_windows=args.min_windows,
            max_windows=args.max_windows,
            last_m=args.last_m,
            accept_min=args.accept_min,
            mean_thresh=args.mean_thresh_control,
            diff_thresh=args.diff_thresh,
            ci_thresh=args.ci_thresh,
            min_strobe_transitions=args.min_strobe_transitions,
            min_strobe_unique=args.min_strobe_unique,
            min_strobe_bidir=args.min_strobe_bidir,
            cycle=cycle_rev,
            control_stats=control_stats,
            start_total=start_total,
            max_seconds_total=args.max_seconds_total,
            max_seconds_per_run=args.max_seconds_per_run,
            weights_mode="protocol_cycle",
            reversal_check_after_windows=reversal_check_after,
            metric_mode=args.metric_mode,
        )

    def rewrite_case(case_name: str, rows: List[Dict[str, Any]]) -> None:
        case_dir = out_dir / case_name
        raw_path = case_dir / "raw.csv"
        agg_path = case_dir / "agg.csv"
        if not rows:
            return
        fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
        with raw_path.open("w", encoding="utf-8", newline="") as rh:
            writer = csv.DictWriter(rh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        pass_count = sum(1 for r in rows if r["status"] == "PASS_EARLY")
        with agg_path.open("w", encoding="utf-8", newline="") as ah:
            writer = csv.DictWriter(ah, fieldnames=["case", "pass_count", "total", "pass_rate"])
            writer.writeheader()
            writer.writerow(
                {
                    "case": case_name,
                    "pass_count": pass_count,
                    "total": len(rows),
                    "pass_rate": pass_count / max(1, len(rows)),
                }
            )

    if args.metric_mode == "reversal_overlap":
        # Update protocol statuses based on current reversal
        fwd_map = {r["seed"]: r for r in proto_fwd_rows}
        rev_map = {r["seed"]: r for r in proto_rev_rows}
        for seed in seeds:
            f = fwd_map.get(seed)
            r = rev_map.get(seed)
            if not f or not r:
                continue
            if f["status"].startswith("FAIL") or r["status"].startswith("FAIL"):
                continue
            windows_used = min(int(f.get("windows_used", 0)), int(r.get("windows_used", 0)))
            if not _should_check_reversal(windows_used, reversal_check_after):
                continue
            f_map = f.get("_current_map", {})
            r_map = r.get("_current_map", {})
            norm_f, norm_r, overlap, rev_error = _current_overlap(f_map, r_map)
            overlap_neg = -overlap
            shared = len(set(f_map) & set(r_map))
            min_edges = min(len(f_map), len(r_map)) if f_map and r_map else 0
            shared_ratio = shared / min_edges if min_edges else 0.0
            f["current_norm"] = norm_f
            r["current_norm"] = norm_r
            f["overlap"] = overlap
            f["rev_error"] = rev_error
            f["overlap_neg"] = overlap_neg
            f["edges_fwd"] = len(f_map)
            f["edges_rev"] = len(r_map)
            f["shared_edges"] = shared
            f["shared_ratio"] = shared_ratio
            r["overlap"] = overlap
            r["rev_error"] = rev_error
            r["overlap_neg"] = overlap_neg
            r["edges_fwd"] = len(f_map)
            r["edges_rev"] = len(r_map)
            r["shared_edges"] = shared
            r["shared_ratio"] = shared_ratio
            f_ci = float(f.get("strobe_current_l2_ci_half", 0.0))
            r_ci = float(r.get("strobe_current_l2_ci_half", 0.0))
            f_acc = float(f.get("acceptedFracWindowMean", 0.0))
            r_acc = float(r.get("acceptedFracWindowMean", 0.0))
            if seed == seeds[0]:
                print(
                    "REVERSAL_METRICS "
                    f"seed={seed} shared_ratio={shared_ratio:.6g} "
                    f"overlap={overlap:.6g} overlap_neg={overlap_neg:.6g} "
                    f"windows={windows_used}"
                )
            if min(norm_f, norm_r) < args.min_current_l2:
                f["status"] = "FAIL_WEAK_CURRENT"
                r["status"] = "FAIL_WEAK_CURRENT"
            elif overlap <= args.overlap_thresh and rev_error <= args.rev_error_thresh and max(f_ci, r_ci) <= args.ci_thresh and f_acc >= args.accept_min and r_acc >= args.accept_min:
                f["status"] = "PASS_EARLY"
                r["status"] = "PASS_EARLY"
            else:
                f["status"] = "FAIL_REVERSAL"
                r["status"] = "FAIL_REVERSAL"

    rewrite_case(proto_case, proto_fwd_rows)
    if args.metric_mode == "reversal_overlap":
        rewrite_case("protocol_p3_rev", proto_rev_rows)

    print("STATUS_COUNTS control_p3_off:", {r["status"]: 1 for r in control_rows})
    print(f"STATUS_COUNTS {proto_case}:", {r['status']: 1 for r in proto_fwd_rows})
    if args.metric_mode == "reversal_overlap":
        print("STATUS_COUNTS protocol_p3_rev:", {r["status"]: 1 for r in proto_rev_rows})

    report_path = out_dir / "PHASE3_P3_PUMPING_REPORT.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 3 P3 pumping v4\n\n")
        fh.write(f"preset: {args.preset}\neta={args.eta}\n\n")
        fh.write("strobe_rate_metric: per_proposal\n\n")
        control_map = {r["seed"]: r for r in control_rows}
        fwd_map = {r["seed"]: r for r in proto_fwd_rows}
        if args.metric_mode == "diff_vs_control":
            fh.write("| seed | control_status | protocol_status | control_current_l2 | protocol_current_l2 | rel_change | ")
            fh.write("control_symgap | protocol_symgap | accept_control | accept_protocol |\n")
            fh.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for seed in seeds:
                c = control_map.get(seed, {})
                f = fwd_map.get(seed, {})
                fh.write(
                    f"| {seed} | {c.get('status','')} | {f.get('status','')} | "
                    f"{c.get('strobe_current_l2_mean_last_m',0.0):.6g} | {f.get('strobe_current_l2_mean_last_m',0.0):.6g} | "
                    f"{f.get('rel_change',0.0):.6g} | {c.get('strobe_symgap_mean_last_m',0.0):.6g} | "
                    f"{f.get('strobe_symgap_mean_last_m',0.0):.6g} | {c.get('acceptedFracWindowMean',0.0):.6g} | "
                    f"{f.get('acceptedFracWindowMean',0.0):.6g} |\n"
                )
        else:
            rev_map = {r["seed"]: r for r in proto_rev_rows}
            fh.write("| seed | control_status | fwd_status | rev_status | ")
            fh.write("control_symgap | fwd_symgap | rev_symgap | ")
            fh.write("control_current_l2 | fwd_current_l2 | rev_current_l2 | ")
            fh.write("norm_f | norm_r | overlap | rev_error | ")
            fh.write("fwd_edge | j_fwd |\n")
            fh.write("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n")
            for seed in seeds:
                c = control_map.get(seed, {})
                f = fwd_map.get(seed, {})
                r = rev_map.get(seed, {})
                edge_str = f.get("top_edge", "")
                j_fwd = float(f.get("top_edge_j", 0.0))
                norm_f = float(f.get("current_norm", 0.0))
                norm_r = float(r.get("current_norm", 0.0))
                overlap = float(f.get("overlap", 0.0))
                rev_error = float(f.get("rev_error", 0.0))
                fh.write(
                    f"| {seed} | {c.get('status','')} | {f.get('status','')} | {r.get('status','')} | "
                    f"{c.get('strobe_symgap_mean_last_m',0.0):.6g} | {f.get('strobe_symgap_mean_last_m',0.0):.6g} | "
                    f"{r.get('strobe_symgap_mean_last_m',0.0):.6g} | "
                    f"{c.get('strobe_current_l2_mean_last_m',0.0):.6g} | {f.get('strobe_current_l2_mean_last_m',0.0):.6g} | "
                    f"{r.get('strobe_current_l2_mean_last_m',0.0):.6g} | "
                    f"{norm_f:.6g} | {norm_r:.6g} | {overlap:.6g} | {rev_error:.6g} | "
                    f"{edge_str} | {j_fwd:.6g} |\n"
                )
            fh.write("\n## Top currents (fwd vs rev)\n\n")
            for seed in seeds:
                f = fwd_map.get(seed, {})
                r = rev_map.get(seed, {})
                f_map = f.get("_current_map", {})
                r_map = r.get("_current_map", {})
                if not f_map:
                    continue
                top = sorted(f_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
                fh.write(f"### seed {seed}\n\n")
                for edge, j in top:
                    j_rev = float(r_map.get(edge, 0.0)) if r_map else 0.0
                    fh.write(f"{edge}: j_fwd={j:.6g} j_rev={j_rev:.6g}\n")
                fh.write("\n")


if __name__ == "__main__":
    main()
