from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch

from .energy import mismatch_mean
from .ep import EPTracker, StrobeTracker
from .kernels import (
    excitable_step_color,
    k_local_exchange,
    k_neighbor_trade,
    k_p5_exchange,
    n_flip,
    s_step,
    spin_flip_color,
    w_local_exchange,
    w_neighbor_exchange,
)
from .params import Params
from .state import State

KernelFn = Callable[[State, torch.Generator], tuple[bool, float]]


def _kernel_map() -> dict[str, KernelFn]:
    return {
        "spin_flip_color0": lambda st, gen: spin_flip_color(st, 0, gen),
        "spin_flip_color1": lambda st, gen: spin_flip_color(st, 1, gen),
        "excitable_color0": lambda st, gen: excitable_step_color(st, 0, gen),
        "excitable_color1": lambda st, gen: excitable_step_color(st, 1, gen),
        "n_flip": n_flip,
        "s_step": s_step,
        "w_local": w_local_exchange,
        "k_local": k_local_exchange,
        "k_neighbor_trade": k_neighbor_trade,
        "k_p5_exchange": k_p5_exchange,
        "w_neighbor": w_neighbor_exchange,
    }


def _cycle_list() -> list[str]:
    return [
        "spin_flip_color0",
        "w_local",
        "s_step",
        "k_p5_exchange",
        "k_local",
        "n_flip",
        "spin_flip_color1",
        "w_neighbor",
        "k_neighbor_trade",
    ]


def run_sim(
    params: Params,
    seed: int | None,
    steps: int,
    report_every: int | None = None,
    device: torch.device | str | None = None,
    report_callback: Callable[[State, int, dict[str, float], float], None] | None = None,
    stop_callback: Callable[[State, int, dict[str, float], float], bool] | None = None,
    protocol_cycle: list[str] | None = None,
    initial_state: State | None = None,
    initial_rng_state: torch.Tensor | None = None,
    return_state: bool = False,
) -> dict[str, float]:
    if device is not None and initial_state is not None:
        raise ValueError("device override is not supported when initial_state is provided")
    if device is not None:
        params = replace(params, device=device)
    if report_every is None:
        report_every = params.report_every

    state = initial_state or State.initialize(params, seed=seed)
    generator = torch.Generator(device=state.device)
    if initial_rng_state is not None:
        generator.set_state(initial_rng_state)
    elif seed is not None:
        generator.manual_seed(seed + 1)

    tracker = EPTracker(beta=params.beta)
    strobe_enabled = getattr(params, "strobe_on", params.p3_on)
    strobe_sig = getattr(params, "strobe_signature", "mag_s")
    strobe_total = StrobeTracker(signature=strobe_sig) if strobe_enabled else None
    strobe_window = StrobeTracker(signature=strobe_sig) if strobe_enabled else None
    kernel_map = _kernel_map()
    kernel_names = list(kernel_map)

    cycle = protocol_cycle if (params.p3_on and protocol_cycle) else (_cycle_list() if params.p3_on else [])
    cycle_len = len(cycle)
    strobe_cycle_len = cycle_len if cycle_len else len(protocol_cycle or _cycle_list())
    strobe_stride = strobe_cycle_len if strobe_cycle_len else max(1, (report_every or 1) // 4)

    kernel_stats = {name: {"proposals": 0, "accepted": 0} for name in kernel_map}
    window_stats = {name: {"proposals": 0, "accepted": 0} for name in kernel_map}
    ep_window_by_kernel = {name: 0.0 for name in kernel_map}
    ep_total_by_kernel = {name: 0.0 for name in kernel_map}
    last_rate_by_kernel = {name: 0.0 for name in kernel_map}
    last_accept_window_by_kernel = {name: 0.0 for name in kernel_map}
    last_window_proposals_by_kernel = {name: 0 for name in kernel_map}
    last_window_accepted_by_kernel = {name: 0 for name in kernel_map}

    weights = None
    if not params.p3_on:
        weight_vals = torch.tensor(
            [params.kernel_weights.get(name, 0.0) for name in kernel_names],
            dtype=torch.float32,
            device=state.device,
        )
        if torch.any(weight_vals < 0):
            raise ValueError("kernel_weights must be non-negative")
        if torch.all(weight_vals == 0):
            weight_vals = torch.ones_like(weight_vals)
        weights = weight_vals / weight_vals.sum()

    last_rate = 0.0
    cycle_len = len(cycle)

    last_report_step = 0
    stop_early = False
    for step in range(1, steps + 1):
        if params.p3_on:
            name = cycle[(step - 1) % cycle_len]
        else:
            idx = int(torch.multinomial(weights, 1, generator=generator).item())
            name = kernel_names[idx]

        result = kernel_map[name](state, generator)
        if len(result) == 2:
            accepted, ep_inc = result
            proposals = 1
            accepted_count = 1 if accepted else 0
        else:
            accepted, ep_inc, proposals, accepted_count = result

        tracker.record(accepted, ep_inc, proposals=proposals, accepted_count=accepted_count)

        kernel_stats[name]["proposals"] += int(proposals)
        kernel_stats[name]["accepted"] += int(accepted_count)
        window_stats[name]["proposals"] += int(proposals)
        window_stats[name]["accepted"] += int(accepted_count)
        if accepted_count:
            ep_window_by_kernel[name] += float(ep_inc)
            ep_total_by_kernel[name] += float(ep_inc)

        if strobe_enabled and step % strobe_stride == 0:
            strobe_total.record_state(state)  # type: ignore[union-attr]
            strobe_window.record_state(state)  # type: ignore[union-attr]

        if report_every and step % report_every == 0:
            last_rate = tracker.ep_micro_window / max(1, tracker.window_steps)
            window_proposals_total = 0
            window_accepted_total = 0
            for k in kernel_names:
                last_rate_by_kernel[k] = ep_window_by_kernel[k] / max(1, tracker.window_steps)
                proposals = window_stats[k]["proposals"]
                accepted_k = window_stats[k]["accepted"]
                last_window_proposals_by_kernel[k] = proposals
                last_window_accepted_by_kernel[k] = accepted_k
                last_accept_window_by_kernel[k] = (
                    accepted_k / proposals if proposals else 0.0
                )
                window_proposals_total += proposals
                window_accepted_total += accepted_k
                ep_window_by_kernel[k] = 0.0
                window_stats[k]["proposals"] = 0
                window_stats[k]["accepted"] = 0
            strobe_rate_window = 0.0
            strobe_transitions_window = 0
            strobe_unique_window = 0
            strobe_edges_window = 0
            strobe_top_states = []
            strobe_symgap_window = 0.0
            strobe_current_l2_window = 0.0
            strobe_currents_count_window = 0
            strobe_currents_window = []
            if strobe_enabled and strobe_window is not None:
                strobe_rate_window = strobe_window.ep_rate()
                strobe_transitions_window = strobe_window.total
                strobe_unique_window = strobe_window.unique_states_count()
                strobe_edges_window = strobe_window.bidirectional_edge_count()
                strobe_top_states = [
                    {"state": list(state_id), "count": count}
                    for state_id, count in strobe_window.top_states(10)
                ]
                strobe_symgap_window = strobe_window.symgap()
                strobe_current_l2_window = strobe_window.current_l2()
                current_map = strobe_window.current_map()
                strobe_currents_count_window = len(current_map)
                strobe_currents_window = strobe_window.current_items(None)
                strobe_window = StrobeTracker(
                    signature=strobe_sig,
                    bins_sigma=strobe_window.bins_sigma,
                    bins_s=strobe_window.bins_s,
                    bins_w=strobe_window.bins_w,
                )

            ep_ledger = {
                "ep_total_exact": tracker.ep_micro_total,
                "ep_total_micro": tracker.ep_micro_total,
                "ep_by_kernel": dict(ep_total_by_kernel),
                "window_steps": int(tracker.window_steps),
                "window_proposals": int(window_proposals_total),
                "window_accepted": int(window_accepted_total),
                "window_accept_frac": float(
                    window_accepted_total / window_proposals_total
                    if window_proposals_total
                    else 0.0
                ),
                "strobe_rate_window": float(strobe_rate_window),
                "strobe_transitions_window": int(strobe_transitions_window),
                "strobe_cycle_len": int(strobe_cycle_len),
                "strobe_stride": int(strobe_stride),
                "strobe_unique_states_window": int(strobe_unique_window),
                "strobe_bidirectional_edges_window": int(strobe_edges_window),
                "strobe_signature": str(strobe_sig),
                "strobe_top_states_window": strobe_top_states,
                "strobe_symgap_window": float(strobe_symgap_window),
                "strobe_current_l2_window": float(strobe_current_l2_window),
                "strobe_currents_window": strobe_currents_window,
                "strobe_currents_count_window": int(strobe_currents_count_window),
                "strobe_current_map_items_window": strobe_currents_window,
                "strobe_current_map_items_count_window": int(strobe_currents_count_window),
                "strobe_signature_effective_window": str(strobe_window.signature)
                if strobe_window is not None
                else "",
                "strobe_signature_effective_total": str(strobe_total.signature)
                if strobe_total is not None
                else "",
            }
            ep_ledger["window_proposals_by_kernel"] = {
                name: int(last_window_proposals_by_kernel[name]) for name in kernel_names
            }
            ep_ledger["window_accepted_by_kernel"] = {
                name: int(last_window_accepted_by_kernel[name]) for name in kernel_names
            }
            ep_ledger["window_accept_frac_by_kernel"] = {
                name: float(last_accept_window_by_kernel[name]) for name in kernel_names
            }
            if report_callback is not None:
                report_callback(state, step, ep_ledger, tracker.accepted_frac)
            if stop_callback is not None and stop_callback(
                state, step, ep_ledger, tracker.accepted_frac
            ):
                stop_early = True
            tracker.reset_window()
            last_report_step = step
            if stop_early:
                break

    if tracker.window_steps:
        last_rate = tracker.ep_micro_window / max(1, tracker.window_steps)
        for k in kernel_names:
            last_rate_by_kernel[k] = ep_window_by_kernel[k] / max(1, tracker.window_steps)
            proposals = window_stats[k]["proposals"]
            accepted_k = window_stats[k]["accepted"]
            last_window_proposals_by_kernel[k] = proposals
            last_window_accepted_by_kernel[k] = accepted_k
            last_accept_window_by_kernel[k] = (
                accepted_k / proposals if proposals else 0.0
            )

    if report_callback is not None and steps != last_report_step and not stop_early:
        window_proposals_total = sum(stats["proposals"] for stats in window_stats.values())
        window_accepted_total = sum(stats["accepted"] for stats in window_stats.values())
        strobe_rate_window = 0.0
        strobe_transitions_window = 0
        strobe_unique_window = 0
        strobe_edges_window = 0
        strobe_top_states = []
        strobe_symgap_window = 0.0
        strobe_current_l2_window = 0.0
        strobe_currents_count_window = 0
        strobe_currents_window = []
        if strobe_enabled and strobe_window is not None:
            strobe_rate_window = strobe_window.ep_rate()
            strobe_transitions_window = strobe_window.total
            strobe_unique_window = strobe_window.unique_states_count()
            strobe_edges_window = strobe_window.bidirectional_edge_count()
            strobe_top_states = [
                {"state": list(state_id), "count": count}
                for state_id, count in strobe_window.top_states(10)
            ]
            strobe_symgap_window = strobe_window.symgap()
            strobe_current_l2_window = strobe_window.current_l2()
            current_map = strobe_window.current_map()
            strobe_currents_count_window = len(current_map)
            strobe_currents_window = strobe_window.current_items(None)
        ep_ledger = {
            "ep_total_exact": tracker.ep_micro_total,
            "ep_total_micro": tracker.ep_micro_total,
            "ep_by_kernel": dict(ep_total_by_kernel),
            "window_steps": int(tracker.window_steps),
            "window_proposals": int(window_proposals_total),
            "window_accepted": int(window_accepted_total),
            "window_accept_frac": float(
                window_accepted_total / window_proposals_total if window_proposals_total else 0.0
            ),
            "strobe_rate_window": float(strobe_rate_window),
            "strobe_transitions_window": int(strobe_transitions_window),
            "strobe_cycle_len": int(strobe_cycle_len),
            "strobe_stride": int(strobe_stride),
            "strobe_unique_states_window": int(strobe_unique_window),
            "strobe_bidirectional_edges_window": int(strobe_edges_window),
            "strobe_signature": str(strobe_sig),
            "strobe_top_states_window": strobe_top_states,
            "strobe_symgap_window": float(strobe_symgap_window),
            "strobe_current_l2_window": float(strobe_current_l2_window),
            "strobe_currents_window": strobe_currents_window,
            "strobe_currents_count_window": int(strobe_currents_count_window),
            "strobe_current_map_items_window": strobe_currents_window,
            "strobe_current_map_items_count_window": int(strobe_currents_count_window),
            "strobe_signature_effective_window": str(strobe_window.signature)
            if strobe_window is not None
            else "",
            "strobe_signature_effective_total": str(strobe_total.signature)
            if strobe_total is not None
            else "",
        }
        ep_ledger["window_proposals_by_kernel"] = {
            name: int(last_window_proposals_by_kernel[name]) for name in kernel_names
        }
        ep_ledger["window_accepted_by_kernel"] = {
            name: int(last_window_accepted_by_kernel[name]) for name in kernel_names
        }
        ep_ledger["window_accept_frac_by_kernel"] = {
            name: float(last_accept_window_by_kernel[name]) for name in kernel_names
        }
        report_callback(state, steps, ep_ledger, tracker.accepted_frac)

    strobe_rate = strobe_total.ep_rate() if strobe_enabled and strobe_total is not None else 0.0
    accepted_frac = tracker.accepted_frac
    mismatch_end = mismatch_mean(state)

    summary = {
        "epMicroRateWindowLast": float(last_rate),
        "acceptedFrac": float(accepted_frac),
        "epStrobeRate": float(strobe_rate),
        "mismatchMean": float(mismatch_end),
    }

    for name, stats in kernel_stats.items():
        proposals = stats["proposals"]
        accepted = stats["accepted"]
        summary[f"accept_{name}"] = accepted / proposals if proposals else 0.0
        summary[f"epMicroRateWindowLast_{name}"] = float(last_rate_by_kernel[name])
        summary[f"epMicroTotal_{name}"] = float(ep_total_by_kernel[name])
        summary[f"acceptWindow_{name}"] = float(last_accept_window_by_kernel[name])
        summary[f"proposalsWindow_{name}"] = int(last_window_proposals_by_kernel[name])
        summary[f"acceptedWindow_{name}"] = int(last_window_accepted_by_kernel[name])

    if return_state:
        summary["state"] = state
        summary["rng_state"] = generator.get_state()

    return summary


def run_null(
    params: Params,
    seed: int | None,
    steps: int,
    report_every: int,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    params = replace(
        params,
        p3_on=False,
        p6_on=False,
        kernel_weights={"k_local": 1.0},
    )
    summary = run_sim(params, seed, steps, report_every, device=device)
    return {
        "epExactRateWindowLast": summary["epMicroRateWindowLast"],
        "acceptedFrac": summary["acceptedFrac"],
    }
