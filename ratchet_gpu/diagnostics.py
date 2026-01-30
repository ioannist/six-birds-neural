from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import torch

from .lattice import gather_neighbors
from .state import State


def ep_totals(ep_ledger: Dict[str, Any]) -> Dict[str, Any]:
    ep_total_exact = float(
        ep_ledger.get(
            "ep_total_exact",
            ep_ledger.get("ep_total_micro", ep_ledger.get("epMicroTotal", 0.0)),
        )
    )
    ep_total_micro = float(
        ep_ledger.get("ep_total_micro", ep_ledger.get("epMicroTotal", ep_total_exact))
    )
    ep_by_kernel = ep_ledger.get("ep_by_kernel", ep_ledger.get("epByKernel", {}))
    ep_by_kernel = {str(k): float(v) for k, v in ep_by_kernel.items()}
    window_proposals = int(
        ep_ledger.get("window_proposals", ep_ledger.get("window_steps", 0))
    )
    window_steps = int(ep_ledger.get("window_steps", window_proposals))
    return {
        "ep_total_exact": ep_total_exact,
        "ep_total_micro": ep_total_micro,
        "ep_by_kernel": ep_by_kernel,
        "window_proposals": window_proposals,
        "window_steps": window_steps,
        "window_proposals_by_kernel": ep_ledger.get("window_proposals_by_kernel", {}),
        "window_accepted_by_kernel": ep_ledger.get("window_accepted_by_kernel", {}),
        "window_accept_frac_by_kernel": ep_ledger.get("window_accept_frac_by_kernel", {}),
        "strobe_rate_window": ep_ledger.get("strobe_rate_window", 0.0),
        "strobe_transitions_window": ep_ledger.get("strobe_transitions_window", 0),
        "strobe_cycle_len": ep_ledger.get("strobe_cycle_len", 0),
        "strobe_stride": ep_ledger.get("strobe_stride", 0),
        "strobe_unique_states_window": ep_ledger.get("strobe_unique_states_window", 0),
        "strobe_bidirectional_edges_window": ep_ledger.get("strobe_bidirectional_edges_window", 0),
        "strobe_signature": ep_ledger.get("strobe_signature", ""),
        "strobe_top_states_window": ep_ledger.get("strobe_top_states_window", []),
        "strobe_symgap_window": ep_ledger.get("strobe_symgap_window", 0.0),
        "strobe_current_l2_window": ep_ledger.get("strobe_current_l2_window", 0.0),
        "strobe_currents_window": ep_ledger.get("strobe_currents_window", []),
        "strobe_currents_count_window": ep_ledger.get("strobe_currents_count_window", 0),
        "strobe_current_map_items_window": ep_ledger.get(
            "strobe_current_map_items_window", ep_ledger.get("strobe_currents_window", [])
        ),
        "strobe_current_map_items_count_window": ep_ledger.get(
            "strobe_current_map_items_count_window",
            ep_ledger.get("strobe_currents_count_window", 0),
        ),
        "strobe_signature_effective_window": ep_ledger.get(
            "strobe_signature_effective_window", ""
        ),
        "strobe_signature_effective_total": ep_ledger.get(
            "strobe_signature_effective_total", ""
        ),
    }


def ep_windowed(
    prev_totals: Dict[str, Any],
    curr_totals: Dict[str, Any],
    prev_step: int,
    curr_step: int,
) -> Dict[str, Any]:
    window_proposals = int(
        curr_totals.get("window_proposals", curr_totals.get("window_steps", 0))
    )
    window_props_by_kernel = curr_totals.get("window_proposals_by_kernel", {})
    denom = max(1, window_proposals)
    rate_exact = (curr_totals["ep_total_exact"] - prev_totals["ep_total_exact"]) / denom

    prev_by_kernel = prev_totals.get("ep_by_kernel", {})
    curr_by_kernel = curr_totals.get("ep_by_kernel", {})
    all_keys = set(prev_by_kernel) | set(curr_by_kernel)
    rate_by_kernel = {
        key: (curr_by_kernel.get(key, 0.0) - prev_by_kernel.get(key, 0.0)) / denom
        for key in all_keys
    }

    rate_by_kernel_proposal = {}
    for key in all_keys:
        k_props = int(window_props_by_kernel.get(key, 0))
        k_denom = max(1, k_props)
        rate_by_kernel_proposal[key] = (
            curr_by_kernel.get(key, 0.0) - prev_by_kernel.get(key, 0.0)
        ) / k_denom

    return {
        "ep_rate_exact_window": float(rate_exact),
        "ep_rate_by_kernel_window": {k: float(v) for k, v in rate_by_kernel.items()},
        "ep_rate_by_kernel_proposal_window": {
            k: float(v) for k, v in rate_by_kernel_proposal.items()
        },
    }


def cross_pred_sigma(state: State) -> torch.Tensor:
    pred = torch.zeros(
        (state.layers - 1, state.N), dtype=torch.float64, device=state.device
    )
    if state.layers <= 1 or state.K_K == 0 or state.params.B_k == 0:
        return pred

    for layer in range(1, state.layers):
        sigma_lower = state.sigma[layer - 1].to(dtype=torch.float64)
        neighbors = gather_neighbors(sigma_lower, state.lattice, state.R_K)
        weights = state.K[layer - 1].to(dtype=torch.float64)
        pred[layer - 1] = (weights * neighbors).sum(dim=-1) / float(state.params.B_k)
    return pred


def cross_mismatch(state: State, p: int = 1) -> Dict[str, Any]:
    if state.layers <= 1 or state.K_K == 0 or state.params.B_k == 0:
        return {
            "mismatch_abs_by_interface": [],
            "mismatch_abs_mean": None,
        }

    pred = cross_pred_sigma(state)
    sigma_upper = state.sigma[1:].to(dtype=torch.float64)
    diff = sigma_upper - pred

    if p == 2:
        vals = diff.abs() ** 2
    else:
        vals = diff.abs()

    by_interface = vals.mean(dim=1)
    mismatch_abs_by_interface = [float(v.item()) for v in by_interface]
    mismatch_abs_mean = float(vals.mean().item())

    result = {
        "mismatch_abs_by_interface": mismatch_abs_by_interface,
        "mismatch_abs_mean": mismatch_abs_mean,
    }

    if p == 2:
        result["mismatch_l2_by_interface"] = mismatch_abs_by_interface
        result["mismatch_l2_mean"] = mismatch_abs_mean

    return result


def k_kernel_proxies(state: State, eps: float = 1e-12) -> Dict[str, Any]:
    if state.layers <= 1 or state.K_K == 0 or state.params.B_k == 0:
        empty = [0.0 for _ in range(state.layers - 1)]
        return {
            "k_entropy_mean_by_interface": [],
            "k_r2_mean_by_interface": [],
            "k_coh_mean_by_interface": [],
            "k_entropy_mean": None,
            "k_r2_mean": None,
            "k_coh_mean": None,
        }

    r2 = (state.R_K.to(dtype=torch.float64) ** 2).sum(dim=-1)
    axis_offsets = []
    for axis in range(state.lattice.d):
        vec = [0] * state.lattice.d
        vec[axis] = 1
        axis_offsets.append(vec)
        vec = [0] * state.lattice.d
        vec[axis] = -1
        axis_offsets.append(vec)
    axis_offsets_t = torch.tensor(
        axis_offsets, dtype=torch.long, device=state.device
    )

    entropy_means = []
    r2_means = []
    coh_means = []

    for layer in range(1, state.layers):
        K_layer = state.K[layer - 1].to(dtype=torch.float64)
        probs = K_layer / float(state.params.B_k)

        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
        entropy_means.append(float(entropy.mean().item()))

        r2_vals = (probs * r2).sum(dim=-1)
        r2_means.append(float(r2_vals.mean().item()))

        neighbors = gather_neighbors(probs, state.lattice, axis_offsets_t)
        diff = (neighbors - probs[:, None, :]).abs().sum(dim=-1)
        coh_means.append(float(diff.mean().item()))

    return {
        "k_entropy_mean_by_interface": entropy_means,
        "k_r2_mean_by_interface": r2_means,
        "k_coh_mean_by_interface": coh_means,
        "k_entropy_mean": float(sum(entropy_means) / len(entropy_means)),
        "k_r2_mean": float(sum(r2_means) / len(r2_means)),
        "k_coh_mean": float(sum(coh_means) / len(coh_means)),
    }


def compute_snapshot(
    state: State,
    step: int,
    ep_ledger: Dict[str, Any],
    accepted_frac: float | None,
    prev_diag_state: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    totals = ep_totals(ep_ledger)

    if prev_diag_state is None:
        prev_totals = {"ep_total_exact": 0.0, "ep_by_kernel": {}}
        prev_step = 0
    else:
        prev_totals = prev_diag_state.get("prev_totals", {"ep_total_exact": 0.0})
        prev_step = int(prev_diag_state.get("prev_step", 0))

    rates = ep_windowed(prev_totals, totals, prev_step, step)
    mismatch_metrics = cross_mismatch(state, p=1)
    proxies = k_kernel_proxies(state)

    snapshot = {
        "step": int(step),
        "meta_layers": int(state.layers),
        "shape": list(state.lattice.shape),
        "device": str(state.device),
        "ep_total_exact": totals["ep_total_exact"],
        "ep_total_micro": totals["ep_total_micro"],
        "ep_by_kernel": totals["ep_by_kernel"],
        "ep_rate_exact_window": rates["ep_rate_exact_window"],
        "ep_rate_by_kernel_window": rates["ep_rate_by_kernel_window"],
        "ep_rate_by_kernel_proposal_window": rates["ep_rate_by_kernel_proposal_window"],
        "window_proposals": totals.get("window_proposals", 0),
        "window_steps": totals.get("window_steps", 0),
        "window_proposals_by_kernel": totals.get("window_proposals_by_kernel", {}),
        "window_accepted_by_kernel": totals.get("window_accepted_by_kernel", {}),
        "window_accept_frac_by_kernel": totals.get("window_accept_frac_by_kernel", {}),
        "strobe_rate_window": totals.get("strobe_rate_window", 0.0),
        "strobe_transitions_window": totals.get("strobe_transitions_window", 0),
        "strobe_cycle_len": totals.get("strobe_cycle_len", 0),
        "strobe_stride": totals.get("strobe_stride", 0),
        "strobe_unique_states_window": totals.get("strobe_unique_states_window", 0),
        "strobe_bidirectional_edges_window": totals.get("strobe_bidirectional_edges_window", 0),
        "strobe_signature": totals.get("strobe_signature", ""),
        "strobe_top_states_window": totals.get("strobe_top_states_window", []),
        "strobe_symgap_window": totals.get("strobe_symgap_window", 0.0),
        "strobe_current_l2_window": totals.get("strobe_current_l2_window", 0.0),
        "strobe_currents_window": totals.get("strobe_currents_window", []),
        "strobe_currents_count_window": totals.get("strobe_currents_count_window", 0),
        "strobe_current_map_items_window": totals.get(
            "strobe_current_map_items_window", totals.get("strobe_currents_window", [])
        ),
        "strobe_current_map_items_count_window": totals.get(
            "strobe_current_map_items_count_window",
            totals.get("strobe_currents_count_window", 0),
        ),
        "strobe_signature_effective_window": totals.get(
            "strobe_signature_effective_window", ""
        ),
        "strobe_signature_effective_total": totals.get(
            "strobe_signature_effective_total", ""
        ),
    }

    if accepted_frac is not None:
        snapshot["acceptedFrac"] = float(accepted_frac)

    snapshot.update(mismatch_metrics)
    snapshot.update(proxies)

    new_state = {"prev_totals": totals, "prev_step": int(step)}
    return snapshot, new_state


def to_json_line(snapshot: Dict[str, Any]) -> str:
    return json.dumps(snapshot, sort_keys=True)
