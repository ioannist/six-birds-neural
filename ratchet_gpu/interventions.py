from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple

import torch

from .state import State
from .params import Params


def parse_rect(rect: str, shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*", rect)
    if not match:
        raise ValueError(f"Invalid rect syntax: {rect}")
    r0, r1, c0, c1 = (int(match.group(i)) for i in range(1, 5))
    H, W = shape
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W or r0 >= r1 or c0 >= c1:
        raise ValueError("Rect bounds out of range")
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[r0:r1, c0:c1] = True
    rows = torch.arange(r0, r1, dtype=torch.long)
    cols = torch.arange(c0, c1, dtype=torch.long)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    flat_idx = (grid_r * W + grid_c).reshape(-1)
    return mask, flat_idx


def _resolve_layers(layers: str | Sequence[int] | None, total: int) -> List[int]:
    if layers is None or layers == "all":
        return list(range(total))
    if isinstance(layers, str):
        return [int(x) for x in layers.split(",") if x.strip()]
    return [int(x) for x in layers]


def _resolve_interfaces(interfaces: str | Sequence[int] | None, total: int) -> List[int]:
    if interfaces is None or interfaces == "all":
        return list(range(total))
    if isinstance(interfaces, str):
        return [int(x) for x in interfaces.split(",") if x.strip()]
    return [int(x) for x in interfaces]


def _counts_from_weights(
    weights: torch.Tensor, total: int, cap: int
) -> torch.Tensor:
    if total <= 0:
        return torch.zeros_like(weights, dtype=torch.int64)
    w = weights.to(dtype=torch.float64).clamp(min=0)
    if float(w.sum().item()) <= 0.0:
        w = torch.ones_like(w, dtype=torch.float64)
    w = w / w.sum()
    target = w * float(total)
    counts = torch.floor(target).to(dtype=torch.int64)
    counts = torch.minimum(counts, torch.tensor(cap, dtype=torch.int64, device=counts.device))
    remainder = int(total - int(counts.sum().item()))
    frac = target - counts.to(dtype=torch.float64)
    if remainder > 0:
        order = torch.argsort(frac, descending=True)
        for idx in order.tolist():
            if remainder <= 0:
                break
            if int(counts[idx].item()) < cap:
                counts[idx] += 1
                remainder -= 1
    if remainder < 0:
        order = torch.argsort(frac, descending=False)
        for idx in order.tolist():
            if remainder >= 0:
                break
            if int(counts[idx].item()) > 0:
                counts[idx] -= 1
                remainder += 1
    if int(counts.sum().item()) != total:
        diff = total - int(counts.sum().item())
        if diff > 0:
            for idx in torch.argsort(frac, descending=True).tolist():
                if diff <= 0:
                    break
                if int(counts[idx].item()) < cap:
                    counts[idx] += 1
                    diff -= 1
        elif diff < 0:
            for idx in torch.argsort(frac, descending=False).tolist():
                if diff >= 0:
                    break
                if int(counts[idx].item()) > 0:
                    counts[idx] -= 1
                    diff += 1
    return counts


def apply_k_redistribute_uniform_in_region(
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    interfaces: str | Sequence[int] | None = "all",
    rng=None,
) -> dict:
    interface_list = _resolve_interfaces(interfaces, state.layers - 1)
    if not interface_list or state.K_K == 0:
        return {"sites": 0, "interfaces": []}
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    generator = rng or torch.Generator(device=device)
    sites = int(idx.numel())
    total_entries = state.K_K * params.l_k
    if params.B_k > total_entries:
        raise ValueError("B_k exceeds per-site capacity for K")
    for interface in interface_list:
        K_layer = state.K[interface]
        for site in idx.tolist():
            slots = torch.randperm(total_entries, generator=generator, device=device)[:params.B_k]
            entry_idx = slots // params.l_k
            counts = torch.bincount(entry_idx, minlength=state.K_K).to(dtype=K_layer.dtype)
            K_layer[site] = counts
    return {"sites": sites, "interfaces": interface_list}


def apply_k_redistribute_axis_bias_in_region(
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    interfaces: str | Sequence[int] | None = "all",
    axis: int = 0,
    rng=None,
) -> dict:
    interface_list = _resolve_interfaces(interfaces, state.layers - 1)
    if not interface_list or state.K_K == 0:
        return {"sites": 0, "interfaces": []}
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    axis = int(axis)
    generator = rng or torch.Generator(device=device)
    offsets = state.R_K.to(device=device, dtype=torch.float64)
    axis = max(0, min(axis, offsets.shape[1] - 1))
    weights = offsets[:, axis].abs()
    sites = int(idx.numel())
    for interface in interface_list:
        K_layer = state.K[interface]
        for site in idx.tolist():
            counts = _counts_from_weights(weights, int(params.B_k), int(params.l_k))
            # Add a small random tie-breaker to avoid uniform artifacts.
            if torch.all(counts == counts[0]) and params.B_k > 0:
                noise = torch.rand_like(weights, generator=generator)
                counts = _counts_from_weights(weights + 1e-6 * noise, int(params.B_k), int(params.l_k))
            K_layer[site] = counts.to(dtype=K_layer.dtype)
    return {"sites": sites, "interfaces": interface_list, "axis": axis}


def apply_k_redistribute_axis_bias_random_in_region(
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    interfaces: str | Sequence[int] | None = "all",
    rng=None,
) -> dict:
    interface_list = _resolve_interfaces(interfaces, state.layers - 1)
    if not interface_list or state.K_K == 0:
        return {"sites": 0, "interfaces": []}
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    generator = rng or torch.Generator(device=device)
    offsets = state.R_K.to(device=device, dtype=torch.float64)
    axes = list(range(offsets.shape[1]))
    sites = int(idx.numel())
    for interface in interface_list:
        K_layer = state.K[interface]
        for site in idx.tolist():
            axis = int(axes[int(torch.randint(0, len(axes), (1,), generator=generator, device=device).item())])
            weights = offsets[:, axis].abs()
            counts = _counts_from_weights(weights, int(params.B_k), int(params.l_k))
            K_layer[site] = counts.to(dtype=K_layer.dtype)
    return {"sites": sites, "interfaces": interface_list, "axis": "random"}


def apply_k_redistribute_radial_inward_in_ring(
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    center: Tuple[float, float],
    interfaces: str | Sequence[int] | None = "all",
    strength: float = 1.0,
    rng=None,
) -> dict:
    interface_list = _resolve_interfaces(interfaces, state.layers - 1)
    if not interface_list or state.K_K == 0:
        return {"sites": 0, "interfaces": []}
    if state.lattice.d != 2:
        raise ValueError("radial redistribution expects 2D lattice")
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    coords = state.lattice.index_to_coord(idx).to(dtype=torch.float64)
    offsets = state.R_K.to(device=device, dtype=torch.float64)
    generator = rng or torch.Generator(device=device)
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))
    sites = int(idx.numel())
    for interface in interface_list:
        K_layer = state.K[interface]
        for i, site in enumerate(idx.tolist()):
            dy = float(center[0]) - float(coords[i, 0].item())
            dx = float(center[1]) - float(coords[i, 1].item())
            norm = math.sqrt(dx * dx + dy * dy)
            if norm <= 0:
                weights = torch.ones_like(offsets[:, 0], dtype=torch.float64)
            else:
                unit_y = dy / norm
                unit_x = dx / norm
                align = offsets[:, 0] * unit_y + offsets[:, 1] * unit_x
                weights = align.clamp(min=0)
                if float(weights.sum().item()) <= 0.0:
                    weights = torch.ones_like(weights, dtype=torch.float64)
            if strength < 1.0:
                current = K_layer[site].to(dtype=torch.float64)
                weights = (1.0 - strength) * current + strength * weights
            counts = _counts_from_weights(weights, int(params.B_k), int(params.l_k))
            if torch.all(counts == counts[0]) and params.B_k > 0:
                noise = torch.rand_like(weights, generator=generator)
                counts = _counts_from_weights(weights + 1e-6 * noise, int(params.B_k), int(params.l_k))
            K_layer[site] = counts.to(dtype=K_layer.dtype)
    return {"sites": sites, "interfaces": interface_list, "mode": "radial_inward", "strength": strength}


def apply_k_redistribute_radial_outward_in_ring(
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    center: Tuple[float, float],
    interfaces: str | Sequence[int] | None = "all",
    strength: float = 1.0,
    rng=None,
) -> dict:
    interface_list = _resolve_interfaces(interfaces, state.layers - 1)
    if not interface_list or state.K_K == 0:
        return {"sites": 0, "interfaces": []}
    if state.lattice.d != 2:
        raise ValueError("radial redistribution expects 2D lattice")
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    coords = state.lattice.index_to_coord(idx).to(dtype=torch.float64)
    offsets = state.R_K.to(device=device, dtype=torch.float64)
    generator = rng or torch.Generator(device=device)
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))
    sites = int(idx.numel())
    for interface in interface_list:
        K_layer = state.K[interface]
        for i, site in enumerate(idx.tolist()):
            dy = float(center[0]) - float(coords[i, 0].item())
            dx = float(center[1]) - float(coords[i, 1].item())
            norm = math.sqrt(dx * dx + dy * dy)
            if norm <= 0:
                weights = torch.ones_like(offsets[:, 0], dtype=torch.float64)
            else:
                unit_y = dy / norm
                unit_x = dx / norm
                align = -(offsets[:, 0] * unit_y + offsets[:, 1] * unit_x)
                weights = align.clamp(min=0)
                if float(weights.sum().item()) <= 0.0:
                    weights = torch.ones_like(weights, dtype=torch.float64)
            if strength < 1.0:
                current = K_layer[site].to(dtype=torch.float64)
                weights = (1.0 - strength) * current + strength * weights
            counts = _counts_from_weights(weights, int(params.B_k), int(params.l_k))
            if torch.all(counts == counts[0]) and params.B_k > 0:
                noise = torch.rand_like(weights, generator=generator)
                counts = _counts_from_weights(weights + 1e-6 * noise, int(params.B_k), int(params.l_k))
            K_layer[site] = counts.to(dtype=K_layer.dtype)
    return {"sites": sites, "interfaces": interface_list, "mode": "radial_outward", "strength": strength}


def apply_k_redistribute_radial_random_in_ring(
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    interfaces: str | Sequence[int] | None = "all",
    rng=None,
) -> dict:
    interface_list = _resolve_interfaces(interfaces, state.layers - 1)
    if not interface_list or state.K_K == 0:
        return {"sites": 0, "interfaces": []}
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    generator = rng or torch.Generator(device=device)
    sites = int(idx.numel())
    for interface in interface_list:
        K_layer = state.K[interface]
        for site in idx.tolist():
            weights = torch.rand(state.K_K, dtype=torch.float64, device=device, generator=generator)
            counts = _counts_from_weights(weights, int(params.B_k), int(params.l_k))
            K_layer[site] = counts.to(dtype=K_layer.dtype)
    return {"sites": sites, "interfaces": interface_list, "mode": "radial_random"}


def check_k_invariants(state: State, params: Params) -> Tuple[bool, str]:
    if state.K_K == 0:
        return True, "OK"
    if torch.any(state.K < 0) or torch.any(state.K > params.l_k):
        return False, "K entries out of bounds"
    k_sum = state.K.sum(dim=-1)
    if not torch.all(k_sum == params.B_k):
        return False, "K per-site budget invariant violated"
    return True, "OK"


def apply_sigma_randomize(
    state: State, flat_idx: torch.Tensor, layers: str | Sequence[int] | None = "all", rng=None
) -> None:
    layer_list = _resolve_layers(layers, state.layers)
    if not layer_list:
        return
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    generator = rng or torch.Generator(device=device)
    for layer in layer_list:
        vals = torch.randint(
            0, 2, (idx.numel(),), generator=generator, device=device, dtype=torch.int8
        )
        state.sigma[layer, idx] = vals * 2 - 1


def apply_sigma_flip(
    state: State, flat_idx: torch.Tensor, layers: str | Sequence[int] | None = "all"
) -> None:
    layer_list = _resolve_layers(layers, state.layers)
    if not layer_list:
        return
    device = state.device
    idx = flat_idx.to(device=device, dtype=torch.long)
    for layer in layer_list:
        state.sigma[layer, idx] = -state.sigma[layer, idx]


def apply_w_lesion_redistribute(
    state: State,
    params: Params,
    flat_idx: torch.Tensor,
    layers: str | Sequence[int] | None = "all",
    frac: float = 1.0,
    rng=None,
) -> dict:
    layer_list = _resolve_layers(layers, state.layers)
    device = state.device
    if not layer_list:
        return {"removed_tokens": 0, "pool_size": 0, "layers_used": []}

    idx = flat_idx.to(device=device, dtype=torch.long)
    generator = rng or torch.Generator(device=device)
    removed_total = 0
    pool_total = 0

    for layer in layer_list:
        W_layer = state.W[layer]
        K_W = W_layer.shape[1]
        region_entries = (idx[:, None] * K_W + torch.arange(K_W, device=device)).reshape(-1)
        W_flat = W_layer.reshape(-1)
        region_counts = W_flat[region_entries].to(dtype=torch.int64)
        total_tokens = int(region_counts.sum().item())
        if total_tokens == 0:
            continue
        remove_tokens = total_tokens if frac >= 1.0 else int(round(frac * total_tokens))
        remove_tokens = min(remove_tokens, total_tokens)
        if remove_tokens <= 0:
            continue

        region_slots = torch.repeat_interleave(
            torch.arange(region_entries.numel(), device=device, dtype=torch.long),
            region_counts,
        )
        if remove_tokens > region_slots.numel():
            raise ValueError("Not enough region tokens to remove")
        perm = torch.randperm(region_slots.numel(), generator=generator, device=device)[:remove_tokens]
        removed_entry_idx = region_slots[perm]
        removal_counts = torch.bincount(
            removed_entry_idx, minlength=region_entries.numel()
        )
        region_counts = region_counts - removal_counts
        W_flat[region_entries] = region_counts.to(dtype=W_flat.dtype)

        mask = torch.ones(W_flat.numel(), dtype=torch.bool, device=device)
        mask[region_entries] = False
        outside_entries = mask.nonzero(as_tuple=False).flatten()
        available = (params.l_w - W_flat[outside_entries]).clamp(min=0).to(dtype=torch.int64)
        pool_size = int(available.sum().item())
        if remove_tokens > pool_size:
            raise ValueError("Not enough capacity to redistribute removed tokens")
        pool_total += pool_size
        outside_slots = torch.repeat_interleave(
            torch.arange(outside_entries.numel(), device=device, dtype=torch.long), available
        )
        perm_out = torch.randperm(outside_slots.numel(), generator=generator, device=device)[:remove_tokens]
        add_entry_idx = outside_slots[perm_out]
        add_counts = torch.bincount(add_entry_idx, minlength=outside_entries.numel())
        W_flat[outside_entries] += add_counts.to(dtype=W_flat.dtype)
        removed_total += remove_tokens

    return {"removed_tokens": removed_total, "pool_size": pool_total, "layers_used": layer_list}


def check_w_invariants(state: State, params: Params) -> Tuple[bool, str]:
    if torch.any(state.W < 0) or torch.any(state.W > params.l_w):
        return False, "W entries out of bounds"
    if state.K_W > 0:
        w_total = int(state.W.sum().item())
        if w_total != int(params.B_w):
            return False, "W global budget invariant violated"
    return True, "OK"
