from __future__ import annotations

import torch

from .energy import (
    delta_e_k_local_exchange,
    delta_e_k_neighbor_trade,
    delta_e_n_flip,
    delta_e_s_step,
    delta_e_spin_flip,
    delta_e_w_local_exchange,
    delta_e_w_neighbor_exchange,
    delta_phi_k_local_exchange,
    delta_phi_k_neighbor_trade,
)
from .lattice import gather_neighbors
from .state import State


def _choose_two_distinct(k: int, generator: torch.Generator, device: torch.device) -> tuple[int, int]:
    k1 = int(torch.randint(0, k, (1,), generator=generator, device=device).item())
    k2_raw = int(torch.randint(0, k - 1, (1,), generator=generator, device=device).item())
    k2 = k2_raw + 1 if k2_raw >= k1 else k2_raw
    return k1, k2


def _metropolis_accept(
    delta_e: float,
    delta_phi: float,
    drive_enabled: bool,
    state: State,
    generator: torch.Generator,
) -> tuple[bool, float]:
    if state.params.beta == 0.0:
        return True, 0.0

    w6 = 0.0
    if state.params.p6_on and drive_enabled:
        w6 = -float(state.params.eta_drive) * float(delta_phi)

    delta_eff = delta_e - w6
    log_u = torch.log(torch.rand((), generator=generator, device=state.device)).item()
    accept = log_u < -state.params.beta * delta_eff
    ep = -state.params.beta * delta_eff if accept else 0.0
    return accept, ep


def spin_flip_color(state: State, color: int, generator: torch.Generator) -> tuple[bool, float]:
    layer = int(torch.randint(0, state.layers, (1,), generator=generator, device=state.device).item())
    indices = state.color_indices[layer][color]
    if indices.numel() == 0:
        return False, 0.0
    idx = int(indices[torch.randint(0, indices.numel(), (1,), generator=generator, device=state.device)])

    delta_e, delta_inter = delta_e_spin_flip(state, layer, idx)
    accept, ep = _metropolis_accept(delta_e, delta_inter, False, state, generator)
    if accept:
        state.sigma[layer, idx] *= -1
    return accept, ep


def n_flip(state: State, generator: torch.Generator) -> tuple[bool, float]:
    layer = int(torch.randint(0, state.layers, (1,), generator=generator, device=state.device).item())
    idx = int(torch.randint(0, state.N, (1,), generator=generator, device=state.device).item())

    delta_e = delta_e_n_flip(state, layer, idx)
    accept, ep = _metropolis_accept(delta_e, 0.0, False, state, generator)
    if accept:
        state.n[layer, idx] *= -1
    return accept, ep


def s_step(state: State, generator: torch.Generator) -> tuple[bool, float]:
    layer = int(torch.randint(0, state.layers, (1,), generator=generator, device=state.device).item())
    idx = int(torch.randint(0, state.N, (1,), generator=generator, device=state.device).item())

    delta_s = int(torch.randint(0, 2, (1,), generator=generator, device=state.device).item())
    delta_s = 1 if delta_s == 1 else -1
    new_s = int(state.s[layer, idx].item()) + delta_s
    if new_s < 0 or new_s > state.params.l_s:
        return False, 0.0

    delta_e = delta_e_s_step(state, layer, idx, delta_s)
    accept, ep = _metropolis_accept(delta_e, 0.0, False, state, generator)
    if accept:
        state.s[layer, idx] = new_s
    return accept, ep


def w_local_exchange(state: State, generator: torch.Generator) -> tuple[bool, float]:
    if state.K_W < 2:
        return False, 0.0

    layer = int(torch.randint(0, state.layers, (1,), generator=generator, device=state.device).item())
    idx = int(torch.randint(0, state.N, (1,), generator=generator, device=state.device).item())

    k1, k2 = _choose_two_distinct(state.K_W, generator, state.device)

    W_site = state.W[layer, idx]
    if W_site[k1] <= 0 or W_site[k2] >= state.params.l_w:
        return False, 0.0

    delta_e = delta_e_w_local_exchange(state, layer, idx, k1, k2)
    accept, ep = _metropolis_accept(delta_e, 0.0, False, state, generator)
    if accept:
        state.W[layer, idx, k1] -= 1
        state.W[layer, idx, k2] += 1
    return accept, ep


def w_neighbor_exchange(state: State, generator: torch.Generator) -> tuple[bool, float]:
    if state.K_W < 1:
        return False, 0.0

    if state.device.type == "cuda":
        layer = int(
            torch.randint(0, state.layers, (1,), generator=generator, device=state.device).item()
        )
        src_idx = state.color_indices[layer][0]
        if src_idx.numel() == 0:
            return False, 0.0, 0, 0

        axis = int(
            torch.randint(0, state.lattice.d, (1,), generator=generator, device=state.device).item()
        )
        sign = int(
            torch.randint(0, 2, (1,), generator=generator, device=state.device).item()
        )
        delta = torch.zeros(state.lattice.d, device=state.device, dtype=torch.long)
        delta[axis] = 1 if sign == 0 else -1

        coords_src = state.lattice.index_to_coord(src_idx)
        coords_dst = coords_src + delta
        dst_idx = state.lattice.coord_to_index(state.lattice.wrap_coord(coords_dst))

        k_src = torch.randint(
            0, state.K_W, (src_idx.numel(),), generator=generator, device=state.device
        )
        k_dst = torch.randint(
            0, state.K_W, (src_idx.numel(),), generator=generator, device=state.device
        )

        W_layer = state.W[layer]
        W_src = W_layer[src_idx, k_src]
        W_dst = W_layer[dst_idx, k_dst]
        feasible = (W_dst > 0) & (W_src < state.params.l_w)
        proposals = int(src_idx.numel())
        if proposals == 0:
            return False, 0.0, 0, 0

        sigma_layer = state.sigma[layer].to(dtype=torch.float64)
        neighbors = gather_neighbors(sigma_layer, state.lattice, state.R_W)
        sigma_src = sigma_layer[src_idx]
        sigma_dst = sigma_layer[dst_idx]
        sigma_src_r = neighbors[src_idx, k_src]
        sigma_dst_r = neighbors[dst_idx, k_dst]

        delta_e = -float(state.params.J) * sigma_src * sigma_src_r
        delta_e = delta_e + float(state.params.J) * sigma_dst * sigma_dst_r

        if state.params.beta == 0.0:
            accept_mask = feasible
        else:
            log_u = torch.log(
                torch.rand(delta_e.shape, generator=generator, device=state.device)
            )
            accept_mask = feasible & (log_u < -state.params.beta * delta_e)

        accepted_count = int(accept_mask.sum().item())
        if accepted_count:
            idx_src = src_idx[accept_mask]
            idx_dst = dst_idx[accept_mask]
            k_src_acc = k_src[accept_mask]
            k_dst_acc = k_dst[accept_mask]
            W_layer[idx_src, k_src_acc] += 1
            W_layer[idx_dst, k_dst_acc] -= 1

        if accepted_count and state.params.beta != 0.0:
            ep_inc = (-state.params.beta * delta_e[accept_mask]).sum()
        else:
            ep_inc = torch.zeros((), device=state.device, dtype=torch.float64)

        return accepted_count > 0, float(ep_inc.item()), proposals, accepted_count

    layer = int(torch.randint(0, state.layers, (1,), generator=generator, device=state.device).item())
    u = int(torch.randint(0, state.N, (1,), generator=generator, device=state.device).item())

    coord_u = state.lattice.index_to_coord(torch.tensor(u, device=state.device))
    delta = torch.zeros(state.lattice.d, device=state.device, dtype=torch.long)
    axis = int(
        torch.randint(0, state.lattice.d, (1,), generator=generator, device=state.device).item()
    )
    sign = int(torch.randint(0, 2, (1,), generator=generator, device=state.device).item())
    delta[axis] = 1 if sign == 0 else -1
    coord_v = coord_u + delta
    v = int(state.lattice.coord_to_index(state.lattice.wrap_coord(coord_v)).item())

    k_u = int(torch.randint(0, state.K_W, (1,), generator=generator, device=state.device).item())
    k_v = int(torch.randint(0, state.K_W, (1,), generator=generator, device=state.device).item())

    if state.W[layer, v, k_v] <= 0 or state.W[layer, u, k_u] >= state.params.l_w:
        return False, 0.0

    delta_e = delta_e_w_neighbor_exchange(state, layer, u, v, k_u, k_v)
    accept, ep = _metropolis_accept(delta_e, 0.0, False, state, generator)
    if accept:
        state.W[layer, v, k_v] -= 1
        state.W[layer, u, k_u] += 1
    return accept, ep


def k_local_exchange(state: State, generator: torch.Generator) -> tuple[bool, float]:
    if state.K_K < 2:
        return False, 0.0

    layer = int(torch.randint(1, state.layers, (1,), generator=generator, device=state.device).item())
    idx = int(torch.randint(0, state.N, (1,), generator=generator, device=state.device).item())

    k1, k2 = _choose_two_distinct(state.K_K, generator, state.device)

    K_site = state.K[layer - 1, idx]
    if K_site[k1] <= 0 or K_site[k2] >= state.params.l_k:
        return False, 0.0

    delta_e = delta_e_k_local_exchange(state, layer, idx, k1, k2)
    delta_phi = delta_phi_k_local_exchange(state, layer, idx, k1, k2)
    accept, ep = _metropolis_accept(delta_e, delta_phi, True, state, generator)
    if accept:
        state.K[layer - 1, idx, k1] -= 1
        state.K[layer - 1, idx, k2] += 1
    return accept, ep


def k_neighbor_trade(state: State, generator: torch.Generator) -> tuple[bool, float]:
    if state.K_K < 2:
        return False, 0.0

    layer = int(torch.randint(1, state.layers, (1,), generator=generator, device=state.device).item())
    u = int(torch.randint(0, state.N, (1,), generator=generator, device=state.device).item())

    coord_u = state.lattice.index_to_coord(torch.tensor(u, device=state.device))
    delta = torch.zeros(state.lattice.d, device=state.device, dtype=torch.long)
    axis = int(
        torch.randint(0, state.lattice.d, (1,), generator=generator, device=state.device).item()
    )
    sign = int(torch.randint(0, 2, (1,), generator=generator, device=state.device).item())
    delta[axis] = 1 if sign == 0 else -1
    coord_v = coord_u + delta
    v = int(state.lattice.coord_to_index(state.lattice.wrap_coord(coord_v)).item())

    a, b = _choose_two_distinct(state.K_K, generator, state.device)

    K_u = state.K[layer - 1, u]
    K_v = state.K[layer - 1, v]
    if K_u[b] <= 0 or K_v[a] <= 0:
        return False, 0.0
    if K_u[a] >= state.params.l_k or K_v[b] >= state.params.l_k:
        return False, 0.0

    delta_e = delta_e_k_neighbor_trade(state, layer, u, v, a, b)
    delta_phi = delta_phi_k_neighbor_trade(state, layer, u, v, a, b)
    accept, ep = _metropolis_accept(delta_e, delta_phi, True, state, generator)
    if accept:
        state.K[layer - 1, u, a] += 1
        state.K[layer - 1, u, b] -= 1
        state.K[layer - 1, v, b] += 1
        state.K[layer - 1, v, a] -= 1
    return accept, ep


def k_p5_exchange(state: State, generator: torch.Generator) -> tuple[bool, float]:
    return k_local_exchange(state, generator)


def excitable_step_color(
    state: State, color: int, generator: torch.Generator
) -> tuple[bool, float, int, int]:
    if state.params.sigma_mode != "excitable4":
        return False, 0.0, 0, 0

    total_sites = 0
    total_changed = 0
    for layer in range(state.layers):
        idx = state.color_indices[layer][color]
        if idx.numel() == 0:
            continue

        total_sites += int(idx.numel())
        sigma_layer = state.sigma[layer]
        if state.K_W > 0:
            exc_mask = (sigma_layer == 1).to(dtype=torch.float32)
            neighbors = gather_neighbors(exc_mask, state.lattice, state.R_W)
            weights = state.W[layer].to(dtype=torch.float32)
            input_val = (weights * neighbors).sum(dim=-1)
        else:
            input_val = torch.zeros(state.N, device=state.device, dtype=torch.float32)

        input_idx = input_val[idx]
        cur = sigma_layer[idx].to(dtype=torch.int16)
        next_state = cur.clone()

        mask1 = cur == 1
        next_state[mask1] = 2
        mask2 = cur == 2
        next_state[mask2] = 3
        mask3 = cur == 3
        if mask3.any():
            if state.params.exc_p_recover >= 1.0:
                next_state[mask3] = 0
            else:
                recover = torch.rand(
                    (int(mask3.sum().item()),),
                    generator=generator,
                    device=state.device,
                ) < float(state.params.exc_p_recover)
                next_state[mask3] = torch.where(
                    recover, torch.zeros_like(next_state[mask3]), next_state[mask3]
                )

        mask0 = cur == 0
        if mask0.any():
            p = torch.sigmoid(
                float(state.params.exc_beta)
                * (input_idx[mask0] - float(state.params.exc_theta))
            )
            if state.params.exc_p_spont > 0.0:
                p = torch.clamp(p + float(state.params.exc_p_spont), max=1.0)
            excite = torch.rand(
                p.shape, generator=generator, device=state.device
            ) < p
            next_state[mask0] = torch.where(
                excite, torch.ones_like(next_state[mask0]), torch.zeros_like(next_state[mask0])
            )

        changed = (next_state != cur).sum().item()
        total_changed += int(changed)
        sigma_layer[idx] = next_state.to(dtype=sigma_layer.dtype)

    if total_sites == 0:
        return False, 0.0, 0, 0
    return total_changed > 0, 0.0, total_sites, total_changed
