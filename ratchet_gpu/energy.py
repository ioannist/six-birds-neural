from __future__ import annotations

import torch

from .lattice import gather_neighbors
from .state import State


def cross_layer_pred(state: State) -> torch.Tensor:
    pred = torch.zeros(
        (state.layers, state.N), dtype=torch.float64, device=state.device
    )
    if state.K_K == 0 or state.params.B_k == 0:
        return pred

    for layer in range(1, state.layers):
        sigma_lower = state.sigma[layer - 1].to(dtype=torch.float64)
        neighbors = gather_neighbors(sigma_lower, state.lattice, state.R_K)
        weights = state.K[layer - 1].to(dtype=torch.float64)
        weighted_sum = (weights * neighbors).sum(dim=-1)
        pred[layer] = weighted_sum / float(state.params.B_k)
    return pred


def sigma_hat(state: State, layer: int) -> torch.Tensor:
    if layer <= 0 or layer >= state.layers:
        raise ValueError("layer must be in [1, layers-1]")
    return cross_layer_pred(state)[layer]


def mismatch(state: State) -> torch.Tensor:
    pred = cross_layer_pred(state)
    mismatch_full = torch.zeros_like(pred)
    if state.layers <= 1:
        return mismatch_full

    sigma = state.sigma.to(dtype=torch.float64)
    mismatch_full[1:] = (sigma[1:] - pred[1:]) ** 2
    return mismatch_full


def mismatch_mean(state: State) -> float:
    if state.layers <= 1:
        return 0.0
    mismatch_full = mismatch(state)
    return float(mismatch_full[1:].mean().item())


def energy_w(state: State) -> torch.Tensor:
    if state.K_W == 0:
        return torch.zeros((), dtype=torch.float64, device=state.device)

    total = torch.zeros((), dtype=torch.float64, device=state.device)
    for layer in range(state.layers):
        sigma_layer = state.sigma[layer].to(dtype=torch.float64)
        neighbors = gather_neighbors(sigma_layer, state.lattice, state.R_W)
        weights = state.W[layer].to(dtype=torch.float64)
        total += (weights * sigma_layer[:, None] * neighbors).sum()
    return -float(state.params.J) * total


def energy_inter(state: State) -> torch.Tensor:
    if state.K_K == 0 or state.params.B_k == 0:
        return torch.zeros((), dtype=torch.float64, device=state.device)

    mismatch_full = mismatch(state)
    total = mismatch_full[1:].sum()
    return 0.5 * float(state.params.eta) * total


def energy_bar(state: State) -> torch.Tensor:
    sigma = state.sigma.to(dtype=torch.float64)
    n = state.n.to(dtype=torch.float64)
    s = state.s.to(dtype=torch.float64)
    mismatch_term = 0.5 * (1.0 - sigma * n)
    total = (s * mismatch_term).sum()
    return float(state.params.kappa_T) * total


def energy_total(state: State) -> torch.Tensor:
    return energy_w(state) + energy_bar(state) + energy_inter(state)


def _sigma_neighbors_for_site(
    state: State, layer: int, idx: int, offsets: torch.Tensor
) -> torch.Tensor:
    device = state.device
    coord = state.lattice.index_to_coord(torch.tensor(idx, device=device))
    coords = coord + offsets
    wrapped = state.lattice.wrap_coord(coords)
    indices = state.lattice.coord_to_index(wrapped)
    sigma_layer = state.sigma[layer].to(dtype=torch.float64)
    return sigma_layer[indices]


def delta_e_k_local_exchange(
    state: State, layer: int, i: int, k1: int, k2: int
) -> float:
    if state.params.B_k == 0 or state.K_K == 0:
        return 0.0
    if layer <= 0 or layer >= state.layers:
        raise ValueError("layer must be in [1, layers-1]")

    sigma_neighbors = _sigma_neighbors_for_site(state, layer - 1, i, state.R_K)
    K_site = state.K[layer - 1, i].to(dtype=torch.float64)

    sigma_hat_u = (K_site * sigma_neighbors).sum() / float(state.params.B_k)
    delta_sigma_hat = (sigma_neighbors[k2] - sigma_neighbors[k1]) / float(state.params.B_k)

    sigma_u = state.sigma[layer, i].to(dtype=torch.float64)
    before = (sigma_u - sigma_hat_u) ** 2
    after = (sigma_u - (sigma_hat_u + delta_sigma_hat)) ** 2
    delta_e = 0.5 * float(state.params.eta) * (after - before)
    return float(delta_e.item())


def delta_phi_k_local_exchange(
    state: State, layer: int, i: int, k1: int, k2: int
) -> float:
    if state.params.B_k == 0 or state.K_K == 0:
        return 0.0
    if layer <= 0 or layer >= state.layers:
        raise ValueError("layer must be in [1, layers-1]")

    sigma_neighbors = _sigma_neighbors_for_site(state, layer - 1, i, state.R_K)
    K_site = state.K[layer - 1, i].to(dtype=torch.float64)

    sigma_hat_u = (K_site * sigma_neighbors).sum() / float(state.params.B_k)
    delta_sigma_hat = (sigma_neighbors[k2] - sigma_neighbors[k1]) / float(state.params.B_k)

    sigma_u = state.sigma[layer, i].to(dtype=torch.float64)
    before = (sigma_u - sigma_hat_u) ** 2
    after = (sigma_u - (sigma_hat_u + delta_sigma_hat)) ** 2
    delta_phi = 0.5 * (after - before)
    return float(delta_phi.item())


def delta_e_k_neighbor_trade(
    state: State, layer: int, u: int, v: int, a: int, b: int
) -> float:
    if state.params.B_k == 0 or state.K_K == 0:
        return 0.0
    if layer <= 0 or layer >= state.layers:
        raise ValueError("layer must be in [1, layers-1]")

    delta_phi = delta_phi_k_neighbor_trade(state, layer, u, v, a, b)
    delta_e = float(state.params.eta) * delta_phi
    return delta_e


def delta_phi_k_neighbor_trade(
    state: State, layer: int, u: int, v: int, a: int, b: int
) -> float:
    if state.params.B_k == 0 or state.K_K == 0:
        return 0.0
    if layer <= 0 or layer >= state.layers:
        raise ValueError("layer must be in [1, layers-1]")

    sigma_neighbors_u = _sigma_neighbors_for_site(state, layer - 1, u, state.R_K)
    sigma_neighbors_v = _sigma_neighbors_for_site(state, layer - 1, v, state.R_K)
    K_u = state.K[layer - 1, u].to(dtype=torch.float64)
    K_v = state.K[layer - 1, v].to(dtype=torch.float64)

    sigma_hat_u = (K_u * sigma_neighbors_u).sum() / float(state.params.B_k)
    sigma_hat_v = (K_v * sigma_neighbors_v).sum() / float(state.params.B_k)

    delta_hat_u = (sigma_neighbors_u[a] - sigma_neighbors_u[b]) / float(state.params.B_k)
    delta_hat_v = (sigma_neighbors_v[b] - sigma_neighbors_v[a]) / float(state.params.B_k)

    sigma_u = state.sigma[layer, u].to(dtype=torch.float64)
    sigma_v = state.sigma[layer, v].to(dtype=torch.float64)

    before_u = (sigma_u - sigma_hat_u) ** 2
    after_u = (sigma_u - (sigma_hat_u + delta_hat_u)) ** 2
    before_v = (sigma_v - sigma_hat_v) ** 2
    after_v = (sigma_v - (sigma_hat_v + delta_hat_v)) ** 2

    delta_phi = 0.5 * ((after_u - before_u) + (after_v - before_v))
    return float(delta_phi.item())


def delta_e_w_local_exchange(
    state: State, layer: int, i: int, k1: int, k2: int
) -> float:
    if state.K_W == 0:
        return 0.0

    sigma_neighbors = _sigma_neighbors_for_site(state, layer, i, state.R_W)
    sigma_u = state.sigma[layer, i].to(dtype=torch.float64)

    delta = sigma_neighbors[k2] - sigma_neighbors[k1]
    delta_e = -float(state.params.J) * sigma_u * delta
    return float(delta_e.item())


def delta_e_w_neighbor_exchange(
    state: State,
    layer: int,
    u: int,
    v: int,
    k_u: int,
    k_v: int,
) -> float:
    if state.K_W == 0:
        return 0.0

    device = state.device
    coord_u = state.lattice.index_to_coord(torch.tensor(u, device=device))
    coord_v = state.lattice.index_to_coord(torch.tensor(v, device=device))

    coords_u = coord_u + state.R_W[k_u]
    coords_v = coord_v + state.R_W[k_v]
    idx_u = state.lattice.coord_to_index(state.lattice.wrap_coord(coords_u))
    idx_v = state.lattice.coord_to_index(state.lattice.wrap_coord(coords_v))

    sigma_layer = state.sigma[layer].to(dtype=torch.float64)
    sigma_u = sigma_layer[u]
    sigma_v = sigma_layer[v]

    delta_e = -float(state.params.J) * sigma_u * sigma_layer[idx_u]
    delta_e += float(state.params.J) * sigma_v * sigma_layer[idx_v]
    return float(delta_e.item())


def delta_e_n_flip(state: State, layer: int, i: int) -> float:
    sigma_u = state.sigma[layer, i].to(dtype=torch.float64)
    n_u = state.n[layer, i].to(dtype=torch.float64)
    s_u = state.s[layer, i].to(dtype=torch.float64)
    delta_e = float(state.params.kappa_T) * s_u * sigma_u * n_u
    return float(delta_e.item())


def delta_e_s_step(state: State, layer: int, i: int, delta_s: int) -> float:
    sigma_u = state.sigma[layer, i].to(dtype=torch.float64)
    n_u = state.n[layer, i].to(dtype=torch.float64)
    mismatch_term = 0.5 * (1.0 - sigma_u * n_u)
    delta_e = float(state.params.kappa_T) * float(delta_s) * mismatch_term
    return float(delta_e.item())


def delta_e_spin_flip(state: State, layer: int, i: int) -> tuple[float, float]:
    sigma_u = state.sigma[layer, i].to(dtype=torch.float64)
    n_u = state.n[layer, i].to(dtype=torch.float64)
    s_u = state.s[layer, i].to(dtype=torch.float64)

    delta_bar = float(state.params.kappa_T) * s_u * sigma_u * n_u

    delta_w = 0.0
    if state.K_W > 0:
        device = state.device
        coord = state.lattice.index_to_coord(torch.tensor(i, device=device))
        coords_out = coord + state.R_W
        idx_out = state.lattice.coord_to_index(state.lattice.wrap_coord(coords_out))
        coords_in = coord - state.R_W
        idx_in = state.lattice.coord_to_index(state.lattice.wrap_coord(coords_in))

        sigma_layer = state.sigma[layer].to(dtype=torch.float64)
        sigma_out = sigma_layer[idx_out]
        sigma_in = sigma_layer[idx_in]
        W_out = state.W[layer, i].to(dtype=torch.float64)
        W_in = state.W[layer, idx_in, torch.arange(state.K_W, device=device)]

        sum_out = (W_out * sigma_out).sum()
        sum_in = (W_in * sigma_in).sum()
        delta_w = 2.0 * float(state.params.J) * sigma_u * (sum_out + sum_in)

    delta_inter = 0.0
    if state.params.B_k > 0 and state.K_K > 0:
        device = state.device
        coord = state.lattice.index_to_coord(torch.tensor(i, device=device))
        if layer >= 1:
            coords_up = coord + state.R_K
            idx_up = state.lattice.coord_to_index(state.lattice.wrap_coord(coords_up))
            sigma_lower = state.sigma[layer - 1].to(dtype=torch.float64)
            sigma_neighbors = sigma_lower[idx_up]
            K_site = state.K[layer - 1, i].to(dtype=torch.float64)
            sigma_hat_u = (K_site * sigma_neighbors).sum() / float(state.params.B_k)
            delta_inter += 2.0 * float(state.params.eta) * sigma_u * sigma_hat_u
        if layer < state.layers - 1:
            sigma_lower = state.sigma[layer].to(dtype=torch.float64)
            sigma_upper = state.sigma[layer + 1].to(dtype=torch.float64)
            for k in range(state.K_K):
                r = state.R_K[k]
                coord_v = coord - r
                idx_v = state.lattice.coord_to_index(
                    state.lattice.wrap_coord(coord_v)
                )
                K_vr = state.K[layer, idx_v, k].to(dtype=torch.float64)
                if K_vr.item() == 0:
                    continue
                coords_v = coord_v + state.R_K
                idx_neighbors = state.lattice.coord_to_index(
                    state.lattice.wrap_coord(coords_v)
                )
                sigma_neighbors = sigma_lower[idx_neighbors]
                K_site = state.K[layer, idx_v].to(dtype=torch.float64)
                sigma_hat_v = (K_site * sigma_neighbors).sum() / float(state.params.B_k)
                sigma_v = sigma_upper[idx_v]
                delta = -2.0 * sigma_u * K_vr / float(state.params.B_k)
                diff = sigma_v - sigma_hat_v
                delta_inter += 0.5 * float(state.params.eta) * (
                    delta * delta - 2.0 * diff * delta
                )

    delta_total = float(delta_bar + delta_w + delta_inter)
    return delta_total, float(delta_inter)
