from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch

from .lattice import gather_neighbors
from .state import State


def to_grid(x_1d: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    if x_1d.ndim != 1:
        raise ValueError("to_grid expects a 1D tensor")
    H, W = shape
    if x_1d.numel() != H * W:
        raise ValueError("to_grid size mismatch")
    return x_1d.reshape(H, W)


def to_grid_layers(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("to_grid_layers expects a 2D tensor [L, N]")
    H, W = shape
    if x.shape[1] != H * W:
        raise ValueError("to_grid_layers size mismatch")
    return x.reshape(x.shape[0], H, W)


def sigma_grid(state: State) -> torch.Tensor:
    return to_grid_layers(state.sigma, state.lattice.shape)  # type: ignore[arg-type]


def w_mass_grid(state: State) -> torch.Tensor:
    w_mass = state.W.sum(dim=-1)
    return to_grid_layers(w_mass, state.lattice.shape)  # type: ignore[arg-type]


def w_entropy_grid(state: State, eps: float = 1e-12) -> torch.Tensor:
    if state.K_W == 0:
        return to_grid_layers(
            torch.zeros((state.layers, state.N), device=state.device),
            state.lattice.shape,  # type: ignore[arg-type]
        )
    W = state.W.to(dtype=torch.float32)
    denom = W.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = W / denom
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return to_grid_layers(entropy, state.lattice.shape)  # type: ignore[arg-type]


def w_axis_bias_grid(state: State) -> torch.Tensor:
    if state.K_W == 0:
        return to_grid_layers(
            torch.zeros((state.layers, state.N), device=state.device),
            state.lattice.shape,  # type: ignore[arg-type]
        )
    W = state.W.to(dtype=torch.float32)
    denom = W.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = W / denom
    offsets = state.R_W.to(dtype=torch.float32)
    mu = torch.matmul(probs, offsets)
    bias = torch.linalg.norm(mu, dim=-1)
    return to_grid_layers(bias, state.lattice.shape)  # type: ignore[arg-type]


def k_entropy_grid(state: State, eps: float = 1e-12) -> torch.Tensor:
    if state.layers <= 1 or state.K_K == 0:
        empty = torch.zeros((state.layers - 1, state.N), device=state.device)
        return to_grid_layers(empty, state.lattice.shape)  # type: ignore[arg-type]
    K = state.K.to(dtype=torch.float32)
    denom = K.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = K / denom
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return to_grid_layers(entropy, state.lattice.shape)  # type: ignore[arg-type]


def k_axis_bias_grid(state: State) -> torch.Tensor:
    if state.layers <= 1 or state.K_K == 0:
        empty = torch.zeros((state.layers - 1, state.N), device=state.device)
        return to_grid_layers(empty, state.lattice.shape)  # type: ignore[arg-type]
    K = state.K.to(dtype=torch.float32)
    denom = K.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = K / denom
    offsets = state.R_K.to(dtype=torch.float32)
    mu = torch.matmul(probs, offsets)
    bias = torch.linalg.norm(mu, dim=-1)
    return to_grid_layers(bias, state.lattice.shape)  # type: ignore[arg-type]


def k_radial_focus_grid(state: State, center: Tuple[float, float]) -> torch.Tensor:
    if state.layers <= 1 or state.K_K == 0:
        empty = torch.zeros((state.layers - 1, state.N), device=state.device)
        return to_grid_layers(empty, state.lattice.shape)  # type: ignore[arg-type]
    if state.lattice.d != 2:
        raise ValueError("k_radial_focus_grid expects 2D lattice")
    K = state.K.to(dtype=torch.float32)
    denom = K.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = K / denom
    offsets = state.R_K.to(dtype=torch.float32)
    mu = torch.matmul(probs, offsets)
    coords = state.lattice.index_to_coord(
        torch.arange(state.N, device=state.device)
    ).to(dtype=torch.float32)
    cy, cx = center
    dy = cy - coords[:, 0]
    dx = cx - coords[:, 1]
    norm = torch.sqrt(dx * dx + dy * dy).clamp(min=1.0)
    unit_y = dy / norm
    unit_x = dx / norm
    radial = mu[..., 0] * unit_y + mu[..., 1] * unit_x
    return to_grid_layers(radial, state.lattice.shape)  # type: ignore[arg-type]


def k_r2_grid(state: State) -> torch.Tensor:
    if state.layers <= 1 or state.K_K == 0:
        empty = torch.zeros((state.layers - 1, state.N), device=state.device)
        return to_grid_layers(empty, state.lattice.shape)  # type: ignore[arg-type]
    r2 = (state.R_K.to(dtype=torch.float32) ** 2).sum(dim=-1)
    K = state.K.to(dtype=torch.float32)
    denom = K.sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = K / denom
    r2_vals = (probs * r2).sum(dim=-1)
    return to_grid_layers(r2_vals, state.lattice.shape)  # type: ignore[arg-type]


def mismatch_abs_grid(state: State, p: float = 1.0) -> torch.Tensor:
    if state.layers <= 1 or state.K_K == 0 or state.params.B_k == 0:
        empty = torch.zeros((state.layers - 1, state.N), device=state.device)
        return to_grid_layers(empty, state.lattice.shape)  # type: ignore[arg-type]
    pred = torch.zeros(
        (state.layers - 1, state.N), dtype=torch.float64, device=state.device
    )
    for layer in range(1, state.layers):
        sigma_lower = state.sigma[layer - 1].to(dtype=torch.float64)
        neighbors = gather_neighbors(sigma_lower, state.lattice, state.R_K)
        weights = state.K[layer - 1].to(dtype=torch.float64)
        pred[layer - 1] = (weights * neighbors).sum(dim=-1) / float(state.params.B_k)
    diff = state.sigma[1:].to(dtype=torch.float64) - pred
    if p != 1.0:
        diff = diff.abs().pow(p)
    else:
        diff = diff.abs()
    return to_grid_layers(diff, state.lattice.shape)  # type: ignore[arg-type]


def compute_spatial_maps(
    state: State, keys: Iterable[str], p: float | None = None
) -> Dict[str, torch.Tensor]:
    shape = state.lattice.shape
    if len(shape) != 2:
        raise ValueError("spatial maps require a 2D lattice shape")
    key_list = [k.strip() for k in keys]
    maps: Dict[str, torch.Tensor] = {}
    for key in key_list:
        if key == "sigma":
            maps[key] = sigma_grid(state)
        elif key == "w_mass":
            maps[key] = w_mass_grid(state)
        elif key == "w_entropy":
            maps[key] = w_entropy_grid(state)
        elif key == "w_axis_bias":
            maps[key] = w_axis_bias_grid(state)
        elif key == "k_entropy":
            maps[key] = k_entropy_grid(state)
        elif key == "k_axis_bias":
            maps[key] = k_axis_bias_grid(state)
        elif key == "k_r2":
            maps[key] = k_r2_grid(state)
        elif key == "mismatch":
            maps[key] = mismatch_abs_grid(state, p=p or 1.0)
        else:
            raise ValueError(f"Unknown spatial map key: {key}")
    return maps


def finite_check(maps: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
    bad: List[str] = []
    for key, value in maps.items():
        if not value.dtype.is_floating_point:
            continue
        if not torch.isfinite(value).all():
            bad.append(key)
    return (len(bad) == 0), bad
