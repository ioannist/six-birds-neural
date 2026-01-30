from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .lattice import Lattice, generate_stencil
from .params import Params


@dataclass
class State:
    params: Params
    lattice: Lattice
    R_W: torch.Tensor
    R_K: torch.Tensor
    sigma: torch.Tensor
    n: torch.Tensor
    s: torch.Tensor
    W: torch.Tensor
    K: torch.Tensor
    color_indices: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    @property
    def device(self) -> torch.device:
        return self.sigma.device

    @property
    def layers(self) -> int:
        return self.sigma.shape[0]

    @property
    def N(self) -> int:
        return self.lattice.N

    @property
    def K_W(self) -> int:
        return int(self.R_W.shape[0])

    @property
    def K_K(self) -> int:
        return int(self.R_K.shape[0])

    @property
    def K_cross(self) -> torch.Tensor:
        return self.K

    @K_cross.setter
    def K_cross(self, value: torch.Tensor) -> None:
        self.K = value

    def K_cross_for_layer(self, layer: int) -> torch.Tensor:
        if layer <= 0 or layer >= self.layers:
            raise ValueError("layer must be in [1, layers-1]")
        return self.K[layer - 1]

    def check_invariants(self) -> None:
        if self.params.sigma_mode == "ising":
            sigma_ok = (self.sigma == 1) | (self.sigma == -1)
            if not torch.all(sigma_ok):
                raise ValueError("sigma must be in {-1, +1}")
        elif self.params.sigma_mode == "excitable4":
            if torch.any(self.sigma < 0) or torch.any(self.sigma > 3):
                raise ValueError("sigma must be in {0,1,2,3} for excitable4")
        else:
            raise ValueError(f"unknown sigma_mode {self.params.sigma_mode}")

        n_ok = (self.n == 1) | (self.n == -1)
        if not torch.all(n_ok):
            raise ValueError("n must be in {-1, +1}")

        if torch.any(self.s < 0) or torch.any(self.s > self.params.l_s):
            raise ValueError("s entries out of bounds")

        if torch.any(self.W < 0) or torch.any(self.W > self.params.l_w):
            raise ValueError("W entries out of bounds")

        if torch.any(self.K < 0) or torch.any(self.K > self.params.l_k):
            raise ValueError("K entries out of bounds")

        if self.K_W > 0:
            w_total = int(self.W.sum().item())
            if w_total != int(self.params.B_w):
                raise ValueError("W global budget invariant violated")

        if self.K_K > 0:
            k_sum = self.K.sum(dim=-1)
            if not torch.all(k_sum == self.params.B_k):
                raise ValueError("K per-site budget invariant violated")

    @classmethod
    def initialize(cls, params: Params, seed: int | None = None) -> "State":
        if params.layers < 2:
            raise ValueError("layers must be >= 2")
        if params.l_k < 0 or params.l_w < 0 or params.l_s < 0:
            raise ValueError("l_k, l_w, and l_s must be >= 0")
        if params.B_k < 0 or params.B_w < 0:
            raise ValueError("B_k and B_w must be >= 0")

        lattice = Lattice(params.shape)
        device = params.resolved_device()

        R_W = generate_stencil(
            d=lattice.d,
            policy=params.stencil_policy_w,
            radius=params.radius_w,
            bipartite=True,
            shape=params.shape,
        ).to(device)
        R_K = generate_stencil(
            d=lattice.d,
            policy=params.stencil_policy_k,
            radius=params.radius_k,
            bipartite=False,
            include_zero=params.include_zero_k,
        ).to(device)

        K_K = int(R_K.shape[0])
        if K_K == 0 and params.B_k != 0:
            raise ValueError("B_k must be 0 when R_K is empty")
        if params.B_k > params.l_k * K_K:
            raise ValueError("B_k exceeds per-site capacity l_k * K_K")

        K_W = int(R_W.shape[0])
        if K_W == 0 and params.B_w != 0:
            raise ValueError("B_w must be 0 when R_W is empty")
        total_capacity_w = params.l_w * params.layers * lattice.N * K_W
        if params.B_w > total_capacity_w:
            raise ValueError("B_w exceeds global capacity")

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)

        if params.sigma_mode == "ising":
            sigma = torch.randint(
                0,
                2,
                (params.layers, lattice.N),
                generator=gen,
                device=device,
                dtype=torch.int8,
            )
            sigma = sigma * 2 - 1
        elif params.sigma_mode == "excitable4":
            sigma = torch.zeros(
                (params.layers, lattice.N), device=device, dtype=torch.int8
            )
            if params.exc_init_frac > 0:
                excite = torch.rand(
                    (params.layers, lattice.N), generator=gen, device=device
                ) < float(params.exc_init_frac)
                sigma = sigma.to(dtype=torch.int8)
                sigma[excite] = 1
        else:
            raise ValueError(f"unknown sigma_mode {params.sigma_mode}")

        n = torch.randint(
            0,
            2,
            (params.layers, lattice.N),
            generator=gen,
            device=device,
            dtype=torch.int8,
        )
        n = n * 2 - 1

        s = torch.zeros((params.layers, lattice.N), dtype=torch.int16, device=device)

        W = _init_global_budget_tensor(
            params.layers,
            lattice.N,
            K_W,
            params.B_w,
            params.l_w,
            device,
            gen,
        )

        K = _init_k_tensor(
            params.layers - 1,
            lattice.N,
            K_K,
            params.B_k,
            params.l_k,
            device,
        )

        color_indices = _color_indices(lattice, params.layers, device)

        state = cls(
            params=params,
            lattice=lattice,
            R_W=R_W,
            R_K=R_K,
            sigma=sigma,
            n=n,
            s=s,
            W=W,
            K=K,
            color_indices=color_indices,
        )
        state.check_invariants()
        return state


def _init_k_tensor(
    layers_minus1: int,
    N: int,
    K_K: int,
    B_k: int,
    l_k: int,
    device: torch.device,
) -> torch.Tensor:
    if K_K == 0:
        return torch.empty((layers_minus1, N, 0), dtype=torch.int16, device=device)
    return _init_budget_tensor(layers_minus1, N, K_K, B_k, l_k, device)


def _init_budget_tensor(
    layers: int,
    N: int,
    K: int,
    B: int,
    l_max: int,
    device: torch.device,
) -> torch.Tensor:
    if K == 0:
        return torch.empty((layers, N, 0), dtype=torch.int16, device=device)
    base = B // K
    remainder = B % K

    if base > l_max:
        raise ValueError("budget exceeds per-site capacity")

    counts = torch.full((layers, N, K), base, dtype=torch.int16, device=device)
    if remainder:
        counts[..., :remainder] += 1
    return counts


def _init_global_budget_tensor(
    layers: int,
    N: int,
    K: int,
    B: int,
    l_max: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if K == 0:
        return torch.empty((layers, N, 0), dtype=torch.int16, device=device)
    total_entries = layers * N * K
    max_total = total_entries * l_max
    if B > max_total:
        raise ValueError("budget exceeds global capacity")

    counts = torch.zeros((layers, N, K), dtype=torch.int16, device=device)
    if B == 0:
        return counts

    if B == max_total:
        counts.fill_(l_max)
        return counts

    total_slots = total_entries * l_max
    slots = torch.randperm(
        total_slots, generator=generator, device=device, dtype=torch.int64
    )[:B]
    entry_idx = slots // l_max
    flat_counts = torch.bincount(entry_idx, minlength=total_entries)
    return flat_counts.view(layers, N, K).to(dtype=torch.int16)


def _color_indices(
    lattice: Lattice, layers: int, device: torch.device
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    indices = torch.arange(lattice.N, device=device, dtype=torch.long)
    coords = lattice.index_to_coord(indices)
    parity = (coords.sum(dim=-1) % 2).to(dtype=torch.long)

    per_layer = []
    for layer in range(layers):
        color = parity ^ (layer & 1)
        idx0 = (color == 0).nonzero(as_tuple=False).flatten()
        idx1 = (color == 1).nonzero(as_tuple=False).flatten()
        per_layer.append((idx0, idx1))
    return tuple(per_layer)
