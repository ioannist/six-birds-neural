from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch

DEFAULT_STENCIL_POLICY = "l1_ball_odd"


@dataclass(frozen=True)
class Lattice:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.shape or any(s <= 0 for s in self.shape):
            raise ValueError("shape must be a non-empty tuple of positive ints")
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        d = len(self.shape)
        n = math.prod(self.shape)
        strides = []
        for k in range(d):
            stride = math.prod(self.shape[k + 1 :]) if k + 1 < d else 1
            strides.append(stride)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "N", n)
        object.__setattr__(self, "strides", tuple(strides))

    def wrap_coord(self, coord: Sequence[int] | torch.Tensor) -> tuple[int, ...] | torch.Tensor:
        coord_t = self._as_long_tensor(coord)
        shape_t = torch.as_tensor(self.shape, dtype=torch.long, device=coord_t.device)
        wrapped = coord_t % shape_t
        if torch.is_tensor(coord):
            return wrapped
        if self._is_coord_vector(coord) and wrapped.ndim == 1:
            return tuple(int(v) for v in wrapped.tolist())
        return wrapped

    def coord_to_index(self, coord: Sequence[int] | torch.Tensor) -> int | torch.Tensor:
        coord_t = self._as_long_tensor(coord)
        shape_t = torch.as_tensor(self.shape, dtype=torch.long, device=coord_t.device)
        strides_t = torch.as_tensor(self.strides, dtype=torch.long, device=coord_t.device)
        wrapped = coord_t % shape_t
        flat = (wrapped * strides_t).sum(dim=-1)
        if torch.is_tensor(coord):
            return flat
        if self._is_coord_vector(coord) and flat.ndim == 0:
            return int(flat.item())
        return flat

    def index_to_coord(self, index: int | Sequence[int] | torch.Tensor) -> tuple[int, ...] | torch.Tensor:
        index_t = self._as_long_tensor(index)
        index_t = index_t % self.N
        coords = []
        for stride, size in zip(self.strides, self.shape):
            coord_k = (index_t // stride) % size
            coords.append(coord_k)
        coord_t = torch.stack(coords, dim=-1)
        if torch.is_tensor(index):
            return coord_t
        if not isinstance(index, (list, tuple)) and coord_t.ndim == 1:
            return tuple(int(v) for v in coord_t.tolist())
        return coord_t

    @staticmethod
    def _as_long_tensor(value: int | Sequence[int] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=torch.long)
        return torch.as_tensor(value, dtype=torch.long)

    @staticmethod
    def _is_coord_vector(value: Sequence[int] | torch.Tensor) -> bool:
        if not isinstance(value, (list, tuple)):
            return False
        return all(
            not isinstance(elem, (list, tuple)) and not torch.is_tensor(elem) for elem in value
        )


def generate_stencil(
    d: int,
    *,
    policy: str = DEFAULT_STENCIL_POLICY,
    radius: int = 1,
    bipartite: bool = True,
    parity: Literal["any", "odd", "even"] | None = None,
    include_zero: bool = False,
    shape: Sequence[int] | None = None,
) -> torch.LongTensor:
    if d <= 0:
        raise ValueError("d must be >= 1")
    if radius < 0:
        raise ValueError("radius must be >= 0")

    policy_parity = None
    if policy.endswith("_odd"):
        policy_parity = "odd"
    elif policy.endswith("_even"):
        policy_parity = "even"

    if parity is None:
        if policy_parity is not None:
            parity = policy_parity
        elif bipartite:
            parity = "odd"
        else:
            parity = "any"

    if parity not in {"any", "odd", "even"}:
        raise ValueError("parity must be one of: any, odd, even")
    if shape is not None:
        if len(shape) != d:
            raise ValueError("shape length must match d")
        if parity == "odd" and any((int(s) % 2) != 0 for s in shape):
            warnings.warn(
                "odd-parity stencils only yield a bipartite torus when all dimensions are even",
                RuntimeWarning,
                stacklevel=2,
            )

    if policy == "unit":
        offsets = []
        for axis in range(d):
            vec = [0] * d
            vec[axis] = 1
            offsets.append(tuple(vec))
            vec[axis] = -1
            offsets.append(tuple(vec))
    elif policy.startswith("hypercube") or policy.startswith("l1_ball"):
        offsets = []
        grid = range(-radius, radius + 1)
        for vec in itertools.product(grid, repeat=d):
            if not include_zero and all(v == 0 for v in vec):
                continue
            if policy.startswith("l1_ball") and sum(abs(v) for v in vec) > radius:
                continue
            if parity != "any":
                if parity == "odd" and (sum(vec) % 2 == 0):
                    continue
                if parity == "even" and (sum(vec) % 2 != 0):
                    continue
            offsets.append(tuple(int(v) for v in vec))
    else:
        raise ValueError(f"unknown policy: {policy}")

    if parity != "any" and policy == "unit":
        offsets = [vec for vec in offsets if (sum(vec) % 2 == (0 if parity == "even" else 1))]

    unique = sorted(set(offsets))
    if not unique:
        return torch.empty((0, d), dtype=torch.long)
    return torch.tensor(unique, dtype=torch.long)


def gather_neighbors(
    x: torch.Tensor, lattice: Lattice, offsets: torch.Tensor
) -> torch.Tensor:
    if x.shape[0] != lattice.N:
        raise ValueError("x.shape[0] must match lattice.N")
    if offsets.ndim != 2 or offsets.shape[1] != lattice.d:
        raise ValueError("offsets must have shape [R, d]")

    extra_shape = x.shape[1:]
    x_view = x.reshape(*lattice.shape, *extra_shape)
    offsets_list: Iterable[Sequence[int]] = offsets.detach().cpu().tolist()

    gathered = []
    for offset in offsets_list:
        shifts = tuple(-int(shift) for shift in offset)
        rolled = torch.roll(x_view, shifts=shifts, dims=tuple(range(lattice.d)))
        gathered.append(rolled.reshape(lattice.N, *extra_shape))

    if not gathered:
        return x.new_empty((lattice.N, 0, *extra_shape))

    return torch.stack(gathered, dim=1)
