from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _default_kernel_weights() -> dict[str, float]:
    return {
        "spin_flip_color0": 1.0,
        "spin_flip_color1": 1.0,
        "n_flip": 1.0,
        "s_step": 1.0,
        "w_local": 1.0,
        "k_local": 1.0,
        "k_neighbor_trade": 1.0,
        "k_p5_exchange": 1.0,
        "w_neighbor": 1.0,
        "excitable_color0": 0.0,
        "excitable_color1": 0.0,
    }


@dataclass(frozen=True)
class Params:
    shape: tuple[int, ...]
    layers: int
    sigma_mode: str = "ising"
    p3_on: bool = False
    p6_on: bool = False
    beta: float = 1.0
    J: float = 1.0
    kappa_T: float = 1.0
    eta: float = 0.2
    eta_drive: float = 0.0
    exc_init_frac: float = 0.02
    exc_p_spont: float = 1e-3
    exc_theta: float = 1.0
    exc_beta: float = 2.0
    exc_p_recover: float = 1.0
    strobe_on: bool = False
    strobe_signature: str = "mag_s"
    strobe_on: bool = False
    l_s: int = 1
    l_w: int = 3
    l_k: int = 3
    B_w: int = 2
    B_k: int = 2
    stencil_policy_w: str = "l1_ball_odd"
    stencil_policy_k: str = "l1_ball_even"
    radius_w: int = 1
    radius_k: int = 2
    include_zero_k: bool = False
    kernel_weights: dict[str, float] = field(default_factory=_default_kernel_weights)
    report_every: int = 1000
    device: torch.device | str = "cpu"

    def resolved_device(self) -> torch.device:
        return torch.device(self.device)

    @classmethod
    def from_dict(cls, base: "Params", overrides: dict[str, Any]) -> "Params":
        data = {**base.__dict__}
        data.update(overrides)
        return cls(**data)
