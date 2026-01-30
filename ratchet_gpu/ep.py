from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Tuple

import torch

from .state import State


@dataclass
class EPTracker:
    beta: float
    ep_micro_total: float = 0.0
    ep_micro_window: float = 0.0
    steps: int = 0
    accepted: int = 0
    window_steps: int = 0

    def record(
        self,
        accepted: bool,
        ep_inc: float,
        proposals: int = 1,
        accepted_count: int | None = None,
    ) -> None:
        if accepted_count is None:
            accepted_count = 1 if accepted else 0
        self.steps += int(proposals)
        self.window_steps += int(proposals)
        if accepted_count:
            self.accepted += int(accepted_count)
            self.ep_micro_total += float(ep_inc)
            self.ep_micro_window += float(ep_inc)

    def reset_window(self) -> None:
        self.ep_micro_window = 0.0
        self.window_steps = 0

    @property
    def accepted_frac(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.accepted / self.steps


@dataclass
class StrobeTracker:
    bins_sigma: int = 11
    bins_s: int = 6
    bins_w: int = 11
    signature: str = "mag_s"
    transitions: Dict[Tuple[int, ...], Dict[Tuple[int, ...], int]] = field(
        default_factory=dict
    )
    prev_bin: Tuple[int, ...] | None = None
    total: int = 0

    def record_state(self, state: State) -> None:
        b = _coarse_bin(state, self.signature, self.bins_sigma, self.bins_s, self.bins_w)
        if self.prev_bin is not None:
            self.transitions.setdefault(self.prev_bin, {})
            self.transitions[self.prev_bin][b] = self.transitions[self.prev_bin].get(b, 0) + 1
            self.total += 1
        self.prev_bin = b

    def unique_states_count(self) -> int:
        states = set(self.transitions.keys())
        for outs in self.transitions.values():
            states.update(outs.keys())
        return len(states)

    def bidirectional_edge_count(self) -> int:
        seen = set()
        count = 0
        for a, outs in self.transitions.items():
            for b, cab in outs.items():
                if cab <= 0 or a == b:
                    continue
                if self.transitions.get(b, {}).get(a, 0) > 0:
                    edge = tuple(sorted((a, b)))
                    if edge not in seen:
                        seen.add(edge)
                        count += 1
        return count

    def current_map(self) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]:
        if self.total == 0:
            return {}
        currents: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        seen = set()
        for a, outs in self.transitions.items():
            for b, cab in outs.items():
                if cab <= 0 or a == b:
                    continue
                edge = tuple(sorted((a, b)))
                if edge in seen:
                    continue
                cba = self.transitions.get(b, {}).get(a, 0)
                net = cab - cba
                currents[edge] = net / self.total
                seen.add(edge)
        return currents

    def symgap(self) -> float:
        if self.total == 0:
            return 0.0
        seen = set()
        gap = 0.0
        for a, outs in self.transitions.items():
            for b, cab in outs.items():
                if cab <= 0 or a == b:
                    continue
                edge = tuple(sorted((a, b)))
                if edge in seen:
                    continue
                cba = self.transitions.get(b, {}).get(a, 0)
                gap += abs(cab - cba)
                seen.add(edge)
        return gap / self.total

    def current_l2(self) -> float:
        if self.total == 0:
            return 0.0
        seen = set()
        sq = 0.0
        for a, outs in self.transitions.items():
            for b, cab in outs.items():
                if cab <= 0 or a == b:
                    continue
                edge = tuple(sorted((a, b)))
                if edge in seen:
                    continue
                cba = self.transitions.get(b, {}).get(a, 0)
                net = cab - cba
                sq += net * net
                seen.add(edge)
        return math.sqrt(sq) / self.total

    def top_currents(self, n: int = 10) -> list[dict]:
        return self.current_items(n)

    def current_items(self, max_edges: int | None = 10) -> list[dict]:
        currents = self.current_map()
        if not currents:
            return []
        def key_fn(item: tuple[tuple[tuple[int, ...], tuple[int, ...]], float]) -> float:
            return abs(item[1])
        items = sorted(currents.items(), key=key_fn, reverse=True)
        if max_edges is not None:
            items = items[:max_edges]
        out = []
        for edge, j in items:
            a, b = edge
            cab = self.transitions.get(a, {}).get(b, 0)
            cba = self.transitions.get(b, {}).get(a, 0)
            out.append(
                {
                    "u": list(a),
                    "v": list(b),
                    "j": float(j),
                    "count_ab": int(cab),
                    "count_ba": int(cba),
                }
            )
        return out

    def state_counts(self) -> Dict[Tuple[int, ...], int]:
        counts: Dict[Tuple[int, ...], int] = {}
        for a, outs in self.transitions.items():
            for b, cab in outs.items():
                if cab <= 0:
                    continue
                counts[a] = counts.get(a, 0) + cab
                counts[b] = counts.get(b, 0) + cab
        return counts

    def top_states(self, n: int = 10) -> list[tuple[tuple[int, ...], int]]:
        counts = self.state_counts()
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return items[:n]

    def ep_rate(self) -> float:
        if self.total == 0:
            return 0.0
        ep = 0.0
        for a, outs in self.transitions.items():
            total_a = sum(outs.values())
            if total_a == 0:
                continue
            p_a = total_a / self.total
            for b, count_ab in outs.items():
                if count_ab == 0:
                    continue
                count_ba = self.transitions.get(b, {}).get(a, 0)
                if count_ba == 0:
                    continue
                t_ab = count_ab / total_a
                t_ba = count_ba / sum(self.transitions.get(b, {}).values())
                if t_ab > 0 and t_ba > 0:
                    ep += p_a * t_ab * torch.log(torch.tensor(t_ab / t_ba)).item()
        return float(ep)


def _coarse_bin(state: State, signature: str, bins_sigma: int, bins_s: int, bins_w: int) -> Tuple[int, ...]:
    sigma = state.sigma.to(dtype=torch.float32)
    s = state.s.to(dtype=torch.float32)

    layer0 = float(sigma[0].mean().item())
    layer1 = float(sigma[1].mean().item()) if state.layers > 1 else 0.0
    s_mean = float(s.mean().item())

    def bin_val(value: float, low: float, high: float, bins: int) -> int:
        if bins <= 1:
            return 0
        scaled = (value - low) / (high - low)
        idx = int(round(scaled * (bins - 1)))
        return max(0, min(bins - 1, idx))

    b0 = bin_val(layer0, -1.0, 1.0, bins_sigma)
    b1 = bin_val(layer1, -1.0, 1.0, bins_sigma)
    if signature == "mag_stag":
        Q = 5
        scale = max(4, int(round(math.sqrt(state.N) / 2)))
        idx0, idx1 = state.color_indices[0]
        m0 = float(sigma[0].sum().item())
        ms0 = float(sigma[0][idx0].sum().item() - sigma[0][idx1].sum().item())
        if state.layers > 1:
            m1 = float(sigma[1].sum().item())
            ms1 = float(sigma[1][idx0].sum().item() - sigma[1][idx1].sum().item())
        else:
            m1 = 0.0
            ms1 = 0.0
        def q(val: float) -> int:
            return max(-Q, min(Q, int(round(val / scale))))
        return (q(m0), q(ms0), q(m1), q(ms1))
    if signature == "mag_wmass":
        if state.K_W == 0 or state.lattice.d < 2:
            b2 = 0
        else:
            offsets = state.R_W
            mask_axis0 = (offsets.abs().sum(dim=1) == 1) & (offsets[:, 0].abs() == 1)
            mask_axis1 = (offsets.abs().sum(dim=1) == 1) & (offsets[:, 1].abs() == 1)
            W = state.W.to(dtype=torch.float32)
            mass0 = W[:, :, mask_axis0].sum()
            mass1 = (
                W[:, :, mask_axis1].sum() if mask_axis1.any() else torch.tensor(0.0, device=W.device)
            )
            denom = torch.maximum(mass0 + mass1, torch.tensor(1.0, device=W.device))
            aniso = float((mass0 - mass1).item() / denom.item())
            b2 = bin_val(aniso, -1.0, 1.0, bins_w)
    else:
        b2 = bin_val(s_mean, 0.0, max(1.0, float(state.params.l_s)), bins_s)
    return (b0, b1, b2)
