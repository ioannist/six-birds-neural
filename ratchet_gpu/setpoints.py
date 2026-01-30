from __future__ import annotations

from typing import Iterable, Tuple

import torch


def block_mean(grid: torch.Tensor, block: int) -> torch.Tensor:
    if block <= 0:
        raise ValueError("block must be > 0")
    if grid.ndim == 2:
        H, W = grid.shape
        if H % block != 0 or W % block != 0:
            raise ValueError("grid shape not divisible by block")
        return grid.reshape(H // block, block, W // block, block).mean(dim=(1, 3))
    if grid.ndim == 3:
        L, H, W = grid.shape
        if H % block != 0 or W % block != 0:
            raise ValueError("grid shape not divisible by block")
        return grid.reshape(L, H // block, block, W // block, block).mean(dim=(2, 4))
    raise ValueError("grid must be 2D or 3D")


def target_distance(a: torch.Tensor, b: torch.Tensor, p: float = 1.0) -> float:
    if a.shape != b.shape:
        raise ValueError("target_distance shape mismatch")
    diff = (a - b).abs().to(dtype=torch.float64)
    if p != 1.0:
        diff = diff.pow(p)
    return float(diff.mean().item())


def block_mask(mask: torch.Tensor, block: int, threshold: float = 0.0) -> torch.Tensor:
    if block <= 0:
        raise ValueError("block must be > 0")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    H, W = mask.shape
    if H % block != 0 or W % block != 0:
        raise ValueError("mask shape not divisible by block")
    pooled = mask.to(dtype=torch.float64).reshape(
        H // block, block, W // block, block
    ).mean(dim=(1, 3))
    return pooled > threshold


def masked_distance(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, p: float = 1.0
) -> float:
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("masked_distance shape mismatch")
    diff = (a - b).abs().to(dtype=torch.float64)
    if p != 1.0:
        diff = diff.pow(p)
    count = int(mask.sum().item())
    if count == 0:
        return 0.0
    return float(diff[mask].mean().item())


def pre_peak_post(
    values: Iterable[float],
    injury_window: int,
    injury_duration: int,
    last_m: int,
) -> Tuple[float, float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0, 0.0
    total_windows = len(vals)
    pre_idx = [i for i in range(injury_window - 2, injury_window) if i >= 1]
    haz_start = injury_window
    haz_end = min(total_windows, injury_window + injury_duration - 1)
    haz_idx = list(range(haz_start, haz_end + 1))
    post_start = max(1, total_windows - last_m + 1)
    post_idx = list(range(post_start, total_windows + 1))

    def _mean_at(idx: list[int]) -> float:
        if not idx:
            return 0.0
        return float(sum(vals[i - 1] for i in idx) / len(idx))

    pre = _mean_at(pre_idx)
    peak = max(vals[i - 1] for i in haz_idx) if haz_idx else pre
    post = _mean_at(post_idx)
    return pre, peak, post
