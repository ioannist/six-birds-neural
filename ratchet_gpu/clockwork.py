from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _signed_k(idx: int, size: int) -> int:
    return idx if idx <= size // 2 else idx - size


@dataclass(frozen=True)
class TravelingModeResult:
    best_k: Tuple[int, int]
    omega: float
    r2: float
    power_ratio: float
    travel_score: float


@dataclass(frozen=True)
class PhaseStructureResult:
    freq_idx: int
    temporal_snr: float
    phase_grad: float
    phase_coherence: float
    phase_score: float


def traveling_mode_score(frames: np.ndarray, omega_min: float = 0.1) -> TravelingModeResult:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3:
        raise ValueError("frames must have shape [T,H,W]")
    T, H, W = frames.shape
    if T < 3:
        return TravelingModeResult((0, 0), 0.0, 0.0, 0.0, 0.0)

    demean = frames - frames.mean(axis=(1, 2), keepdims=True)
    F = np.fft.fft2(demean, axes=(1, 2))
    power_total = np.sum(np.abs(F) ** 2) - np.sum(np.abs(F[:, 0, 0]) ** 2)
    if power_total <= 0:
        return TravelingModeResult((0, 0), 0.0, 0.0, 0.0, 0.0)

    t = np.arange(T, dtype=np.float64)
    best = TravelingModeResult((0, 0), 0.0, 0.0, 0.0, 0.0)

    for ky in range(H):
        for kx in range(W):
            if ky == 0 and kx == 0:
                continue
            c_t = F[:, ky, kx]
            power_k = np.sum(np.abs(c_t) ** 2)
            if power_k <= 0:
                continue
            phi = np.unwrap(np.angle(c_t))
            slope, intercept = np.polyfit(t, phi, 1)
            if abs(slope) < omega_min:
                continue
            phi_pred = slope * t + intercept
            ss_res = np.sum((phi - phi_pred) ** 2)
            ss_tot = np.sum((phi - phi.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            power_ratio = float(power_k / power_total)
            score = power_ratio * r2
            if score > best.travel_score:
                best = TravelingModeResult(
                    (_signed_k(ky, H), _signed_k(kx, W)),
                    float(slope),
                    float(r2),
                    float(power_ratio),
                    float(score),
                )

    return best


def phase_structure_score(frames: np.ndarray) -> PhaseStructureResult:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3:
        raise ValueError("frames must have shape [T,H,W]")
    T, H, W = frames.shape
    if T < 3:
        return PhaseStructureResult(0, 0.0, 0.0, 1.0, 0.0)

    centered = frames - frames.mean(axis=0, keepdims=True)
    G = np.fft.fft(centered, axis=0)
    power_total = np.sum(np.abs(G[1:]) ** 2)
    if power_total <= 0:
        return PhaseStructureResult(0, 0.0, 0.0, 1.0, 0.0)

    mean_amp = np.mean(np.abs(G[1:]), axis=(1, 2))
    freq_idx = int(np.argmax(mean_amp)) + 1
    P = G[freq_idx]

    temporal_snr = float(np.sum(np.abs(P) ** 2) / power_total)
    dx = np.angle(np.conj(P[:, :-1]) * P[:, 1:])
    dy = np.angle(np.conj(P[:-1, :]) * P[1:, :])
    phase_grad = float(0.5 * (np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / np.pi)

    amp = np.abs(P)
    mask = amp > 0
    if np.any(mask):
        unit = P[mask] / amp[mask]
        phase_coherence = float(np.abs(unit.mean()))
    else:
        phase_coherence = 1.0

    score = temporal_snr * phase_grad * (1.0 - phase_coherence)
    return PhaseStructureResult(freq_idx, temporal_snr, phase_grad, phase_coherence, float(score))


def fabric_scores(frames: np.ndarray, omega_min: float = 0.1) -> Dict[str, float]:
    travel = traveling_mode_score(frames, omega_min=omega_min)
    phase = phase_structure_score(frames)
    return {
        "travel_score": travel.travel_score,
        "travel_r2": travel.r2,
        "travel_omega": travel.omega,
        "travel_power_ratio": travel.power_ratio,
        "travel_k0": travel.best_k[0],
        "travel_k1": travel.best_k[1],
        "phase_score": phase.phase_score,
        "phase_grad": phase.phase_grad,
        "phase_coherence": phase.phase_coherence,
        "phase_temporal_snr": phase.temporal_snr,
        "phase_freq_idx": phase.freq_idx,
        "fabric_score": max(travel.travel_score, phase.phase_score),
    }
