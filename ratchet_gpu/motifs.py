from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class MotifBins:
    edges: Dict[str, np.ndarray]
    bins: Dict[str, int]


def quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    flat = np.asarray(values, dtype=float).ravel()
    if flat.size == 0:
        raise ValueError("empty values for binning")
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(flat, qs)
    if np.all(np.diff(edges) > 0):
        return edges
    vmin = float(flat.min())
    vmax = float(flat.max())
    if vmin == vmax:
        eps = 1e-9
        return np.linspace(vmin - eps, vmax + eps, bins + 1)
    return np.linspace(vmin, vmax, bins + 1)


def symmetric_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    flat = np.asarray(values, dtype=float).ravel()
    if flat.size == 0:
        raise ValueError("empty values for binning")
    max_abs = float(np.max(np.abs(flat)))
    if max_abs == 0.0:
        max_abs = 1e-6
    return np.linspace(-max_abs, max_abs, bins + 1)


def build_bins(
    baseline_features: Dict[str, List[np.ndarray]],
    bins_by_key: Dict[str, int],
    edges_by_key: Dict[str, np.ndarray] | None = None,
) -> MotifBins:
    edges: Dict[str, np.ndarray] = {}
    bins: Dict[str, int] = {}
    for key, bins_count in bins_by_key.items():
        if key not in baseline_features or not baseline_features[key]:
            raise ValueError(f"missing baseline features for {key}")
        values = np.concatenate([arr.ravel() for arr in baseline_features[key]], axis=0)
        if edges_by_key is not None and key in edges_by_key:
            edges[key] = np.asarray(edges_by_key[key], dtype=float)
        else:
            edges[key] = quantile_edges(values, bins_count)
        bins[key] = bins_count
    return MotifBins(edges=edges, bins=bins)


def quantize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges[1:-1], right=False)
    bins = np.clip(bins, 0, len(edges) - 2)
    return bins.astype(np.int64)


def motif_ids(
    features: Dict[str, np.ndarray],
    bins: MotifBins,
) -> np.ndarray:
    keys = list(bins.edges.keys())
    if not keys:
        raise ValueError("no bins configured")
    shape = next(iter(features.values())).shape
    for key in keys:
        if key not in features:
            raise ValueError(f"missing feature {key}")
        if features[key].shape != shape:
            raise ValueError("feature shapes must match")
    digits = [quantize(features[key], bins.edges[key]) for key in keys]
    mult = 1
    ids = np.zeros(shape, dtype=np.int64)
    for key, digs in zip(keys, digits):
        ids += digs * mult
        mult *= bins.bins[key]
    return ids


def motif_hist(ids: np.ndarray, num_motifs: int) -> np.ndarray:
    counts = np.bincount(ids.ravel(), minlength=num_motifs).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def motif_histogram(ids: np.ndarray, mask: np.ndarray, num_motifs: int) -> np.ndarray:
    if mask.shape != ids.shape:
        raise ValueError("mask and ids must have the same shape")
    selected = ids[mask]
    if selected.size == 0:
        return np.zeros((num_motifs,), dtype=np.float64)
    counts = np.bincount(selected.ravel(), minlength=num_motifs).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def top_n_coverage(p: np.ndarray, top_n: int) -> float:
    if p.size == 0:
        return 0.0
    top = np.sort(p)[::-1][: max(1, top_n)]
    return float(top.sum())


def motif_entropy(p: np.ndarray) -> float:
    if p.size == 0:
        return 0.0
    mask = p > 0
    vals = p[mask]
    return float(-np.sum(vals * np.log(vals)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * (_kl(p, m) + _kl(q, m))


def dictionary_weights(p_out: np.ndarray, p_in: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p_out = np.asarray(p_out, dtype=np.float64)
    p_in = np.asarray(p_in, dtype=np.float64)
    if p_out.shape != p_in.shape:
        raise ValueError("p_out and p_in must have the same shape")
    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return np.log((p_out + eps) / (p_in + eps))


def dictionary_score(hist: np.ndarray, weights: np.ndarray) -> float:
    hist = np.asarray(hist, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if hist.shape != weights.shape:
        raise ValueError("hist and weights must have the same shape")
    return float(np.dot(hist, weights))


def dictionary_threshold(scores_out: np.ndarray, scores_in: np.ndarray) -> Tuple[float, int]:
    scores_out = np.asarray(scores_out, dtype=np.float64)
    scores_in = np.asarray(scores_in, dtype=np.float64)
    if scores_out.size == 0 or scores_in.size == 0:
        return 0.0, 1
    mean_out = float(np.mean(scores_out))
    mean_in = float(np.mean(scores_in))
    threshold = 0.5 * (mean_out + mean_in)
    direction = 1 if mean_out >= mean_in else -1
    return threshold, direction


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    return float(np.sum(np.abs(p - q)))


def motif_dictionary_eval(
    in_hists: np.ndarray,
    out_hists: np.ndarray,
    shuffle_n: int,
    rng: np.random.Generator | None = None,
) -> Dict[str, float | int]:
    if in_hists.shape != out_hists.shape:
        raise ValueError("in_hists and out_hists must have the same shape")
    if in_hists.ndim != 2:
        raise ValueError("hist arrays must be 2D (T, num_motifs)")
    total_windows = in_hists.shape[0]
    if total_windows < 2:
        return {
            "dict_delta": 0.0,
            "dict_p": 1.0,
            "dict_delta_shuf_mean": 0.0,
            "dict_delta_shuf_std": 0.0,
            "dict_eval_windows": 0,
        }
    if rng is None:
        rng = np.random.default_rng()
    split = total_windows // 2
    if split < 1 or split >= total_windows:
        split = max(1, total_windows - 1)
    in_train = in_hists[:split]
    out_train = out_hists[:split]
    in_eval = in_hists[split:]
    out_eval = out_hists[split:]
    d = np.mean(in_train, axis=0) - np.mean(out_train, axis=0)
    eval_delta = np.mean(np.sum(d * in_eval, axis=1) - np.sum(d * out_eval, axis=1))

    null_vals = np.zeros(shuffle_n, dtype=np.float64)
    for i in range(shuffle_n):
        swaps = rng.random(in_eval.shape[0]) < 0.5
        in_swapped = np.where(swaps[:, None], out_eval, in_eval)
        out_swapped = np.where(swaps[:, None], in_eval, out_eval)
        null_vals[i] = np.mean(np.sum(d * in_swapped, axis=1) - np.sum(d * out_swapped, axis=1))
    count_ge = int(np.sum(null_vals >= eval_delta))
    p_val = (count_ge + 1) / (shuffle_n + 1)
    return {
        "dict_delta": float(eval_delta),
        "dict_p": float(p_val),
        "dict_delta_shuf_mean": float(np.mean(null_vals)) if shuffle_n > 0 else 0.0,
        "dict_delta_shuf_std": float(np.std(null_vals)) if shuffle_n > 0 else 0.0,
        "dict_eval_windows": int(in_eval.shape[0]),
    }


def propagation_score(
    motif_maps: List[np.ndarray],
    motifs: List[int],
    shift_max: int,
) -> Tuple[float, int, Tuple[int, int]]:
    if len(motif_maps) < 2 or not motifs:
        return 0.0, -1, (0, 0)
    best_score = 0.0
    best_motif = -1
    best_shift = (0, 0)
    for motif in motifs:
        best_avg = 0.0
        best_shift_m = (0, 0)
        for dx in range(-shift_max, shift_max + 1):
            for dy in range(-shift_max, shift_max + 1):
                if dx == 0 and dy == 0:
                    continue
                corrs: List[float] = []
                for t in range(len(motif_maps) - 1):
                    m1 = motif_maps[t] == motif
                    m2 = motif_maps[t + 1] == motif
                    if m1.sum() == 0 or m2.sum() == 0:
                        continue
                    m2s = np.roll(m2, shift=(dy, dx), axis=(0, 1))
                    denom = float(np.sqrt(m1.sum() * m2.sum()))
                    if denom == 0:
                        continue
                    corr = float((m1 & m2s).sum() / denom)
                    corrs.append(corr)
                if corrs:
                    avg = float(np.mean(corrs))
                    if avg > best_avg:
                        best_avg = avg
                        best_shift_m = (dx, dy)
        if best_avg > best_score:
            best_score = best_avg
            best_motif = motif
            best_shift = best_shift_m
    return best_score, best_motif, best_shift


def split_condition_counts(
    windows: List[int],
    hazard_start: int,
    hazard_duration: int,
    histograms: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(windows) != len(histograms):
        raise ValueError("windows and histograms length mismatch")
    pre = None
    haz = None
    post = None
    haz_end = hazard_start + hazard_duration - 1
    for win, hist in zip(windows, histograms):
        if pre is None:
            pre = np.zeros_like(hist, dtype=np.float64)
            haz = np.zeros_like(hist, dtype=np.float64)
            post = np.zeros_like(hist, dtype=np.float64)
        if win < hazard_start:
            pre += hist
        elif hazard_start <= win <= haz_end:
            haz += hist
        else:
            post += hist
    if pre is None:
        raise ValueError("no histograms provided")
    return pre, haz, post


def transition_counts(
    prev_ids: np.ndarray,
    next_ids: np.ndarray,
    n_motifs: int,
) -> np.ndarray:
    if prev_ids.shape != next_ids.shape:
        raise ValueError("prev_ids and next_ids must have the same shape")
    if n_motifs <= 0:
        raise ValueError("n_motifs must be > 0")
    prev_flat = prev_ids.ravel().astype(np.int64)
    next_flat = next_ids.ravel().astype(np.int64)
    if prev_flat.size == 0:
        return np.zeros((n_motifs, n_motifs), dtype=np.int64)
    pairs = prev_flat * n_motifs + next_flat
    counts = np.bincount(pairs, minlength=n_motifs * n_motifs).astype(np.int64)
    return counts.reshape((n_motifs, n_motifs))


def transition_counts_over_time(
    motif_ids_by_t: List[np.ndarray],
    n_motifs: int,
) -> np.ndarray:
    if len(motif_ids_by_t) < 2:
        return np.zeros((n_motifs, n_motifs), dtype=np.int64)
    total = np.zeros((n_motifs, n_motifs), dtype=np.int64)
    for t in range(len(motif_ids_by_t) - 1):
        total += transition_counts(motif_ids_by_t[t], motif_ids_by_t[t + 1], n_motifs)
    return total


def transition_jsd(Ca: np.ndarray, Cb: np.ndarray, eps: float = 1e-12) -> float:
    if Ca.shape != Cb.shape:
        raise ValueError("Ca and Cb must have the same shape")
    pa = Ca.astype(np.float64)
    pb = Cb.astype(np.float64)
    pa = pa + eps
    pb = pb + eps
    pa /= pa.sum()
    pb /= pb.sum()
    return jsd(pa, pb)


def conditional_entropy_from_counts(C: np.ndarray, eps: float = 1e-12) -> float:
    counts = C.astype(np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("C must be square")
    if counts.sum() <= 0:
        return 0.0
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = counts / np.maximum(row_sums, eps)
    mask = counts > 0
    logp = np.zeros_like(probs)
    logp[mask] = np.log(probs[mask])
    ent = -np.sum(counts * logp)
    return float(ent / max(counts.sum(), eps))


def topk_transition_mass(C: np.ndarray, k: int = 10) -> float:
    if C.size == 0:
        return 0.0
    flat = C.astype(np.float64).ravel()
    total = flat.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(k))
    top = np.sort(flat)[::-1][:k]
    return float(top.sum() / total)
