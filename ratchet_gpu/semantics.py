from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import numpy as np


def parse_rect_bounds(rect: str, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*", rect)
    if not match:
        raise ValueError(f"Invalid rect syntax: {rect}")
    r0, r1, c0, c1 = (int(match.group(i)) for i in range(1, 5))
    H, W = shape
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W or r0 >= r1 or c0 >= c1:
        raise ValueError("Rect bounds out of range")
    return r0, r1, c0, c1


def ring_masks_from_rect(
    rect: str, shape: Tuple[int, int], width: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if width < 1:
        raise ValueError("ring width must be >= 1")
    H, W = shape
    r0, r1, c0, c1 = parse_rect_bounds(rect, shape)
    region = np.zeros((H, W), dtype=bool)
    region[r0:r1, c0:c1] = True
    r0o = max(0, r0 - width)
    r1o = min(H, r1 + width)
    c0o = max(0, c0 - width)
    c1o = min(W, c1 + width)
    outer = np.zeros((H, W), dtype=bool)
    outer[r0o:r1o, c0o:c1o] = True
    ring = np.logical_and(outer, ~region)
    outside = ~(region | ring)
    return region, ring, outside


def ring_hist(
    motif_ids: np.ndarray, ring_mask: np.ndarray, num_motifs: int
) -> np.ndarray:
    ring_ids = motif_ids[ring_mask]
    if ring_ids.size == 0:
        return np.zeros((num_motifs,), dtype=np.float64)
    counts = np.bincount(ring_ids.ravel(), minlength=num_motifs).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def shift_ids_map(ids_map: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(ids_map, shift=(dy, dx), axis=(0, 1))


def semantic_effect(
    motif_ids_by_window: List[np.ndarray],
    mismatch_by_window: List[np.ndarray],
    windows: List[int],
    ring_mask: np.ndarray,
    num_motifs: int,
    hazard_start: int,
    hazard_duration: int,
    support_min: float,
    shuffle_n: int,
    shuffle_mode: str = "permute",
    shuffle_dxdy_max: int | None = None,
    shuffle_block_size: int | None = None,
    candidate_top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, float | int]:
    if len(motif_ids_by_window) != len(mismatch_by_window):
        raise ValueError("motif_ids_by_window and mismatch_by_window length mismatch")
    if len(motif_ids_by_window) < 2:
        return {
            "semantic_best": 0.0,
            "semantic_best_motif": -1,
            "semantic_support": 0.0,
            "semantic_z": 0.0,
            "semantic_shuf_mean": 0.0,
            "semantic_shuf_std": 0.0,
        }
    if rng is None:
        rng = np.random.default_rng()
    hazard_end = hazard_start + hazard_duration - 1
    idxs = [
        idx
        for idx, win in enumerate(windows)
        if hazard_start <= win <= hazard_end and idx + 1 < len(windows)
    ]
    if not idxs:
        return {
            "semantic_best": 0.0,
            "semantic_best_motif": -1,
            "semantic_support": 0.0,
            "semantic_z": 0.0,
            "semantic_shuf_mean": 0.0,
            "semantic_shuf_std": 0.0,
        }
    ring_count = int(ring_mask.sum())
    if ring_count == 0:
        return {
            "semantic_best": 0.0,
            "semantic_best_motif": -1,
            "semantic_support": 0.0,
            "semantic_z": 0.0,
            "semantic_shuf_mean": 0.0,
            "semantic_shuf_std": 0.0,
        }

    candidate_counts = np.zeros(num_motifs, dtype=np.float64)
    for idx, win in enumerate(windows):
        if win < hazard_start or hazard_start <= win <= hazard_end:
            ids = motif_ids_by_window[idx][ring_mask].ravel()
            if ids.size:
                candidate_counts += np.bincount(ids, minlength=num_motifs)
    candidate_support = candidate_counts / max(candidate_counts.sum(), 1.0)
    if candidate_top_k is None or candidate_top_k <= 0:
        candidate_ids = np.where(candidate_support >= support_min)[0]
    else:
        order = np.argsort(candidate_support)[::-1]
        candidate_ids = [
            int(idx)
            for idx in order[: int(candidate_top_k)]
            if candidate_support[idx] >= support_min
        ]

    sums = np.zeros(num_motifs, dtype=np.float64)
    counts = np.zeros(num_motifs, dtype=np.float64)
    for idx in idxs:
        ids = motif_ids_by_window[idx][ring_mask].ravel()
        delta = (mismatch_by_window[idx + 1] - mismatch_by_window[idx])[ring_mask].ravel()
        if ids.size == 0:
            continue
        counts += np.bincount(ids, minlength=num_motifs)
        sums += np.bincount(ids, weights=delta, minlength=num_motifs)

    support_hazard = counts / max(counts.sum(), 1.0)
    effects = np.zeros(num_motifs, dtype=np.float64)
    mask = counts > 0
    effects[mask] = sums[mask] / counts[mask]
    eligible = np.array([idx for idx in candidate_ids if counts[idx] > 0], dtype=int)
    if eligible.size == 0:
        return {
            "semantic_best": 0.0,
            "semantic_best_motif": -1,
            "semantic_support": 0.0,
            "semantic_z": 0.0,
            "semantic_p": 0.0,
            "semantic_shuf_mean": 0.0,
            "semantic_shuf_std": 0.0,
            "semantic_candidate_ids": [],
            "semantic_candidate_supports": [],
        }
    best_idx = int(eligible[np.argmin(effects[eligible])])
    semantic_best = float(effects[best_idx])
    semantic_support = float(candidate_support[best_idx])

    shuf_vals: List[float] = []
    if shuffle_n > 0:
        H, W = motif_ids_by_window[0].shape
        if shuffle_dxdy_max is None:
            dx_choices = list(range(W))
            dy_choices = list(range(H))
        else:
            dx_choices = list(range(-shuffle_dxdy_max, shuffle_dxdy_max + 1))
            dy_choices = list(range(-shuffle_dxdy_max, shuffle_dxdy_max + 1))
        for _ in range(int(shuffle_n)):
            sums = np.zeros(num_motifs, dtype=np.float64)
            counts = np.zeros(num_motifs, dtype=np.float64)
            for idx in idxs:
                ids_map = motif_ids_by_window[idx]
                delta = (mismatch_by_window[idx + 1] - mismatch_by_window[idx])[ring_mask].ravel()
                if ids_map.size == 0:
                    continue
                if shuffle_mode == "permute":
                    ids = ids_map[ring_mask].ravel()
                    perm = rng.permutation(ids.size)
                    ids = ids[perm]
                else:
                    if shuffle_mode == "block_shift":
                        step = int(shuffle_block_size or 1)
                        dx = rng.choice(dx_choices) * step
                        dy = rng.choice(dy_choices) * step
                    else:
                        dx = rng.choice(dx_choices)
                        dy = rng.choice(dy_choices)
                    ids = shift_ids_map(ids_map, dx, dy)[ring_mask].ravel()
                counts += np.bincount(ids, minlength=num_motifs)
                sums += np.bincount(ids, weights=delta, minlength=num_motifs)
            effects_sh = np.zeros(num_motifs, dtype=np.float64)
            mask_sh = counts > 0
            effects_sh[mask_sh] = sums[mask_sh] / counts[mask_sh]
            eligible_sh = np.array([idx for idx in candidate_ids if counts[idx] > 0], dtype=int)
            if eligible_sh.size == 0:
                shuf_vals.append(0.0)
            else:
                best_sh = float(np.min(effects_sh[eligible_sh]))
                shuf_vals.append(best_sh)

    if shuf_vals:
        mean_shuf = float(np.mean(shuf_vals))
        std_shuf = float(np.std(shuf_vals, ddof=1)) if len(shuf_vals) > 1 else 0.0
    else:
        mean_shuf = 0.0
        std_shuf = 0.0
    if std_shuf > 0:
        z_semantic = float((semantic_best - mean_shuf) / std_shuf)
    else:
        z_semantic = 0.0
    if shuf_vals:
        semantic_p = float(np.mean(np.array(shuf_vals) <= semantic_best))
    else:
        semantic_p = 0.0

    return {
        "semantic_best": semantic_best,
        "semantic_best_motif": best_idx,
        "semantic_support": semantic_support,
        "semantic_z": z_semantic,
        "semantic_p": semantic_p,
        "semantic_shuf_mean": mean_shuf,
        "semantic_shuf_std": std_shuf,
        "semantic_candidate_ids": [int(x) for x in candidate_ids],
        "semantic_candidate_supports": [float(candidate_support[x]) for x in candidate_ids],
    }


def motif_fraction_timeseries(
    motif_ids_by_window: List[np.ndarray],
    ring_mask: np.ndarray,
    motif_ids: List[int],
    window_indices: List[int] | None = None,
) -> np.ndarray:
    indices = window_indices if window_indices is not None else list(range(len(motif_ids_by_window)))
    if not motif_ids:
        return np.zeros((len(indices), 0), dtype=np.float64)
    ring_count = int(ring_mask.sum())
    if ring_count == 0:
        return np.zeros((len(indices), len(motif_ids)), dtype=np.float64)
    id_to_col = {mid: col for col, mid in enumerate(motif_ids)}
    out = np.zeros((len(indices), len(motif_ids)), dtype=np.float64)
    for row_idx, win_idx in enumerate(indices):
        ids = motif_ids_by_window[win_idx][ring_mask].ravel()
        if ids.size == 0:
            continue
        counts = np.bincount(ids, minlength=max(motif_ids) + 1).astype(np.float64)
        for mid, col in id_to_col.items():
            if mid < counts.size:
                out[row_idx, col] = counts[mid] / ring_count
    return out


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def predictive_semantic_score(
    x_tm: np.ndarray,
    y_t: np.ndarray,
    metric: str = "corr",
) -> Tuple[float, int, np.ndarray]:
    if x_tm.size == 0 or y_t.size == 0:
        return 0.0, -1, np.zeros((x_tm.shape[1] if x_tm.ndim == 2 else 0,), dtype=np.float64)
    if metric != "corr":
        raise ValueError(f"unsupported metric: {metric}")
    scores = np.zeros((x_tm.shape[1],), dtype=np.float64)
    for idx in range(x_tm.shape[1]):
        scores[idx] = _corrcoef(x_tm[:, idx], y_t)
    best_idx = int(np.argmin(scores))
    return float(scores[best_idx]), best_idx, scores


def predictive_semantic_pvalue(
    x_tm: np.ndarray,
    y_t: np.ndarray,
    shift_n: int,
    rng: np.random.Generator,
    metric: str = "corr",
) -> Tuple[float, float, float, float, int]:
    best_score, best_idx, _scores = predictive_semantic_score(x_tm, y_t, metric=metric)
    if shift_n <= 0 or x_tm.shape[0] == 0:
        return best_score, 1.0, 0.0, 0.0, best_idx
    null_vals: List[float] = []
    for _ in range(int(shift_n)):
        shift = int(rng.integers(1, x_tm.shape[0] + 1))
        shifted = np.roll(x_tm, shift=shift, axis=0)
        null_best, _null_idx, _ = predictive_semantic_score(shifted, y_t, metric=metric)
        null_vals.append(null_best)
    null_vals_arr = np.array(null_vals, dtype=np.float64)
    pval = float(np.mean(null_vals_arr <= best_score))
    mean_null = float(null_vals_arr.mean())
    std_null = float(null_vals_arr.std(ddof=1)) if null_vals_arr.size > 1 else 0.0
    return best_score, pval, mean_null, std_null, best_idx


def hazard_center(rect: str, shape: Tuple[int, int]) -> Tuple[float, float]:
    r0, r1, c0, c1 = parse_rect_bounds(rect, shape)
    cy = (r0 + r1 - 1) / 2.0
    cx = (c0 + c1 - 1) / 2.0
    return cy, cx


def pref_axis_map(shape: Tuple[int, int], center: Tuple[float, float]) -> np.ndarray:
    H, W = shape
    cy, cx = center
    yy, xx = np.indices((H, W))
    dx = np.abs(xx - cx)
    dy = np.abs(yy - cy)
    pref = np.where(dx >= dy, 1.0, -1.0)
    return pref.astype(np.float64)


def alignment_score(
    axis_bias: np.ndarray, pref_axis: np.ndarray, mask: np.ndarray
) -> float:
    if axis_bias.shape != pref_axis.shape:
        raise ValueError("axis_bias and pref_axis shape mismatch")
    if axis_bias.shape != mask.shape:
        raise ValueError("axis_bias and mask shape mismatch")
    if mask.sum() == 0:
        return 0.0
    values = axis_bias[mask] * pref_axis[mask]
    return float(np.mean(values))


def radial_focus_score(radial_map: np.ndarray, mask: np.ndarray) -> float:
    if radial_map.shape != mask.shape:
        raise ValueError("radial_map and mask shape mismatch")
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(radial_map[mask]))


def alignment_delta(
    scores_pre: List[float], scores_hazard: List[float]
) -> float:
    if not scores_pre or not scores_hazard:
        return 0.0
    return float(np.mean(scores_hazard) - np.mean(scores_pre))


def alignment_shift_null(
    axis_bias_by_window: List[np.ndarray],
    pref_axis: np.ndarray,
    ring_mask: np.ndarray,
    pre_idx: List[int],
    hazard_idx: List[int],
    shuffle_n: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    if shuffle_n <= 0:
        return 1.0, 0.0, 0.0
    H, W = pref_axis.shape
    if H == 0 or W == 0:
        return 1.0, 0.0, 0.0
    obs_pre = [
        alignment_score(axis_bias_by_window[i], pref_axis, ring_mask) for i in pre_idx
    ]
    obs_haz = [
        alignment_score(axis_bias_by_window[i], pref_axis, ring_mask)
        for i in hazard_idx
    ]
    obs = alignment_delta(obs_pre, obs_haz)
    null_vals: List[float] = []
    for _ in range(int(shuffle_n)):
        dx = int(rng.integers(0, W))
        dy = int(rng.integers(0, H))
        scores_pre = [
            alignment_score(np.roll(axis_bias_by_window[i], (dy, dx), axis=(0, 1)), pref_axis, ring_mask)
            for i in pre_idx
        ]
        scores_haz = [
            alignment_score(np.roll(axis_bias_by_window[i], (dy, dx), axis=(0, 1)), pref_axis, ring_mask)
            for i in hazard_idx
        ]
        null_vals.append(alignment_delta(scores_pre, scores_haz))
    null_arr = np.array(null_vals, dtype=np.float64)
    p_val = float(np.mean(null_arr >= obs))
    mean_null = float(null_arr.mean())
    std_null = float(null_arr.std(ddof=1)) if null_arr.size > 1 else 0.0
    return p_val, mean_null, std_null


def accuracy_score(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have the same shape")
    if preds.size == 0:
        return 0.0
    return float(np.mean(preds == labels))


def balanced_accuracy_score(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have the same shape")
    if preds.size == 0:
        return 0.0
    unique = np.unique(labels)
    if unique.size == 0:
        return 0.0
    recalls: List[float] = []
    for value in unique:
        mask = labels == value
        if mask.sum() == 0:
            continue
        recalls.append(float(np.mean(preds[mask] == labels[mask])))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def valid_nontrivial_circular_shifts(labels: np.ndarray) -> List[int]:
    labels = np.asarray(labels)
    n = int(labels.size)
    if n < 2:
        return []
    shifts: List[int] = []
    for shift in range(1, n):
        if not np.array_equal(np.roll(labels, shift=shift), labels):
            shifts.append(shift)
    return shifts


def generate_phrase_schedule(
    mode: str,
    hazard_len: int,
    token_hold_windows: int,
    phrase_start: int,
) -> List[str]:
    if hazard_len <= 0:
        return []
    if token_hold_windows < 1:
        raise ValueError("token_hold_windows must be >= 1")
    if phrase_start not in (0, 1):
        raise ValueError("phrase_start must be 0 or 1")
    mode = mode.strip()
    if mode not in {"alternating", "chunked"}:
        raise ValueError(f"unsupported phrase mode {mode}")

    base_len = int(math.ceil(hazard_len / token_hold_windows))
    start_token = "OUT" if phrase_start == 0 else "IN"
    other_token = "IN" if start_token == "OUT" else "OUT"

    if mode == "alternating":
        base = [start_token if i % 2 == 0 else other_token for i in range(base_len)]
    else:
        half = base_len // 2
        base = [start_token] * half + [other_token] * (base_len - half)

    expanded: List[str] = []
    for token in base:
        expanded.extend([token] * token_hold_windows)
    return expanded[:hazard_len]


def shift_null_p_value_for_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    shuffle_n: int,
    rng: np.random.Generator | None = None,
    skip_identical: bool = True,
) -> Tuple[float, float, float, float, int]:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have the same shape")
    obs = accuracy_score(preds, labels)
    if shuffle_n <= 0:
        return obs, 1.0, 0.0, 0.0, 0
    if skip_identical:
        valid_shifts = valid_nontrivial_circular_shifts(labels)
    else:
        valid_shifts = list(range(1, int(labels.size)))
    if not valid_shifts:
        return obs, 1.0, 0.0, 0.0, 0
    if rng is None:
        rng = np.random.default_rng()
    shifts = rng.choice(valid_shifts, size=int(shuffle_n), replace=True)
    null_vals = np.zeros(int(shuffle_n), dtype=np.float64)
    for i, shift in enumerate(shifts):
        null_vals[i] = accuracy_score(preds, np.roll(labels, shift=shift))
    count_ge = int(np.sum(null_vals >= obs))
    p_val = (count_ge + 1) / (null_vals.size + 1)
    mean_null = float(null_vals.mean())
    std_null = float(null_vals.std(ddof=1)) if null_vals.size > 1 else 0.0
    return obs, float(p_val), mean_null, std_null, int(null_vals.size)


def radial_focus_shift_null(
    radial_by_window: List[np.ndarray],
    ring_mask: np.ndarray,
    pre_idx: List[int],
    hazard_idx: List[int],
    shuffle_n: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    if shuffle_n <= 0:
        return 1.0, 0.0, 0.0
    H, W = ring_mask.shape
    if H == 0 or W == 0:
        return 1.0, 0.0, 0.0
    obs_pre = [radial_focus_score(radial_by_window[i], ring_mask) for i in pre_idx]
    obs_haz = [radial_focus_score(radial_by_window[i], ring_mask) for i in hazard_idx]
    obs = alignment_delta(obs_pre, obs_haz)
    null_vals: List[float] = []
    for _ in range(int(shuffle_n)):
        dx = int(rng.integers(0, W))
        dy = int(rng.integers(0, H))
        scores_pre = [
            radial_focus_score(np.roll(radial_by_window[i], (dy, dx), axis=(0, 1)), ring_mask)
            for i in pre_idx
        ]
        scores_haz = [
            radial_focus_score(np.roll(radial_by_window[i], (dy, dx), axis=(0, 1)), ring_mask)
            for i in hazard_idx
        ]
        null_vals.append(alignment_delta(scores_pre, scores_haz))
    null_arr = np.array(null_vals, dtype=np.float64)
    p_val = float(np.mean(null_arr >= obs))
    mean_null = float(null_arr.mean())
    std_null = float(null_arr.std(ddof=1)) if null_arr.size > 1 else 0.0
    return p_val, mean_null, std_null


def shift_null_corr(
    scores: np.ndarray,
    labels: np.ndarray,
    shuffle_n: int,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    n = int(scores.size)
    if n < 2:
        return 0.0, 1.0, 0.0, 0.0

    def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
        xs = float(np.std(x))
        ys = float(np.std(y))
        if xs <= 0 or ys <= 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    obs = _pearson_corr(scores, labels)
    if shuffle_n <= 0:
        return obs, 1.0, 0.0, 0.0

    if rng is None:
        rng = np.random.default_rng()
    target = int(shuffle_n)
    null_vals: List[float] = []
    attempts = 0
    max_attempts = target * 10
    while len(null_vals) < target and attempts < max_attempts:
        shift = int(rng.integers(1, n))
        shifted = np.roll(labels, shift=shift)
        if np.array_equal(shifted, labels):
            attempts += 1
            continue
        null_vals.append(_pearson_corr(scores, shifted))
        attempts += 1
    if not null_vals:
        return obs, 1.0, 0.0, 0.0
    null_arr = np.array(null_vals, dtype=np.float64)
    count_ge = int(np.sum(null_arr >= obs))
    p_val = (count_ge + 1) / (null_arr.size + 1)
    mean_null = float(null_arr.mean())
    std_null = float(null_arr.std(ddof=1)) if null_arr.size > 1 else 0.0
    return obs, float(p_val), mean_null, std_null
