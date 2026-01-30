#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np


def _parse_keys(value: str) -> List[str]:
    return [k.strip() for k in value.split(",") if k.strip()]


def _load_frames(npz_dir: Path, seed: int, key: str, max_frames: int) -> List[np.ndarray]:
    pattern = f"seed{seed}_win"
    files = sorted(
        (p for p in npz_dir.glob("*.npz") if p.name.startswith(pattern)),
        key=lambda p: p.name,
    )
    if max_frames > 0:
        files = files[:max_frames]
    frames: List[np.ndarray] = []
    for path in files:
        with np.load(path) as npz:
            if key not in npz:
                raise KeyError(f"Key {key} not found in {path}")
            arr = npz[key]
        if arr.ndim != 2:
            raise ValueError(f"Key {key} in {path} is not 2D (shape={arr.shape})")
        frames.append(arr.astype(np.float32))
    return frames


def _scale_frames(frames: Iterable[np.ndarray]) -> List[np.ndarray]:
    stacked = np.stack(list(frames), axis=0)
    vmin = float(stacked.min())
    vmax = float(stacked.max())
    if vmax <= vmin:
        return [np.zeros_like(stacked[0], dtype=np.uint8) for _ in range(stacked.shape[0])]
    norm = (stacked - vmin) / (vmax - vmin)
    return [np.clip(frame * 255.0, 0, 255).astype(np.uint8) for frame in norm]


def render_npz_dir(
    npz_dir: Path,
    out_dir: Path,
    seed: int,
    keys: List[str],
    max_frames: int,
    fps: int,
) -> List[Path]:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "imageio is required for rendering. Install with: pip install imageio pillow"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for key in keys:
        frames = _load_frames(npz_dir, seed, key, max_frames)
        if not frames:
            raise ValueError(f"No frames found in {npz_dir} for seed {seed}")
        scaled = _scale_frames(frames)
        rgb_frames = [np.stack([f, f, f], axis=-1) for f in scaled]
        out_path = out_dir / f"seed{seed}_{key}.gif"
        imageio.mimsave(out_path, rgb_frames, duration=1.0 / max(fps, 1))
        outputs.append(out_path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Render NPZ snapshots into GIFs.")
    parser.add_argument("--npz-dir", required=True, help="Directory with seed*win*.npz files.")
    parser.add_argument("--out-dir", default="", help="Output directory for renders.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--keys", default="sigma_l0,k_axis_bias_i0")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (npz_dir.parent / "renders")
    keys = _parse_keys(args.keys)

    outputs = render_npz_dir(npz_dir, out_dir, args.seed, keys, args.max_frames, args.fps)
    print("RENDERS:", ", ".join(str(p) for p in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
