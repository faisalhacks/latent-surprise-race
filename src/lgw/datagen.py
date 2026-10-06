"""Phase 1.1 - synthetic bouncing-disc dataset with injected anomalies.

Deterministic given `config.SEED`. See PREREGISTRATION.md Sec. 1.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def _reflect(p: float, v: float, lo: float, hi: float, restitution: float):
    """Elastic (or damped) reflection into [lo, hi], robust to large |v|."""
    for _ in range(8):
        if p < lo:
            p = 2.0 * lo - p
            v = -v * restitution
        elif p > hi:
            p = 2.0 * hi - p
            v = -v * restitution
        else:
            break
    return float(np.clip(p, lo, hi)), v


def _simulate(rng: np.random.Generator, condition: str | None) -> np.ndarray:
    """Return disc centre positions, shape (N_FRAMES, 2) as (x, y)."""
    lo = C.DISC_RADIUS
    hi = C.RES - 1.0 - C.DISC_RADIUS
    p = np.array([rng.uniform(lo + 2, hi - 2), rng.uniform(lo + 2, hi - 2)])
    speed = rng.uniform(*C.SPEED_RANGE)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    v = np.array([speed * np.cos(ang), speed * np.sin(ang)])

    out = np.empty((C.N_FRAMES, 2), dtype=np.float64)
    out[0] = p
    for t in range(1, C.N_FRAMES):
        anomalous = condition is not None and t >= C.INJECT_FRAME
        g = C.GRAVITY
        retention = 1.0
        restitution = 1.0
        if anomalous and condition == "gravity_flip":
            g = -C.GRAVITY
        elif anomalous and condition == "friction_high":
            retention = C.FRICTION_RETENTION
            restitution = C.FRICTION_RESTITUTION

        v = v * retention
        v[1] += g
        p = p + v
        p[0], v[0] = _reflect(p[0], v[0], lo, hi, restitution)
        p[1], v[1] = _reflect(p[1], v[1], lo, hi, restitution)
        out[t] = p
    return out


_YY, _XX = np.mgrid[0:C.RES, 0:C.RES].astype(np.float64)


def _render(traj: np.ndarray) -> np.ndarray:
    """Soft-edged white disc on black, uint8 (N_FRAMES, RES, RES)."""
    dx = _XX[None] - traj[:, 0, None, None]
    dy = _YY[None] - traj[:, 1, None, None]
    dist = np.sqrt(dx * dx + dy * dy)
    # 1 px soft edge keeps sub-pixel motion visible to the encoder
    alpha = np.clip(C.DISC_RADIUS + 0.5 - dist, 0.0, 1.0)
    return (alpha * 255.0).astype(np.uint8)


def _apply_occlusion(frames: np.ndarray, traj: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    side = int(round(np.sqrt(C.OCCLUDER_AREA_FRAC) * C.RES))  # 35 px -> 29.9% area
    # centre the occluder on a point of the disc's path after injection, so it
    # genuinely intersects the trajectory (prereg Sec. 1)
    anchor = traj[rng.integers(C.INJECT_FRAME + 3, C.N_FRAMES)]
    x0 = int(np.clip(round(anchor[0] - side / 2), 0, C.RES - side))
    y0 = int(np.clip(round(anchor[1] - side / 2), 0, C.RES - side))
    frames = frames.copy()
    frames[C.INJECT_FRAME:, y0:y0 + side, x0:x0 + side] = 0
    return frames


def _apply_frame_drop(frames: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Hold the previous delivered frame in place of every other frame."""
    frames = frames.copy()
    held: list[int] = []
    a, b = C.FRAME_DROP_SPAN
    for t in range(a, b):
        if (t - a) % 2 == 1:
            frames[t] = frames[t - 1]
            held.append(t)
    return frames, held


def make_clip(clip_index: int, condition: str | None):
    """Build one clip. `condition=None` -> control."""
    rng = np.random.default_rng([C.SEED, clip_index])
    traj = _simulate(rng, condition)
    frames = _render(traj)
    held: list[int] = []
    if condition == "occlusion":
        frames = _apply_occlusion(frames, traj, rng)
    elif condition == "frame_drop":
        frames, held = _apply_frame_drop(frames)
    return frames, traj, held


def build_split(split: str) -> dict:
    """split in {'anomaly', 'control_calib', 'control_eval'}."""
    frames_all, conds, ids, trajs = [], [], [], []
    if split == "anomaly":
        idx = 0
        for cond in C.CONDITIONS:
            for _ in range(C.N_PER_CONDITION):
                f, tr, _ = make_clip(idx, cond)
                frames_all.append(f); conds.append(cond); ids.append(idx); trajs.append(tr)
                idx += 1
    elif split in ("control_calib", "control_eval"):
        n = C.N_CONTROL_CALIB if split == "control_calib" else C.N_CONTROL_EVAL
        # disjoint seed ranges keep calibration and evaluation controls independent
        base = 1_000_000 if split == "control_calib" else 2_000_000
        for j in range(n):
            f, tr, _ = make_clip(base + j, None)
            frames_all.append(f); conds.append("control"); ids.append(base + j); trajs.append(tr)
    else:
        raise ValueError(split)
    return {
        "frames": np.stack(frames_all),
        "condition": np.array(conds),
        "clip_id": np.array(ids, dtype=np.int64),
        "traj": np.stack(trajs).astype(np.float32),
        "split": split,
    }


def main(out_dir: str = "data") -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    for split in ("anomaly", "control_calib", "control_eval"):
        d = build_split(split)
        path = os.path.join(out_dir, f"{split}.npz")
        np.savez_compressed(path, **{k: v for k, v in d.items() if k != "split"})
        print(f"{split:15s} {d['frames'].shape} -> {path} "
              f"({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
