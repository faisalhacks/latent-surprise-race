"""Phase 2 - strictly online extraction loop (model-agnostic).

The windowing helpers here are the single place where "what may frame t see"
is decided, so causality is auditable in one spot.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from . import config as C
from .probes import run_clip


def causal_window(clip: np.ndarray, t: int, t_ctx: int = C.T_CTX) -> np.ndarray:
    """Frames [t-t_ctx+1 .. t], left-padded by repeating frame 0.

    Contains no frame later than `t`. The final tubelet spans (t-1, t).
    """
    if t < 0 or t >= clip.shape[0]:
        raise IndexError(t)
    idx = np.arange(t - t_ctx + 1, t + 1)
    return clip[np.clip(idx, 0, None)]


def context_window(clip: np.ndarray, t: int, t_ctx: int = C.T_CTX,
                   tubelet: int = C.TUBELET) -> np.ndarray:
    """`causal_window` with the final tubelet overwritten by a repeat of f_{t-2}.

    Tubelets 0..6 of the result are exactly frames t-15..t-2, and no token in
    them can attend to frame t-1 or t (prereg Sec. 2.2, baseline L0).
    """
    w = causal_window(clip, t, t_ctx).copy()
    w[-tubelet:] = clip[max(t - tubelet, 0)]
    return w


class WorldModelAdapter(Protocol):
    """A frozen video world model, wrapped for online probing."""
    dim: int
    name: str

    def process(self, clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, T, H, W) uint8 -> (Z (B, T, D) float32, loss_L0 (B, T) float64).

        Z[b, t] must be a function of clips[b, :t+1] only.
        """
        ...


def run_extraction(clips: np.ndarray, adapter: "WorldModelAdapter",
                   batch_size: int = 8, progress: bool = True) -> dict:
    """Extract latents, L0 loss and geometry probes for every clip."""
    N, T = clips.shape[0], clips.shape[1]
    Z = np.empty((N, T, adapter.dim), dtype=np.float32)
    L0 = np.full((N, T), np.nan, dtype=np.float64)
    probes = {k: np.full((N, T), np.nan) for k in ("probe_A", "probe_B", "probe_C")}

    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        z, l0 = adapter.process(clips[sl])
        Z[sl] = z.astype(np.float32)
        L0[sl] = l0
        for j in range(z.shape[0]):
            r = run_clip(z[j].astype(np.float64))
            for k in probes:
                probes[k][i + j] = r[k]
        if progress:
            print(f"  [{adapter.name}] {sl.stop}/{N} clips", flush=True)

    return {"Z": Z, "loss_L0": L0, **probes}
