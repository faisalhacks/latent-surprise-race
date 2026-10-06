"""A cheap CPU stand-in world model, used ONLY to validate the harness.

It exercises exactly the code paths the V-JEPA 2 adapter uses (causal window,
tubelet-aligned context window, two encoder passes, mean-pooled state), so the
plumbing, the probes and the statistics can be verified end-to-end without
spending GPU budget. Its numbers are NOT a scientific result and must never be
reported as one.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .extract import causal_window, context_window


class SurrogateAdapter:
    name = "surrogate"

    def __init__(self, dim: int = 64, seed: int = 0, down: int = 16):
        self.dim = dim
        self.down = down
        rng = np.random.default_rng(seed)
        # fixed random "tubelet patch embedding": (2 frames x down x down) -> dim
        self.R = rng.normal(0, 1.0 / np.sqrt(2 * down * down), size=(2 * down * down, dim))

    def _embed_tubelet(self, pair: np.ndarray) -> np.ndarray:
        """pair: (B, 2, H, W) uint8 -> (B, dim)"""
        B = pair.shape[0]
        k = C.RES // self.down
        x = pair.astype(np.float64) / 255.0
        x = x.reshape(B, 2, self.down, k, self.down, k).mean(axis=(3, 5))
        return np.tanh(x.reshape(B, -1) @ self.R)

    def process(self, clips: np.ndarray):
        B, T = clips.shape[0], clips.shape[1]
        Z = np.empty((B, T, self.dim), dtype=np.float32)
        L0 = np.full((B, T), np.nan, dtype=np.float64)
        for t in range(T):
            full = np.stack([causal_window(clips[b], t)[-C.TUBELET:] for b in range(B)])
            ctx = np.stack([context_window(clips[b], t)[-C.TUBELET:] for b in range(B)])
            z = self._embed_tubelet(full)
            zhat = self._embed_tubelet(ctx)          # persistence prediction
            Z[:, t] = z
            L0[:, t] = np.linalg.norm(z - zhat, axis=1)
        return Z, L0
