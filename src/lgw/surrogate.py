"""A cheap CPU stand-in world model, used ONLY to validate the harness.

It exercises exactly the code paths a real adapter uses - causal window,
tubelet-aligned context window, two encoder passes, a pooled latent, a
prediction loss - so the plumbing, the probes and the statistics can be
verified end to end without spending GPU budget.

Its numbers are NOT a scientific result and must never be reported as one.
It also doubles as the minimal worked example for ADAPTERS.md: everything a new
architecture needs is the two hooks below.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .adapters import BaseWindowAdapter, register


@register("surrogate")
class SurrogateAdapter(BaseWindowAdapter):
    """Fixed random tubelet projection, with a persistence predictor."""

    def __init__(self, dim: int = 64, seed: int = 0, down: int = 16,
                 t_ctx: int = C.T_CTX, loss_from_frame: int = C.EVAL_START):
        super().__init__(dim=dim, name="surrogate", t_ctx=t_ctx,
                         loss_from_frame=loss_from_frame)
        self.down = down
        rng = np.random.default_rng(seed)
        self.R = rng.normal(0, 1.0 / np.sqrt(2 * down * down),
                            size=(2 * down * down, dim))

    def _embed_tubelet(self, pair: np.ndarray) -> np.ndarray:
        """(B, 2, H, W) uint8 -> (B, dim)"""
        B = pair.shape[0]
        k = pair.shape[-1] // self.down
        x = pair.astype(np.float64) / 255.0
        x = x.reshape(B, 2, self.down, k, self.down, k).mean(axis=(3, 5))
        return np.tanh(x.reshape(B, -1) @ self.R)

    # ---- adapter contract ----
    def encode_window(self, windows: np.ndarray):
        z = self._embed_tubelet(windows[:, -self.tubelet:])
        return z, z                      # the latent is its own L0 target

    def predict_final_step(self, ctx_windows: np.ndarray):
        # the context window's final tubelet is a repeat of f_{t-2}, so this is
        # a persistence prediction: "nothing changes"
        return self._embed_tubelet(ctx_windows[:, -self.tubelet:])
