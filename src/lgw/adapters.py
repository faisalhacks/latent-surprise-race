"""Model-agnostic adapter base and registry.

The causal loop - which frames frame `t` may see, where the L0 context window
stops, which frames get a prediction loss - is identical for every architecture
and is implemented once, here. A new world model only has to say how to encode
a window and how to predict its final step.

Adding a model
--------------
    from lgw.adapters import BaseWindowAdapter, register

    @register("my_model")
    class MyAdapter(BaseWindowAdapter):
        def __init__(self, **kw):
            super().__init__(dim=512, name="my_model", **kw)
            self.net = load_my_model().eval()

        def encode_window(self, windows):
            # windows: (B, t_ctx, H, W) uint8, no frame later than t
            h = self.net.encode(windows)          # (B, n_tokens, D)
            return h.mean(1), h                   # (pooled latent, L0 target)

        def predict_final_step(self, ctx_windows):
            # ctx_windows: same shape, final tubelet overwritten, so nothing
            # here has seen frame t-1 or t
            return self.net.predict_next(ctx_windows)   # same space as target

That is the whole contract. `process()`, the probes, the thresholds, the
survival analysis and the gate are then identical across architectures, which
is what makes a cross-architecture comparison meaningful rather than a
collection of incomparable pipelines.

See ADAPTERS.md for the full walkthrough and the correctness tests any new
adapter must pass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from . import config as C
from .extract import causal_window, context_window

REGISTRY: dict[str, Callable[..., "BaseWindowAdapter"]] = {}


def register(name: str):
    """Class decorator adding an adapter to the registry."""
    def deco(cls):
        if name in REGISTRY:
            raise ValueError("adapter already registered: " + name)
        REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def build_adapter(name: str, **kwargs) -> "BaseWindowAdapter":
    if name not in REGISTRY:
        raise KeyError("unknown adapter {!r}; available: {}".format(
            name, sorted(REGISTRY)))
    return REGISTRY[name](**kwargs)


def available_adapters() -> list[str]:
    return sorted(REGISTRY)


class BaseWindowAdapter(ABC):
    """Shared strictly-online extraction loop.

    Subclasses must not look at any frame beyond what they are handed; the
    windows passed in already enforce that, and
    `tests/test_adapter_contract.py` verifies it for every registered adapter.
    """

    registry_name: str = "base"

    def __init__(self, dim: int, name: str, t_ctx: int = C.T_CTX,
                 tubelet: int = C.TUBELET,
                 loss_from_frame: int = C.EVAL_START):
        self.dim = int(dim)
        self.name = name
        self.t_ctx = int(t_ctx)
        self.tubelet = int(tubelet)
        # L0 is never scored before the evaluation window, so the context pass
        # and the predictor pass are skipped there. Saves a third of the
        # forwards and changes no reported quantity.
        self.loss_from_frame = int(loss_from_frame)

    # ---- the two hooks a model must implement ----
    @abstractmethod
    def encode_window(self, windows: np.ndarray) -> tuple[np.ndarray, Any]:
        """(B, t_ctx, H, W) uint8 -> (latent (B, dim), L0 target or None)."""

    @abstractmethod
    def predict_final_step(self, ctx_windows: np.ndarray) -> Any:
        """Prediction of the final step, in the same space as the L0 target.

        Return None to declare that this model has no native prediction loss;
        baseline L1 still applies, since it is computed from the latents.
        """

    # ---- optional hook ----
    def l0(self, pred: Any, target: Any) -> np.ndarray:
        """Mean squared L2 distance, averaged over all non-batch axes."""
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        return ((p - t) ** 2).reshape(p.shape[0], -1).mean(axis=1)

    # ---- the shared causal loop ----
    def process(self, clips: np.ndarray):
        """(B, T, H, W) uint8 -> (Z (B, T, dim) float32, L0 (B, T) float64)."""
        B, T = clips.shape[0], clips.shape[1]
        Z = np.empty((B, T, self.dim), dtype=np.float32)
        L0 = np.full((B, T), np.nan, dtype=np.float64)

        for t in range(T):
            full = np.stack([causal_window(clips[b], t, self.t_ctx)
                             for b in range(B)])
            z, target = self.encode_window(full)
            Z[:, t] = np.asarray(z, dtype=np.float32)

            if t >= self.loss_from_frame and target is not None:
                ctx = np.stack([context_window(clips[b], t, self.t_ctx,
                                               self.tubelet)
                                for b in range(B)])
                pred = self.predict_final_step(ctx)
                if pred is not None:
                    L0[:, t] = self.l0(pred, target)
        return Z, L0
