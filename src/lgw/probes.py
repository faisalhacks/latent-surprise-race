"""Phase 1.2 - online geometry probes (prereg Sec. 2.3).

The only public entry point is a streaming object: you push `z_t` in and get
frame-`t` scores out. It holds no forward state, so lookahead is structurally
impossible, not merely avoided by convention. `tests/test_causality.py` verifies
prefix invariance.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import config as C

NAN = float("nan")


def _fro_sq_diff_lowrank(A: np.ndarray, B: np.ndarray,
                         dfa: int, dfb: int, bb: float | None = None) -> float:
    """||A'A/dfa - B'B/dfb||_F^2 for centred (k, D) factors, in O(k^2 D).

    Uses tr(XY) = ||.||_F^2 identities on the k-by-k Gram matrices instead of
    forming the D-by-D covariances.
    """
    aa = float(np.sum((A @ A.T) ** 2)) / (dfa * dfa)
    if bb is None:
        bb = float(np.sum((B @ B.T) ** 2)) / (dfb * dfb)
    ab = float(np.sum((A @ B.T) ** 2)) / (dfa * dfb)
    return max(aa - 2.0 * ab + bb, 0.0)


class OnlineGeometryProbes:
    """Streaming computation of probes A, B, C for one clip."""

    def __init__(self) -> None:
        self._recent: deque[np.ndarray] = deque(maxlen=max(C.ROLL_WIN, C.COV_WIN) + 1)
        self._base: list[np.ndarray] = []      # frames [0, COV_BASE_END)
        self._base_centred: np.ndarray | None = None
        self._base_bb: float | None = None
        self._t = -1

    def update(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=np.float64).ravel()
        self._t += 1
        t = self._t

        if t < C.COV_BASE_END:
            self._base.append(z)
            if t == C.COV_BASE_END - 1:          # freeze Sigma_base at t = 14
                Xb = np.stack(self._base)
                Ab = Xb - Xb.mean(axis=0, keepdims=True)
                self._base_centred = Ab
                dfb = Xb.shape[0] - 1
                self._base_bb = float(np.sum((Ab @ Ab.T) ** 2)) / (dfb * dfb)

        self._recent.append(z)
        hist = list(self._recent)                # hist[-1] is z_t

        out = {"probe_A": NAN, "probe_B": NAN, "probe_C": NAN}

        # Probe A: ||z_t - 2 z_{t-1} + z_{t-2}||
        if len(hist) >= 3:
            out["probe_A"] = float(np.linalg.norm(hist[-1] - 2.0 * hist[-2] + hist[-3]))

        # Probe B: 1 - cos(z_t, mean of previous ROLL_WIN states)
        if len(hist) >= C.ROLL_WIN + 1:
            ref = np.mean(np.stack(hist[-(C.ROLL_WIN + 1):-1]), axis=0)
            na, nb = np.linalg.norm(z), np.linalg.norm(ref)
            if na > 0 and nb > 0:
                out["probe_B"] = float(1.0 - np.dot(z, ref) / (na * nb))

        # Probe C: ||Cov(last COV_WIN states) - Sigma_base||_F
        if self._base_centred is not None and len(hist) >= C.COV_WIN:
            Xt = np.stack(hist[-C.COV_WIN:])
            At = Xt - Xt.mean(axis=0, keepdims=True)
            val = _fro_sq_diff_lowrank(
                At, self._base_centred,
                Xt.shape[0] - 1, self._base_centred.shape[0] - 1, self._base_bb,
            )
            out["probe_C"] = float(np.sqrt(val))

        return out


def run_clip(Z: np.ndarray) -> dict[str, np.ndarray]:
    """Convenience: stream a full (T, D) latent sequence through the probes."""
    mon = OnlineGeometryProbes()
    acc: dict[str, list[float]] = {k: [] for k in ("probe_A", "probe_B", "probe_C")}
    for t in range(Z.shape[0]):
        r = mon.update(Z[t])
        for k in acc:
            acc[k].append(r[k])
    return {k: np.asarray(v, dtype=np.float64) for k, v in acc.items()}


def fit_ridge_predictor(Z_calib: np.ndarray, lam: float = C.RIDGE_LAMBDA) -> np.ndarray:
    """Baseline L1 (prereg Sec. 2.2): ridge map [z_{t-1}; z_{t-2}] -> z_t.

    Fit on control-calibration clips ONLY. `Z_calib` is (N, T, D).
    Returns W of shape (2D, D).
    """
    N, T, D = Z_calib.shape
    X = np.concatenate([Z_calib[:, 1:T - 1, :], Z_calib[:, 0:T - 2, :]], axis=2)
    Y = Z_calib[:, 2:T, :]
    X = X.reshape(-1, 2 * D).astype(np.float64)
    Y = Y.reshape(-1, D).astype(np.float64)
    G = X.T @ X + lam * np.eye(2 * D)
    return np.linalg.solve(G, X.T @ Y)


def ridge_loss(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Per-frame L1 loss for one clip, (T,) with NaN for t < 2. Causal."""
    T, D = Z.shape
    out = np.full(T, NAN)
    Zf = Z.astype(np.float64)
    feat = np.concatenate([Zf[1:T - 1], Zf[0:T - 2]], axis=1)
    pred = feat @ W
    out[2:] = np.linalg.norm(Zf[2:] - pred, axis=1)
    return out
