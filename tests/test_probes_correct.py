"""Probe values must equal their closed-form definitions in the pre-registration."""
import numpy as np

from lgw import config as C
from lgw.probes import fit_ridge_predictor, ridge_loss, run_clip


def test_probes_match_naive_definitions():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(C.N_FRAMES, 48))
    got = run_clip(Z)
    Sb = np.cov(Z[: C.COV_BASE_END], rowvar=False, ddof=1)
    for t in range(C.EVAL_START, C.N_FRAMES):
        a = np.linalg.norm(Z[t] - 2 * Z[t - 1] + Z[t - 2])
        ref = Z[t - C.ROLL_WIN:t].mean(axis=0)
        b = 1.0 - np.dot(Z[t], ref) / (np.linalg.norm(Z[t]) * np.linalg.norm(ref))
        St = np.cov(Z[t - C.COV_WIN + 1:t + 1], rowvar=False, ddof=1)
        c = np.sqrt(np.sum((St - Sb) ** 2))
        assert abs(got["probe_A"][t] - a) < 1e-9
        assert abs(got["probe_B"][t] - b) < 1e-12
        assert abs(got["probe_C"][t] - c) / max(c, 1e-12) < 1e-7


def test_ridge_baseline_is_causal_and_exact():
    rng = np.random.default_rng(4)
    Zc = rng.normal(size=(8, C.N_FRAMES, 12))
    W = fit_ridge_predictor(Zc)
    assert W.shape == (24, 12)
    Z = rng.normal(size=(C.N_FRAMES, 12))
    L = ridge_loss(Z, W)
    assert np.all(np.isnan(L[:2]))
    for t in range(2, C.N_FRAMES):
        exp = np.linalg.norm(Z[t] - np.concatenate([Z[t - 1], Z[t - 2]]) @ W)
        assert abs(L[t] - exp) < 1e-9
