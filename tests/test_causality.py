"""Causality: frame-t computation cannot see frame t+1. Tested, not asserted."""
import numpy as np

from lgw import config as C
from lgw.datagen import make_clip
from lgw.extract import causal_window, context_window, run_extraction
from lgw.probes import OnlineGeometryProbes, run_clip
from lgw.surrogate import SurrogateAdapter


def test_causal_window_never_sees_future():
    clip = np.arange(C.N_FRAMES, dtype=np.uint8).reshape(-1, 1, 1) * np.ones((1, 4, 4), np.uint8)
    for t in range(C.N_FRAMES):
        assert causal_window(clip, t).max() <= t
        assert context_window(clip, t).max() <= max(t - 2, 0)


def test_probe_prefix_invariance():
    """Scores for frames 0..t must not change when later frames are appended."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(C.N_FRAMES, 32))
    full = run_clip(Z)
    for t in (16, 20, 25, C.N_FRAMES - 1):
        pref = run_clip(Z[: t + 1])
        for k, v in pref.items():
            a, b = v, full[k][: t + 1]
            np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
            np.testing.assert_allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-12)


def test_probe_future_perturbation_does_not_change_past():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(C.N_FRAMES, 32))
    Z2 = Z.copy()
    Z2[20:] += 1000.0
    a, b = run_clip(Z), run_clip(Z2)
    for k in a:
        x, y = a[k][:20], b[k][:20]
        m = np.isfinite(x)
        np.testing.assert_allclose(x[m], y[m], rtol=1e-12)


def test_adapter_future_perturbation_does_not_change_past():
    """End-to-end: perturbing frames > t leaves z_t and loss_t identical."""
    clip, _, _ = make_clip(3, "gravity_flip")
    adapter = SurrogateAdapter()
    cut = 18
    pert = clip.copy()
    pert[cut + 1:] = np.asarray(255, np.uint8) - pert[cut + 1:]
    out_a = run_extraction(clip[None], adapter, progress=False)
    out_b = run_extraction(pert[None], adapter, progress=False)
    np.testing.assert_array_equal(out_a["Z"][0, : cut + 1], out_b["Z"][0, : cut + 1])
    np.testing.assert_allclose(out_a["loss_L0"][0, : cut + 1],
                               out_b["loss_L0"][0, : cut + 1], rtol=1e-12)
    for k in ("probe_A", "probe_B", "probe_C"):
        x, y = out_a[k][0, : cut + 1], out_b[k][0, : cut + 1]
        m = np.isfinite(x)
        np.testing.assert_allclose(x[m], y[m], rtol=1e-12)


def test_probe_c_baseline_frozen_at_frame_14():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(C.N_FRAMES, 16))
    mon = OnlineGeometryProbes()
    for t in range(C.COV_BASE_END):
        mon.update(Z[t])
    frozen = mon._base_centred.copy()
    for t in range(C.COV_BASE_END, C.N_FRAMES):
        mon.update(Z[t] * 50.0)
    np.testing.assert_array_equal(frozen, mon._base_centred)
