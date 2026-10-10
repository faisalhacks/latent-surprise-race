"""Phase 4.3 probes must not report a score that only reflects clip identity."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from lgw import config as C                                  # noqa: E402
from lgw.mechanistic import (analyse_lag_window, collect_frames,  # noqa: E402
                             lag_window_slices, linear_probe)


def test_lag_window_is_between_the_two_alarms():
    dt_f = np.array([2, 5, 0])
    ev_f = np.array([True, True, False])
    dt_s = np.array([6, 7, 9])
    ev_s = np.array([True, True, True])
    sl = lag_window_slices(dt_f, ev_f, dt_s, ev_s)
    assert sl[0] == (C.EVAL_START + 2, C.EVAL_START + 6)
    assert sl[1] == (C.EVAL_START + 5, C.EVAL_START + 7)
    assert sl[2] is None                      # fast monitor never alarmed


def test_no_lag_window_when_fast_is_not_faster():
    sl = lag_window_slices(np.array([8]), np.array([True]),
                           np.array([3]), np.array([True]))
    assert sl == [None]


def test_uncensored_slow_monitor_extends_to_the_bound():
    sl = lag_window_slices(np.array([4]), np.array([True]),
                           np.array([C.CENSOR_DT]), np.array([False]))
    assert sl[0] == (C.EVAL_START + 4, C.EVAL_START + C.CENSOR_DT + 1)
    assert lag_window_slices(np.array([4]), np.array([True]),
                             np.array([C.CENSOR_DT]), np.array([False]),
                             require_both=True) == [None]


def test_collect_frames_tags_each_frame_with_its_clip():
    Z = np.arange(3 * C.N_FRAMES * 2).reshape(3, C.N_FRAMES, 2).astype(float)
    sl = [(15, 18), None, (20, 22)]
    X, y, g = collect_frames(Z, sl, np.array([0, 1, 2]))
    assert X.shape == (5, 2)
    assert list(g) == [0, 0, 0, 2, 2]
    assert list(y) == [0, 0, 0, 2, 2]


def test_probe_recovers_a_real_signal():
    """A genuinely decodable label should beat chance with a clean control."""
    rng = np.random.default_rng(0)
    n_clips, n_per, D = 40, 6, 24
    y_clip = rng.integers(0, 2, n_clips)
    X, y, g = [], [], []
    for i in range(n_clips):
        base = rng.normal(0, 1, D)
        sig = np.zeros(D); sig[0] = 4.0 * y_clip[i]
        X.append(base + sig + rng.normal(0, 0.3, (n_per, D)))
        y.append(np.full(n_per, y_clip[i])); g.append(np.full(n_per, i))
    r = linear_probe(np.concatenate(X), np.concatenate(y), np.concatenate(g))
    assert r["score"] > 0.8
    assert r["control_ok"], r


def test_probe_does_not_hallucinate_signal_from_clip_identity():
    """Labels unrelated to the latents must score at chance, not above it.

    Without clip-grouped splits a probe memorises which clip a frame came from
    and reports a high score for a label that carries no information.
    """
    rng = np.random.default_rng(1)
    n_clips, n_per, D = 40, 6, 24
    y_clip = rng.integers(0, 2, n_clips)
    X, y, g = [], [], []
    for i in range(n_clips):
        centre = rng.normal(0, 5, D)          # strong per-clip identity
        X.append(centre + rng.normal(0, 0.2, (n_per, D)))
        y.append(np.full(n_per, y_clip[i])); g.append(np.full(n_per, i))
    r = linear_probe(np.concatenate(X), np.concatenate(y), np.concatenate(g))
    assert r["lift_over_chance"] < 0.25, r    # no real signal to find


def test_empty_lag_window_is_reported_not_guessed():
    Z = np.zeros((5, C.N_FRAMES, 8))
    res = analyse_lag_window(Z, np.array(["gravity_flip"] * 5), [None] * 5)
    assert res["n_clips_with_lag"] == 0
    assert np.isnan(res["which_anomaly"]["score"])
    assert "note" in res["which_anomaly"]
