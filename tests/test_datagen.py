import numpy as np

from lgw import config as C
from lgw.datagen import make_clip


def test_anomaly_matches_control_before_injection():
    """Paired design: an anomalous clip and its control share pre-injection history."""
    for cond in C.CONDITIONS:
        a, _, _ = make_clip(11, cond)
        c, _, _ = make_clip(11, None)
        np.testing.assert_array_equal(a[: C.INJECT_FRAME], c[: C.INJECT_FRAME])


def test_anomaly_actually_changes_the_clip():
    c, _, _ = make_clip(11, None)
    for cond in C.CONDITIONS:
        a, _, _ = make_clip(11, cond)
        assert not np.array_equal(a[C.INJECT_FRAME:], c[C.INJECT_FRAME:]), cond


def test_frame_drop_holds_exactly_four_frames():
    f, _, held = make_clip(11, "frame_drop")
    assert held == [16, 18, 20, 22]
    for t in held:
        np.testing.assert_array_equal(f[t], f[t - 1])


def test_occluder_area_is_30_percent():
    side = int(round(np.sqrt(C.OCCLUDER_AREA_FRAC) * C.RES))
    assert abs(side * side / (C.RES ** 2) - C.OCCLUDER_AREA_FRAC) < 0.01


def test_disc_stays_in_frame():
    for cond in (None,) + C.CONDITIONS:
        for i in range(25):
            _, tr, _ = make_clip(i, cond)
            assert tr.min() >= C.DISC_RADIUS - 1e-6
            assert tr.max() <= C.RES - 1 - C.DISC_RADIUS + 1e-6
