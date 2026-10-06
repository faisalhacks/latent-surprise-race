"""Thresholds must be a function of control-calibration data alone."""
import numpy as np

from lgw import config as C
from lgw import stats as S


def test_threshold_is_deterministic():
    rng = np.random.default_rng(5)
    calib = rng.normal(size=(C.N_CONTROL_CALIB, C.N_FRAMES))
    assert S.threshold(calib) == S.threshold(calib.copy())


def test_threshold_uses_only_eval_window():
    rng = np.random.default_rng(6)
    calib = rng.normal(size=(50, C.N_FRAMES))
    base = S.threshold(calib)
    calib[:, : C.EVAL_START] += 1e6
    assert S.threshold(calib) == base


def test_threshold_hits_target_fpr_on_calibration():
    rng = np.random.default_rng(7)
    calib = rng.normal(size=(400, C.N_FRAMES))
    thr = S.threshold(calib)
    per_frame = (calib[:, C.EVAL_START:C.EVAL_END] > thr).mean()
    assert abs(per_frame - C.FPR_TARGET) < 0.003


def test_censoring_marks_non_alarming_clips():
    scores = np.zeros((10, C.N_FRAMES))
    dt, ev = S.alarm_times(scores, thr=1.0)
    assert not ev.any() and (dt == C.CENSOR_DT).all()
    scores[0, C.EVAL_START + 3] = 2.0
    dt, ev = S.alarm_times(scores, thr=1.0)
    assert ev[0] and dt[0] == 3
