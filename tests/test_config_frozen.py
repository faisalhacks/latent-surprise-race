"""Pins the frozen protocol constants so an accidental edit fails CI."""
from lgw import config as C


def test_frozen_values():
    assert (C.SEED, C.RES, C.N_FRAMES, C.INJECT_FRAME) == (20261006, 64, 30, 15)
    assert C.CONDITIONS == ("gravity_flip", "friction_high", "occlusion", "frame_drop")
    assert (C.N_PER_CONDITION, C.N_CONTROL_CALIB, C.N_CONTROL_EVAL) == (250, 100, 100)
    assert (C.ROLL_WIN, C.COV_WIN, C.COV_BASE_END) == (5, 5, 15)
    assert (C.EVAL_START, C.EVAL_END, C.CENSOR_DT) == (15, 30, 14)
    assert (C.FPR_TARGET, C.ALPHA, C.MIN_MEDIAN_GAIN) == (0.01, 0.01, 1.0)
    assert (C.T_CTX, C.TUBELET) == (16, 2)
    assert C.MODEL_ID == "facebook/vjepa2-vitl-fpc64-256"
