"""Frozen experimental constants. See PREREGISTRATION.md.

Nothing in this module may change after the first inference run. The values are
mirrored in the pre-registration document; `tests/test_config_frozen.py` pins
them so that an accidental edit fails CI.
"""
from __future__ import annotations

# ---- dataset (prereg Sec. 1) ----
SEED = 20261006
RES = 64
N_FRAMES = 30
INJECT_FRAME = 15
DISC_RADIUS = 5.0
GRAVITY = 0.25
SPEED_RANGE = (1.5, 3.0)

CONDITIONS = ("gravity_flip", "friction_high", "occlusion", "frame_drop")
N_PER_CONDITION = 250
N_CONTROL_CALIB = 100
N_CONTROL_EVAL = 100

FRICTION_RETENTION = 0.8 ** (1.0 / 5.0)   # per-frame velocity decay (prereg 0.2)
FRICTION_RESTITUTION = 0.8
OCCLUDER_AREA_FRAC = 0.30
FRAME_DROP_SPAN = (15, 23)                # underlying frames [15, 23)

# ---- monitors (prereg Sec. 2) ----
ROLL_WIN = 5          # Probe B rolling-mean window
COV_WIN = 5           # Probe C trailing covariance window
COV_BASE_END = 15     # Sigma_base from frames [0, 15)
RIDGE_LAMBDA = 1.0    # baseline L1

MONITORS = ("loss_L0", "loss_L1", "probe_A", "probe_B", "probe_C")
PROBES = ("probe_A", "probe_B", "probe_C")
BASELINES = ("loss_L0", "loss_L1")

# ---- evaluation (prereg Sec. 3) ----
EVAL_START = 15
EVAL_END = 30         # exclusive -> frames 15..29
FPR_TARGET = 0.01     # threshold = 99th percentile of control_calib
CENSOR_DT = EVAL_END - EVAL_START - 1   # 14
ALPHA = 0.01
MIN_MEDIAN_GAIN = 1.0  # frames, Phase 3 gate

# ---- model (prereg Sec. 4) ----
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
T_CTX = 16
TUBELET = 2
MODEL_RES = 256
