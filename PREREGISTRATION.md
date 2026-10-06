# Pre-Registration: Latent Geometry vs. Prediction Error in Video World Models

**Frozen:** 2026-10-06
**Status:** FROZEN. No definition in this document may be altered after the first
inference run. Any change must be recorded in `DEVIATIONS.md` with a timestamp
and justification, and reported in the paper.

**Hypothesis (H1).** Sudden changes in physics, occlusion, or effective frame
rate cause measurable shifts in the geometry of a frozen video world model's
internal representations *before* its own prediction error spikes.

**Primary claim under test.** For at least one geometry probe `P`, the median
Time-to-Alarm `median(dt_P)` is lower than `median(dt_loss)` by >= 1 frame, with
Wilcoxon signed-rank `p < 0.01` after Holm correction.

---

## 0. Protocol ambiguities resolved before freezing

The source protocol contained three internal inconsistencies. They are resolved
here, *before* any code ran, and the resolutions are themselves frozen.

**(0.1) "Constant velocity" vs. "gravity suddenly inverted".** These cannot both
hold: a constant-velocity baseline has no gravity to invert. **Resolution:** the
baseline includes a constant downward gravity `g0 = 0.25 px/frame^2` with
perfectly elastic collisions (restitution 1.0). The dynamics are therefore
energy-conserving and statistically stationary, so "baseline physics" is
well-defined, and "gravity inverted" (`g -> -g0`) is literally implementable.

**(0.2) "Friction increased to 0.8" has no immediate observable effect** if
applied only at wall collisions, since the next collision may be >15 frames
after injection, leaving Time-to-Alarm undefined for much of the sample.
**Resolution:** the friction anomaly applies *both* a per-frame linear drag
(velocity decays to 0.8 of its value over 5 frames, i.e. `v <- 0.8^(1/5) v` each
frame) *and* a bounce restitution of 0.8. The effect is observable from the
first post-injection frame.

**(0.3) "Extract the latent state `z_t` at each frame" is under-determined for
V-JEPA 2,** whose patch embedding uses a 2-frame tubelet, so no representation
is a function of a single frame. **Resolution:** see Section 2.1. `z_t` is the
spatially mean-pooled final-tubelet token of the encoder applied to the causal
window ending at frame `t`. It is a function of frames `<= t` only.

---

## 1. Dataset (Section 1.1 of protocol)

Synthetic grayscale clips, **64x64 px, 30 frames**, one disc of radius 5 px
bouncing inside the frame.

- **Baseline physics:** gravity `g0 = 0.25 px/frame^2` downward, restitution
  1.0, no drag. Initial position uniform in the interior; initial speed
  `~U[1.5, 3.0] px/frame`; initial direction uniform on the circle.
- **Injection frame:** `t* = 15` for every anomalous clip.
- **Anomaly conditions (250 clips each, 1000 total):**
  | id | definition from `t >= 15` |
  |---|---|
  | `gravity_flip` | `g <- -g0` (inverted) |
  | `friction_high` | per-frame `v <- 0.8^(1/5) v`; restitution `1.0 -> 0.8` |
  | `occlusion` | static black square, area = 30% of frame (side 35 px), drawn over all later frames, positioned to intersect the disc's baseline path |
  | `frame_drop` | for underlying frames 15..22, every other frame is not delivered; the previously delivered frame is held in its place (4 held frames), normal delivery resumes at frame 23 |
- **Control:** 200 clips, baseline physics throughout, no anomaly.
- **Control split (leakage control):** controls are partitioned by seed into
  `control_calib` (100 clips, used *only* to set thresholds) and
  `control_eval` (100 clips, used *only* to report false-alarm rates).
  Thresholds are never set on data used to measure false alarms, and never on
  anomalous data.
- **Seed:** 20261006. Generation is deterministic given the seed.

## 2. Monitors (Section 1.2 of protocol)

Every monitor emits one scalar per frame `t` and is a function of frames `<= t`
only. This is enforced in code by a streaming interface and verified by a
prefix-invariance test (`tests/test_causality.py`).

### 2.1 Latent state
`z_t in R^D` = spatial mean of the encoder's final-tubelet patch tokens, for the
encoder applied to the window `W_t = (f_{t-15}, ..., f_t)` (left-padded by
repeating `f_0` when `t < 15`). `D = 1024` for ViT-L. The final tubelet spans
frames `(t-1, t)`, so `z_t` depends on no frame later than `t`.

### 2.2 Baseline monitors (prediction error)
- **L0 - model's own prediction loss.** Mean squared L2 distance between the
  predictor's output for the final tubelet and that tubelet's encoder
  representation. Context is tubelets 0..6 of `W_t` (frames `<= t-2`),
  obtained from a *separate* encoder pass on a window whose final tubelet is
  replaced by a repeat of `f_{t-2}`, so that no context token attends to frame
  `t-1` or `t`. The target is the final-tubelet tokens of `W_t`. This is the
  model's pretraining objective evaluated causally.
- **L1 - information-matched prediction loss.** `||z_t - W [z_{t-1}; z_{t-2}]||`
  where `W` is ridge-regressed (`lambda = 1.0`) to predict `z_t` from the two
  preceding states, **fit on `control_calib` clips only**. To keep the L1
  threshold unbiased, L1 scores *on* `control_calib` are produced by 5-fold
  cross-fitting (each clip scored by a `W` fit without it); anomaly and
  `control_eval` clips are scored with the `W` fit on all of `control_calib`. L1 consumes exactly
  the same `{z_t}` sequence as the geometry probes, removing any "the baseline
  saw different information" objection.

Both baselines are reported. L0 is the primary baseline; a probe must beat
*both* to be called an Early Warner.

### 2.3 Geometry probes
- **Probe A - Trajectory curvature:** `a_t = ||z_t - 2 z_{t-1} + z_{t-2}||_2`
- **Probe B - Representational distance:** `b_t = 1 - cos(z_t, mean(z_{t-5..t-1}))`
- **Probe C - Covariance shift:** `c_t = ||Cov(z_{t-4..t}) - Sigma_base||_F`,
  where `Sigma_base = Cov(z_0..z_14)` is frozen at `t = 14` per clip. Sample
  covariance uses `ddof = 1`.

No normalization is applied to `z_t`.

## 3. Evaluation (Section 1.3 of protocol)

- **Evaluation window:** frames 15..29 inclusive (15 frames). Frames 0..14 are
  never scored; they serve as warm-up and as Probe C's baseline window.
- **Threshold:** for each monitor independently, the 99th percentile of that
  monitor's scores pooled over all `control_calib` clips and all frames in
  15..29. This equalizes the per-frame false-alarm rate at 1% across monitors,
  which is the fair-comparison mechanism.
- **Alarm:** first frame `t` in 15..29 with `score_t > threshold`.
- **Time-to-Alarm:** `dt = t_alarm - 15`, in `{0, ..., 14}`. A clip with no
  alarm in the window is **right-censored at 14**.
- **Primary test (as specified):** Wilcoxon signed-rank, paired per clip,
  `dt_probe` vs `dt_L0`, `p < 0.01`, with censored clips entered at `dt = 14`
  (conservative: it understates differences). **Tail:** H1 is directional
  (probe alarms *earlier*), so the one-sided test (`alternative="less"`) is the
  pre-registered primary; the two-sided p-value is reported alongside it in
  every table so the choice of tail cannot be made post hoc.
- **Secondary test (censoring-aware):** log-rank test on Kaplan-Meier curves of
  time-to-detection, same pairing of monitors.
- **Multiplicity:** Holm-Bonferroni across the 6 comparisons (3 probes x 2
  baselines). `p < 0.01` refers to the Holm-adjusted value.
- **Also reported:** per-condition breakdown, detection rate within the window,
  and clip-level false-alarm rate on `control_eval`.

### Secondary analyses (added 2026-10-06, before any inference run)
These are **reported alongside** the primary and do **not** decide the gate.
They were added because two weaknesses of the primary metric are visible from
its definition alone, without looking at any model output:

1. **The median is degenerate under heavy censoring.** If fewer than half the
   clips alarm inside the 15-frame window, `median(dt) = 14` for every monitor
   and the gate cannot fire regardless of the true effect. **Secondary:**
   restricted mean time to alarm, `RMTA = E[min(dt, 14)]`, computed as the area
   under the Kaplan-Meier curve, with a paired cluster bootstrap over clips
   (2000 resamples, the same resample applied to both monitors to preserve
   pairing). Holm-corrected across the same 6 comparisons.
2. **A 1% per-frame threshold compounds over 15 frames** to a clip-level
   false-alarm rate near 14%, which no episode-level monitor could use.
   **Secondary regime:** thresholds re-derived so the *clip-level* false-alarm
   rate on `control_calib` is 5%, with the entire analysis repeated unchanged.

Both secondaries are fixed here, before any model has been run, so neither can
be selected post hoc. All four cells (2 regimes x {median, RMTA}) are reported
in every results table whatever the outcome.

### Success criterion (Phase 3 gate)
A probe is an **Early Warner** iff, versus **both** L0 and L1:
`median(dt_probe) <= median(dt_baseline) - 1` **and** Holm-adjusted Wilcoxon
`p < 0.01`. If no probe qualifies, the TMLR track terminates and the result is
written up as a negative result (protocol Phase 3).

## 4. Model
`facebook/vjepa2-vitl-fpc64-256`, **frozen** (`eval()`, no gradients, no
fine-tuning). Clips are bilinearly upsampled 64 -> 256 and replicated to 3
channels. Context length `T_ctx = 16` frames, tubelet 2.

## 5. What would falsify H1
A median `dt` for every geometry probe that is greater than or equal to
`median(dt_L0)`, or a difference that fails to reach Holm-adjusted `p < 0.01`.
