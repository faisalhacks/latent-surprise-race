# Latent Geometry vs. Prediction Error in Video World Models

Does the *geometry* of a frozen video world model's internal representations
signal a sudden physical or visual anomaly **earlier** than the model's own
prediction error does?

This repository is the leakage-free evaluation harness for that question:
a synthetic anomaly benchmark, three online geometry probes, two prediction-loss
baselines, and a survival analysis of time-to-detection, all pinned to a frozen
pre-registration.

## Status

| phase | state |
|---|---|
| 1. Pre-registration frozen | done - [`PREREGISTRATION.md`](PREREGISTRATION.md) |
| 2. Harness + V-JEPA 2 inference | code complete, validated on CPU and on GPU |
| 3. Decision gate | automated in [`src/lgw/analysis.py`](src/lgw/analysis.py) |
| 4. Expansion (cross-architecture, MuJoCo, mechanistic) | gated on Phase 3; needs explicit approval before spending compute |

## The two things that make this harness defensible

**Causality is structural, not conventional.** Every monitor is a streaming
object fed one latent at a time, so it cannot physically reference a future
frame. `tests/test_causality.py` additionally inverts every frame after `t` and
asserts that `z_t`, the prediction loss and all three probes are bit-identical.
The L0 baseline takes its context from a *separate* encoder pass whose final
tubelet is overwritten, so no context token attends to the frames being
predicted - also asserted, against the real V-JEPA 2 architecture.

**Thresholds never touch the data they are evaluated on.** The 200 control clips
are split into 100 calibration (sets thresholds) and 100 evaluation (measures
false alarms). The ridge baseline L1 is fit on calibration controls only, and
scored on those same clips by 5-fold cross-fitting so its own threshold is not
optimistic.

## Monitors

| id | definition |
|---|---|
| `loss_L0` | V-JEPA 2 predicting its own final-tubelet tokens from frames `<= t-2` (the pretraining objective, evaluated causally) |
| `loss_L1` | ridge next-step predictor on the same latent sequence the probes see - information-matched |
| `probe_A` | trajectory curvature, `\|\|z_t - 2 z_{t-1} + z_{t-2}\|\|` |
| `probe_B` | representational distance, `1 - cos(z_t, mean(z_{t-5..t-1}))` |
| `probe_C` | covariance shift, `\|\|Cov(z_{t-4..t}) - Sigma_base\|\|_F` |

A probe counts as an **Early Warner** only if it beats **both** baselines by
`>= 1` frame of median Time-to-Alarm at Holm-adjusted `p < 0.01`.

## Layout

```
PREREGISTRATION.md      frozen protocol - the contract
DEVIATIONS.md           dated log of every change to it
modal_app.py            Phase 2: frozen V-JEPA 2 inference on Modal
src/lgw/
  config.py             frozen constants (pinned by a test)
  datagen.py            bouncing-disc clips + 4 anomaly conditions
  extract.py            causal windowing + the online extraction loop
  probes.py             streaming probes A/B/C, ridge baseline L1
  vjepa2_adapter.py     frozen V-JEPA 2 wrapped for online probing
  surrogate.py          cheap CPU stand-in, for harness validation only
  stats.py              thresholds, Time-to-Alarm, Kaplan-Meier, log-rank, RMTA
  analysis.py           Phase 1.3 analysis + Phase 3 gate + report + plots
scripts/
  run_local_validation.py   full pipeline on CPU, no GPU budget
tests/                  causality, leakage, probe correctness, frozen config
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # 22 tests, no GPU needed
python scripts/run_local_validation.py 40      # whole pipeline on CPU

modal run modal_app.py --smoke                 # cents; validates the GPU path
modal run modal_app.py --precision-check-only  # fp32 vs bf16 vs fp16 numerics
modal run modal_app.py                         # full Phase 2
modal volume get lgw-results <tag>/ ./results  # fetch latents + report
```

On Windows set `PYTHONIOENCODING=utf-8` before `modal run`; the Modal CLI
prints glyphs that the default console codepage cannot encode.

## Reporting

Every run writes `results.json`, a `REPORT.md` with the full result set, and
Kaplan-Meier survival curves, under **two threshold regimes** (per-frame-matched
1% FPR, the pre-registered primary; and clip-matched 5% FPR) and **two metrics**
(median Time-to-Alarm, and the censoring-robust RMTA). All four cells are
reported whatever the outcome, so a degenerate median cannot hide an effect and
a favourable cell cannot be cherry-picked.
