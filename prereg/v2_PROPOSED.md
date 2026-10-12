# v2 — Proposed Protocol (NOT FROZEN)

**Status:** PROPOSED. Not frozen. No data collection permitted until
reviewed by Ryan Gomez, equivalence margins set, and a v2 freeze
commit is cut.

## Structural change from v1
Inference and analysis are separate entrypoints. This is the single
change requested by the reviewer to prevent the gate-crossing that
occurred in v1.

### Step 1: Inference
- Script: scripts/run_inference.py
- Action: runs the frozen model on all clips, produces raw latents
  and loss arrays, writes to disk.
- MUST NOT call any analysis function.
- Outputs are checksummed (SHA-256) and committed before Step 2.

### Step 2: Analysis
- Script: scripts/run_analysis.py
- Action: reads saved, checksummed outputs from disk, runs the
  frozen analysis pipeline, computes the Phase 3 gate verdict.
- May not run until Step 1 is complete and outputs are verified.
- The gate verdict is written to results.json and REPORT.md.

No entrypoint may chain both steps. No automatic analysis on launch.

## Scope
- Fresh seed (not 20261006).
- Anomaly families: [four as in v1, or three per Ryan's original
  scope — pending Ryan's ruling]
- Controls: 100 calibration + 100 eval, as in v1.
- Monitors: A (curvature), B (cosine distance), C (covariance shift).
- Baselines: loss_L0 (causal), loss_L1 (ridge).
- Threshold: 99th percentile on calibration controls = 1% per-frame
  primary; 5% clip-matched secondary.
- Pass rule: beat BOTH baselines by >= 1 frame median time-to-alarm,
  Holm-adjusted Wilcoxon p < 0.01.

## Compression arm (Study 2)
Conditions:
- bf16 (encoder / predictor / both)
- fp16 (encoder / predictor / both)
- int8 weight-only (encoder / predictor / both)
- W8A8 (encoder / predictor / both)
- 4-bit (encoder / predictor / both)
- fp32 control (re-run for paired comparison)

Cross-check: second GPU (L4 or L40S) for hardware-independence
verification on a subset.

## Equivalence margins
**TODO — Ryan Gomez to set.**
Proposed starting point: ±2 frames median time-to-alarm between
precision levels. A precision level "passes" if its median
time-to-alarm is within the margin of the fp32 control. Final value
must be set before the v2 freeze commit.

## What v2 does NOT change from v1
- Model: facebook/vjepa2-vitl-fpc64-256, frozen.
- Data: synthetic bouncing disc, 64x64 upsampled to 256, 30 frames,
  anomaly at frame 15.
- Leakage fix: two encoder passes (carried from v1).
- Lift check / vacuous flag (carried from v1, recorded in
  DEVIATIONS.md).

## What v2 labels differently from v1
- v1 results are reported as EXPLORATORY in the paper, not
  confirmatory.
- v2 is the confirmatory study.
- The compression arm is the primary novel contribution.
