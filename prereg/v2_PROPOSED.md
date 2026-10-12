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
- Anomaly families: **Pending reviewer ruling.** See the decision table below.
- Controls: 100 calibration + 100 eval, as in v1.
- Monitors: A (curvature), B (cosine distance), C (covariance shift).
- Baselines: loss_L0 (causal), loss_L1 (ridge).
- Threshold: 99th percentile on calibration controls = 1% per-frame
  primary; 5% clip-matched secondary.
- Pass rule: beat BOTH baselines by >= 1 frame median time-to-alarm,
  Holm-adjusted Wilcoxon p < 0.01.

### Anomaly family scope — decision table

| Aspect | 3 families (reviewer scope) | 4 families (v1 set) |
|---|---|---|
| Families | gravity_flip, friction_high, occlusion | + frame_drop |
| Tests temporal discontinuity | No | Yes |
| Clips | 950 (3x250 + 200 controls) | 1200 (4x250 + 200) |
| Multiple-testing burden | Lower (3 comparisons) | Higher (4, Holm-adjusted) |
| Compute | ~0.75 GPU-h | ~1.0 GPU-h |
| Matches reviewer scope | Yes | No — deviation |
| Argument for | Cleaner, matches original | frame_drop tests temporal failure |
| Argument against | Misses temporal anomaly | Slight power dilution |

**Scope choice: Pending reviewer ruling.**

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

## Equivalence margins — justification method

The margin must be justified from existing measurements, not
from v2 outcomes. Method:

1. Precision sensitivity probe (Step 0, pre-registered):
   - 10 control clips x 6 precisions (fp32, bf16, fp16,
     int8, W8A8, 4-bit)
   - Measure: for each precision, what is the median
     time-to-alarm on control clips (where no anomaly exists)?
   - This gives the noise floor: how much does the monitor's
     timing vary purely due to precision change on
     non-anomalous clips?

2. From the existing exploratory precision check (4 clips x
   3 precisions, already run):
   - bf16 median relative error on loss: 0.23%
   - bf16 max relative error on worst probe: 6.74%
   - loss_L0 spans ~+-2% on real weights
   These bound the expected numerical drift.

3. Margin = 2 x (observed noise floor from Step 0).
   - If Step 0 shows +-1 frame drift, margin = +-2 frames.
   - If Step 0 shows +-0.5 frames, margin = +-1 frame.
   - The multiplier of 2 is pre-registered; the noise floor
     is measured before v2 outcomes.

4. This margin is set BEFORE v2 confirmatory outcomes are
   seen. The previously proposed +-2 frames is a placeholder
   only; the actual value comes from Step 0.

**No margin is approved.** Step 0 has NOT been run. It is described
here for reviewer approval and runs only after v2 is frozen.

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
