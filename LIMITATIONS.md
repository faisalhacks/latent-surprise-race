# Known limitations and boundary conditions

Recorded as they were identified, for Phase 5 Section 8. Items marked
**[design]** are consequences of the frozen protocol that were accepted rather
than fixed, so that the pre-registration stayed intact.

## Threshold calibration

**[design] Pooling thresholds across frames 15-29 ignores non-stationarity in
`t`.** For `t < 15` the causal window is left-padded by repeating frame 0, so
the warm-up latents are computed on partially synthetic windows while `t >= 15`
latents are not. The resulting score distribution is mildly `t`-dependent, which
widens the pooled calibration distribution and makes a single pooled threshold
slightly conservative. Controls and anomalies pass through an identical
pipeline, so this cannot create a spurious difference *between monitors*; it
only costs sensitivity. A per-frame threshold (99th percentile at each `t`
separately) would be the cleaner estimator and is the obvious first robustness
check for a revision.

**[design] A 1% per-frame threshold is not an episode-level operating point.**
Over a 15-frame window it compounds to roughly 14% clip-level false alarms. The
pre-registered primary keeps it because it is what equalises per-frame false
alarms across monitors, which is the stated fairness mechanism; the
clip-matched 5% regime is reported alongside it for operational realism.

## Metric

**[design] The median of a right-censored Time-to-Alarm is degenerate below 50%
detection.** If fewer than half the clips alarm inside the window, every monitor
reports `median = 14` and the pre-registered gate cannot fire whatever the true
effect size. RMTA is reported for exactly this reason. If the primary ends up
degenerate, the honest summary is that the *pre-registered* test was
uninformative and the secondary carried the evidence - not that the effect was
absent.

**The log-rank test assumes independent groups.** Here both arms come from the
same clips, so it is corroborative only. The paired Wilcoxon and the paired
cluster bootstrap on RMTA respect the pairing.

## Latent state

**`z_t` summarises a 2-frame tubelet, not a single frame.** V-JEPA 2 has no
single-frame representation, so `z_t` depends on frames `(t-1, t)`. Time-to-Alarm
is therefore quantised to the tubelet in the sense that an anomaly injected at
frame 15 first enters a tubelet spanning frames (14, 15). Both monitor families
inherit the same quantisation, so the comparison is unaffected, but absolute
`dt` values should not be read as single-frame latencies.

**Spatial mean pooling discards where the anomaly is.** An occluder that changes
one region may be partly averaged away. This biases *against* the probes on the
occlusion condition and is worth revisiting with a max-pooled or
patch-local variant.

**Probes A and C are not scale-invariant.** If the encoder's output norm drifts
systematically with `t`, their scores drift with it. Probe B is scale-invariant
by construction. No normalisation is applied, per the frozen protocol.

## Numerics

**Probe A is a second difference and therefore the worst case for reduced
precision.** `\|\|z_t - 2 z_{t-1} + z_{t-2}\|\|` subtracts nearby vectors, so
relative error in `z` is amplified. This is why precision is measured
(`modal run modal_app.py --precision-check-only`) instead of assumed, and why
fp32 with TF32 matmuls is the default despite costing throughput.

## Generalisation

**One architecture, one synthetic domain.** Phase 2 establishes the phenomenon
or its absence for V-JEPA 2 on bouncing discs only. Phase 4.1 (a second
architecture) and 4.2 (MuJoCo) exist precisely because a single
joint-embedding-predictive model on 64x64 synthetic video cannot support a
general claim.

**The friction and gravity conditions differ in observability.** A gravity
inversion changes the acceleration immediately; the per-frame drag in the
friction condition changes velocity by ~4.4% per frame. Effect sizes are
therefore not comparable across conditions, which is why per-condition results
are reported separately rather than only pooled.
