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

## The occlusion condition tests object permanence, not partial occlusion

The frozen specification asks for a static black square of 30% frame area
positioned to intersect the disc's path, described as "partially blocking" it.
A 35x35 square anchored on the trajectory of a 10-pixel disc does no such thing.
Measured across all 250 clips of the condition (3750 post-injection frames):

| | |
|---|---|
| frames with the disc **completely** hidden | 77.5% |
| frames with the disc partially hidden | 17.2% |
| frames with the disc fully visible | 5.3% |
| clips where the disc is never visible again | 28.0% |

So this condition asks whether the representation registers that a tracked
object has **vanished**, which is a question about object permanence, not about
robustness to partial visual corruption. It follows directly from the
registered specification and is not a defect, but the paper reports it this way
rather than repeating the specification's wording, because the two describe
different experiments. A genuine partial-occlusion condition would need the
occluder sized or placed to guarantee partial overlap, and is a sensible
addition for a future revision rather than a change to a frozen condition.

## Two risks measured on real weights (n = 2 clips, 2026-10-06)

`scripts/smoke_real_cpu.py` ran the real pretrained ViT-L over one
`gravity_flip` clip and one control. It is a correctness check, far too small
for inference, but it exposes two threats to the experiment's power that are
visible from score *magnitudes* alone - no Time-to-Alarm, no gate outcome, and
nothing about the direction of H1.

**1. `loss_L0` has almost no dynamic range.** Observed 7.57 to 7.94 across both
clips and all scored frames: about +/-2% around 7.75. The loss is an MSE averaged
over 256 tokens x 1024 dimensions, and these anomalies are spatially local, so
the average dilutes the signal heavily. If this holds at n = 1200, L0's 99th
percentile sits inside its own noise and it will alarm at roughly its
false-alarm rate, i.e. at chance.

*Why this matters more than it looks:* beating a baseline that detects nothing
is not evidence that geometry warns earlier. It only shows the baseline is
insensitive at this operating point. The analysis therefore reports a **lift**
column (detection rate minus held-out clip false-alarm rate) for every monitor,
flags any baseline whose lift is below 0.05, and marks a PROCEED verdict
**vacuous** if the probe that cleared the gate only beat such a baseline. This
is a diagnostic, not a change to any hypothesis test.

*Possible remedies, none applied:* a max over tokens rather than a mean; a
per-token loss restricted to the tokens whose content actually changed; or
reporting L1 as the primary baseline, since it is information-matched and showed
a wider range.

**2. Probes A and C have between-clip offsets comparable to their within-clip
variation.** Probe C read ~191 on one clip and ~206 on the other, while moving
only ~3 units across the scored frames *within* each clip. Thresholds are
pooled across control clips, so a per-clip offset of that size means detection
is driven largely by which clips happen to sit high, not by the anomaly - a
change detector behaving like a per-clip constant. Probe B, which is
scale-invariant by construction, showed no comparable offset.

The standard fix is to standardise each probe per clip against its own
pre-injection frames. That cannot be applied uniformly here: probe C is
undefined before frame 15, so it has no within-clip pre-injection sample, and
`Sigma_base` would have to serve as the reference instead. This is a genuine
design question rather than a bug, it postdates the freeze, and it is left as a
decision rather than silently patched. **If the real run shows probes A and C
near chance while B does not, this is the first thing to test.**

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
