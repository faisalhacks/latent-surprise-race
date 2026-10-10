# Protocol

A reading copy of the frozen test. **The authority is
[`PREREGISTRATION.md`](PREREGISTRATION.md)**, which is byte-identical to its
state in the freeze commit `033ee1ac2ca27e26bc865194b51ba146d2a3d206`
(2026-10-06). Where this file quotes a definition, it quotes it verbatim; no
test, threshold, metric or stopping rule has been reworded.

Verify the frozen document is unmodified:

```bash
diff --strip-trailing-cr <(git show 033ee1a:PREREGISTRATION.md) PREREGISTRATION.md
```

---

## 1. Hypothesis

Quoted from the frozen document:

> **Hypothesis (H1).** Sudden changes in physics, occlusion, or effective frame
> rate cause measurable shifts in the geometry of a frozen video world model's
> internal representations *before* its own prediction error spikes.

> **Primary claim under test.** For at least one geometry probe `P`, the median
> Time-to-Alarm `median(dt_P)` is lower than `median(dt_loss)` by >= 1 frame,
> with Wilcoxon signed-rank `p < 0.01` after Holm correction.

The manuscript says **detection latency** where the frozen document says
"Time-to-Alarm" / "Early Warner", because the benchmark injects anomalies
abruptly and therefore measures reaction time rather than anticipation. The
*criterion* is unchanged; only the wording in the paper differs. This is logged
in `DEVIATIONS.md`.

## 2. Task

A disc of radius 5 bounces inside a 64x64 frame for 30 frames under constant
downward gravity `g0 = 0.25 px/frame^2` with perfectly elastic collisions, so
the baseline dynamics conserve energy and are statistically stationary.
Initial position, speed and direction are randomised. Generation is
deterministic given `SEED = 20261006`.

At **frame 15** one anomaly is injected. The four families, quoted:

| id | definition from `t >= 15` |
|---|---|
| `gravity_flip` | `g <- -g0` (inverted) |
| `friction_high` | per-frame `v <- 0.8^(1/5) v`; restitution `1.0 -> 0.8` |
| `occlusion` | static black square, area = 30% of frame (side 35 px), drawn over all later frames, positioned to intersect the disc's baseline path |
| `frame_drop` | for underlying frames 15..22, every other frame is not delivered; the previously delivered frame is held in its place (4 held frames), normal delivery resumes at frame 23 |

**All four families are confirmatory.** The frozen design allocates 250 clips
to each (1000 anomalous total); results are reported pooled *and* per family,
and the frozen document states that per-condition results are reported
separately because effect magnitudes are not comparable across families.

**Checkpoint:** `facebook/vjepa2-vitl-fpc64-256`, **frozen** (`eval()`, no
gradients, no fine-tuning), HF revision
`b3c1679b7c34d3255ef3547f27c7b226aefab26f`. Clips are bilinearly upsampled
64 -> 256 and replicated to 3 channels. Context length `T_ctx = 16` frames,
tubelet 2.

## 3. Controls

- **Control clips:** 200, baseline physics throughout, no anomaly.
- **Control split (leakage control), quoted:**

  > controls are partitioned by seed into `control_calib` (100 clips, used
  > *only* to set thresholds) and `control_eval` (100 clips, used *only* to
  > report false-alarm rates). Thresholds are never set on data used to measure
  > false alarms, and never on anomalous data.

- **Paired design:** an anomalous clip and its control share a seed, so their
  frames are identical up to frame 15.
- **Two prediction-error baselines**, both reported. `L0` is the model's own
  pretraining objective evaluated causally via a second encoder pass; `L1` is a
  ridge next-step predictor fitted on `control_calib` only, cross-fitted 5-fold
  for its own scores. Quoted: *"L0 is the primary baseline; a probe must beat
  both to be called an Early Warner."*

## 4. Metrics

**The three geometry probes, quoted verbatim:**

> - **Probe A - Trajectory curvature:** `a_t = ||z_t - 2 z_{t-1} + z_{t-2}||_2`
> - **Probe B - Representational distance:** `b_t = 1 - cos(z_t, mean(z_{t-5..t-1}))`
> - **Probe C - Covariance shift:** `c_t = ||Cov(z_{t-4..t}) - Sigma_base||_F`,
>   where `Sigma_base = Cov(z_0..z_14)` is frozen at `t = 14` per clip. Sample
>   covariance uses `ddof = 1`.
>
> No normalization is applied to `z_t`.

**Evaluation window, false-alarm rule, alarm and latency, quoted verbatim:**

> - **Evaluation window:** frames 15..29 inclusive (15 frames). Frames 0..14 are
>   never scored; they serve as warm-up and as Probe C's baseline window.
> - **Threshold:** for each monitor independently, the 99th percentile of that
>   monitor's scores pooled over all `control_calib` clips and all frames in
>   15..29. This equalizes the per-frame false-alarm rate at 1% across monitors,
>   which is the fair-comparison mechanism.
> - **Alarm:** first frame `t` in 15..29 with `score_t > threshold`.
> - **Time-to-Alarm:** `dt = t_alarm - 15`, in `{0, ..., 14}`. A clip with no
>   alarm in the window is **right-censored at 14**.

**Tests, quoted verbatim:**

> - **Primary test (as specified):** Wilcoxon signed-rank, paired per clip,
>   `dt_probe` vs `dt_L0`, `p < 0.01`, with censored clips entered at `dt = 14`
>   (conservative: it understates differences). **Tail:** H1 is directional
>   (probe alarms *earlier*), so the one-sided test (`alternative="less"`) is the
>   pre-registered primary; the two-sided p-value is reported alongside it in
>   every table so the choice of tail cannot be made post hoc.
> - **Secondary test (censoring-aware):** log-rank test on Kaplan-Meier curves of
>   time-to-detection, same pairing of monitors.
> - **Multiplicity:** Holm-Bonferroni across the 6 comparisons (3 probes x 2
>   baselines). `p < 0.01` refers to the Holm-adjusted value.
> - **Also reported:** per-condition breakdown, detection rate within the window,
>   and clip-level false-alarm rate on `control_eval`.

**Pre-registered secondaries** (added 2026-10-06, before any inference run —
see `DEVIATIONS.md`): restricted mean detection latency (RMDL) with a paired
cluster bootstrap, and a second threshold regime matching the **clip-level**
false-alarm rate at 5%. Both are reported unconditionally alongside the primary.

## 5. Stopping rule / decision gate

Quoted from the frozen document:

> A probe is an **Early Warner** iff, versus **both** L0 and L1:
> `median(dt_probe) <= median(dt_baseline) - 1` **and** Holm-adjusted Wilcoxon
> `p < 0.01`. If no probe qualifies, the TMLR track terminates and the result is
> written up as a negative result (protocol Phase 3).

The gate is evaluated in code (`src/lgw/analysis.py`), not by reading tables.

There is **no sequential stopping or interim peeking**: the sample size is fixed
in advance at 1000 anomalous and 200 control clips, the analysis runs once on
the complete set, and no outcome has been inspected.

## 6. Falsification

Quoted:

> A median `dt` for every geometry probe that is greater than or equal to
> `median(dt_L0)`, or a difference that fails to reach Holm-adjusted `p < 0.01`.

## 7. Status

**The registered experiment has not been run** (GPU credit, not code). No
detection latency, gate verdict, survival curve or p-value exists. See
`CLOSEOUT.md` and `CLAIMS.md`.
