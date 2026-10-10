# DRAFT v2 — Compression and detection latency in video world models

> ## ⚠ DRAFT. NOT FROZEN. NOT IN FORCE.
>
> **Drafted:** 2026-10-10
> **Status:** incomplete by design. Equivalence margins are left as TODO for the
> repository owner to set. **No data may be collected under this document until
> it is completed, reviewed, and frozen** in a separate commit that says so.
>
> This does not amend, replace, or relax `PREREGISTRATION.md` (v1), which
> remains frozen and governs the registered Phase 2 experiment. v1 is untouched.

---

## 0. Why a second study

v1 asks whether internal geometry crosses an alarm threshold sooner than the
model's own prediction error. v2 asks a question that only becomes interesting
once a monitoring harness exists: **does compressing a world model change
*which* anomalies it still catches, and *how fast* — rather than degrading it
uniformly?**

The motivation is empirical. Fu et al. (2026) report that post-training
quantization of a world model produces *asymmetric* sensitivity across
components and failure modes that further optimization does not repair. A
mean-performance number hides that. A per-anomaly-family detection-latency
measurement should not.

The deployment relevance is direct: compressed world models are what actually
run on robots, and a compression that preserves average accuracy while
selectively destroying sensitivity to one class of physical anomaly is a safety
property worth knowing about.

## 1. Factors

### 1.1 Precision (6 levels)
| id | definition |
|---|---|
| `fp32` | reference condition, TF32 matmuls permitted |
| `bf16` | weights and activations bfloat16 |
| `fp16` | weights and activations float16 |
| `int8_w` | int8 weight-only, activations fp16 |
| `w8a8` | int8 weights and int8 activations |
| `int4_w` | 4-bit weight-only (group-wise), activations fp16 |

Group-wise weight quantization is used where applicable; Fu et al. report it
stabilizes low-bit rollouts, so it is the reasonable default rather than a
tuned choice.

### 1.2 Placement (3 levels)
| id | what is compressed |
|---|---|
| `enc` | encoder only; predictor left fp32 |
| `pred` | predictor only; encoder left fp32 |
| `both` | both |

Placement is the factor that tests the asymmetry claim directly. Under `enc`
the latents the probes consume are degraded but the prediction head is intact;
under `pred` the reverse. If geometry probes and the prediction loss degrade at
different rates under different placements, that is the finding.

**Cells:** 1 (`fp32` reference) + 5 x 3 = **16**.

## 2. Primary contrasts

**One primary contrast per anomaly family**, to keep the confirmatory family
small enough to correct honestly:

| anomaly family | primary contrast | quantity |
|---|---|---|
| gravity inverted | `w8a8/both` vs `fp32` | RMTA difference (frames) |
| friction | `w8a8/both` vs `fp32` | RMTA difference |
| occlusion | `w8a8/both` vs `fp32` | RMTA difference |
| frame drop | `w8a8/both` vs `fp32` | RMTA difference |

Four primary contrasts, Holm-corrected across them. Every other cell is
**exploratory** and reported as such, with no inferential claim.

Rationale for `w8a8` as the primary: it is the most aggressive setting in
common deployment use, so it is where a selective failure is most likely and
most consequential. All other cells still run and are reported.

## 3. Equivalence testing

The interesting claim is often that compression **does not** hurt. That is an
equivalence claim and cannot be supported by failing to reject a difference, so
the primary analysis is **TOST** (two one-sided tests) on the RMTA difference,
not a null-hypothesis test.

> ### TODO — margins to be set by the repository owner before freezing
>
> - `DELTA_RMTA`: the largest detection-latency change, in frames, that counts
>   as practically equivalent. **TODO.** Must be justified operationally (what
>   latency increase would matter to a controller at this frame rate), not
>   chosen to make a result come out.
> - `DELTA_DETECT`: the largest acceptable drop in detection rate at the matched
>   operating point. **TODO.**
> - `ALPHA`: significance level for TOST. Suggest 0.05 per contrast before Holm,
>   but **TODO** to confirm.
> - Whether equivalence must hold against **both** baselines or only `L0`.
>   **TODO.**
>
> These are left deliberately blank. Setting a margin after seeing the effect is
> the single easiest way to manufacture an equivalence result, and the draft is
> not freezable until they are fixed.

## 4. Sampling and seeds

- **Fresh seed for confirmatory clips.** v1 used `SEED = 20261006`. v2
  confirmatory clips use a **new seed, to be fixed at freeze time**, so that no
  clip used to develop or debug the harness appears in the confirmatory set.
  The v1 clips may be reused for exploratory analysis only, labelled as such.
- Clip counts per cell: **TODO** at freeze. 1200 gives parity with v1 at ~16
  GPU-hours total; 400 cuts cost to roughly a third with correspondingly wider
  intervals. The choice should follow from `DELTA_RMTA` via a power
  calculation, not the reverse.
- Thresholds are calibrated **per cell** on that cell's own control clips.
  Carrying an fp32 threshold into a quantized cell would confound a shift in
  score scale with a change in sensitivity.

## 5. Second-GPU arm

Every primary contrast is replicated on a **second GPU architecture** (e.g.
A10G and L4, or L40S and A100). Quantization kernels, tensor-core paths and
accumulation order differ across hardware, so a compression effect that does
not reproduce across two GPUs is a kernel artefact, not a property of the
model. Reported as a replication, not pooled.

## 6. Exploratory analyses (explicitly not confirmatory)

Clearly labelled, no inferential claims, reported whatever they show.

**6.1 Blind spots.** For each cell, the set of clips detected under `fp32` but
missed under compression, and vice versa. The question is whether misses
concentrate in one anomaly family or one region of the state space — a
*selective* blind spot — rather than thinning out uniformly. Quantified as the
per-family miss-rate difference and its dispersion across families.

**6.2 Motion energy.** Per clip, mean frame-to-frame pixel difference over the
evaluation window, as a cheap proxy for how much is visibly happening. Tests
whether compression costs are concentrated in low-motion clips — where the
signal to be detected is smallest and a quantized representation has least to
work with. Reported as detection latency against motion-energy decile.

Both are motivated by the observation that aggregate accuracy can be preserved
while the *distribution* of failures shifts, which is precisely the failure mode
a mean cannot show.

## 7. What would falsify the headline claim

If every compression cell is statistically equivalent to `fp32` within the
margins set in §3, on every anomaly family and on both GPUs, then compression
does not selectively change detection latency on this benchmark, and the study
reports that. The design is intended to make that outcome as reportable as the
alternative.

## 8. Inherited from v1 without change

Benchmark generator, the four anomaly definitions, the five monitors and their
closed forms, causal windowing and the two-pass L0 construction, the control
split and leakage discipline, the survival-analysis machinery, Holm correction,
the lift and vacuous-verdict guards. v2 changes the model's numerics, not the
measurement apparatus.

---

## Checklist before this document can be frozen

- [ ] `DELTA_RMTA`, `DELTA_DETECT`, `ALPHA` set and justified
- [ ] Clip count per cell set, with a power calculation
- [ ] Confirmatory seed fixed
- [ ] Second-GPU pair chosen
- [ ] Quantization implementation chosen and its kernels pinned
- [ ] `--precision-matrix` driver implemented in `modal_app.py`
- [ ] Budget approved (~$17 at 1200 clips/cell, ~$35 with the second-GPU arm)
- [ ] Reviewed and frozen in its own commit
