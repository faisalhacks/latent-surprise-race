# Deviations from the pre-registration

Every entry must be dated and justified. Empty means the frozen protocol was
followed exactly.

| date | section | change | justification |
|---|---|---|---|
| 2026-10-06 | 3 | Specified the Wilcoxon tail as one-sided primary (two-sided also always reported) before any inference run. | H1 is directional; the tail had to be fixed before seeing data to avoid a post-hoc choice. |
| 2026-10-06 | 3 | Added two pre-registered SECONDARY analyses (RMTA with paired bootstrap; a clip-matched 5% FPR threshold regime). Primary metric and gate unchanged. | Both weaknesses follow from the definition of the primary metric, not from any observed result: a censored median is degenerate below 50% detection, and 1% per-frame compounds to ~14% per clip over 15 frames. Fixed before any model inference, so neither can be chosen post hoc. |

## Diagnostics added after a 2-clip correctness check (not hypothesis tests)

| date | what | provenance |
|---|---|---|
| 2026-10-06 | Added a **lift** column (detection rate minus held-out clip false-alarm rate) per monitor, a flag for baselines firing at chance, and a `verdict_is_vacuous` marker when a probe clears the gate only against such a baseline. | Motivated by running the real pretrained weights over 2 clips, which showed `loss_L0` spanning only +/-2%. What was observed was score *magnitude*, not any Time-to-Alarm, gate outcome, or effect direction. No threshold, metric, test or gate criterion was altered; this only makes an existing failure mode visible in the report. |

Recorded here rather than buried, because the check that motivated it ran after
the freeze. A reviewer should be able to see exactly what was known when.

## Resolutions made at freeze time (not deviations)
Three internal inconsistencies in the source protocol were resolved *before*
freezing and are documented in PREREGISTRATION.md Sec. 0: the
constant-velocity/gravity-inversion contradiction, the unobservability of a
collision-only friction change, and the under-determination of a per-frame
latent state under V-JEPA 2's 2-frame tubelet.
