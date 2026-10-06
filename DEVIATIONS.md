# Deviations from the pre-registration

Every entry must be dated and justified. Empty means the frozen protocol was
followed exactly.

| date | section | change | justification |
|---|---|---|---|
| 2026-10-06 | 3 | Specified the Wilcoxon tail as one-sided primary (two-sided also always reported) before any inference run. | H1 is directional; the tail had to be fixed before seeing data to avoid a post-hoc choice. |
| 2026-10-06 | 3 | Added two pre-registered SECONDARY analyses (RMTA with paired bootstrap; a clip-matched 5% FPR threshold regime). Primary metric and gate unchanged. | Both weaknesses follow from the definition of the primary metric, not from any observed result: a censored median is degenerate below 50% detection, and 1% per-frame compounds to ~14% per clip over 15 frames. Fixed before any model inference, so neither can be chosen post hoc. |

## Resolutions made at freeze time (not deviations)
Three internal inconsistencies in the source protocol were resolved *before*
freezing and are documented in PREREGISTRATION.md Sec. 0: the
constant-velocity/gravity-inversion contradiction, the unobservability of a
collision-only friction change, and the under-determination of a per-frame
latent state under V-JEPA 2's 2-frame tubelet.
