# Deviations from the pre-registration

Every entry must be dated and justified. Empty means the frozen protocol was
followed exactly.

| date | section | change | justification |
|---|---|---|---|
| 2026-10-06 | 3 | Specified the Wilcoxon tail as one-sided primary (two-sided also always reported) before any inference run. | H1 is directional; the tail had to be fixed before seeing data to avoid a post-hoc choice. |
| 2026-10-06 | 3 | Added two pre-registered SECONDARY analyses (RMTA with paired bootstrap; a clip-matched 5% FPR threshold regime). Primary metric and gate unchanged. | Both weaknesses follow from the definition of the primary metric, not from any observed result: a censored median is degenerate below 50% detection, and 1% per-frame compounds to ~14% per clip over 15 frames. Fixed before any model inference, so neither can be chosen post hoc. |

## Pivot: registered experiment parked (2026-10-10)

| date | what | why |
|---|---|---|
| 2026-10-10 | The registered Phase 2 experiment is **parked, not cancelled or amended**. No data has been collected under `PREREGISTRATION.md` v1, which remains frozen and in force, untouched. | GPU access, not science. Modal refuses GPU functions on a workspace with no payment method, independent of budget; five workspaces were checked and four are refused. The run costs ~1.0 GPU-hour / ~$1.00 and is launch-ready. See `docs/PARKED_EXPERIMENTS.md` entry P1. |
| 2026-10-10 | A second study was **drafted** (`prereg/v2_compression_DRAFT.md`) on compression and detection latency. It is explicitly **not frozen**, its equivalence margins are left as TODO, and no data may be collected under it. | Drafting a follow-up does not amend v1. Recorded here so the existence and date of the draft are on the record, and so it cannot later be presented as having been registered earlier than it was. |
| 2026-10-10 | The paper is being completed as a protocol-and-pitfalls contribution with results reported as pending. | Writing up the methodology while the registered result is outstanding is legitimate; presenting anything in place of that result would not be. No placeholder, estimated or invented numbers appear anywhere in the manuscript. |

## Terminology note (2026-10-10)

The frozen v1 document names its gate criterion **"Early Warner"**. The
manuscript avoids that phrase in prose and says **detection latency** instead,
because the benchmark injects anomalies abruptly at a known frame: the
experiment measures how quickly a monitor reacts to an event that has already
happened, not whether it anticipates one. The frozen criterion itself is
unchanged — only the word used to describe it in the paper.

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
