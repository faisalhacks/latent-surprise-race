# Deviations from the pre-registration

Every entry must be dated and justified. Empty means the frozen protocol was
followed exactly.

| date | section | change | justification |
|---|---|---|---|
| 2026-10-06 | 3 | Specified the Wilcoxon tail as one-sided primary (two-sided also always reported) before any inference run. | H1 is directional; the tail had to be fixed before seeing data to avoid a post-hoc choice. |
| 2026-10-06 | 3 | Added two pre-registered SECONDARY analyses (RMTA with paired bootstrap; a clip-matched 5% FPR threshold regime). Primary metric and gate unchanged. | Both weaknesses follow from the definition of the primary metric, not from any observed result: a censored median is degenerate below 50% detection, and 1% per-frame compounds to ~14% per clip over 15 frames. Fixed before any model inference, so neither can be chosen post hoc. |

## Repository history rewritten to redact workspace names (2026-10-10)

On **2026-10-10**, `git filter-branch --tree-filter` was run over the whole
branch to redact Modal workspace/profile names from `STATUS.md`, where they had
been recorded while diagnosing which account could run the GPU job. The names
were replaced with generic labels (`workspace-A` ... `workspace-E`).

What this did, precisely:

| | |
|---|---|
| Commits re-hashed | **13** (every commit from the second onward, because each contained `STATUS.md` or descended from one that did) |
| Author dates | **preserved** |
| Original SHAs of those 13 | **pruned** and not recoverable from this repository |
| Pre-registration commit `033ee1ac2ca27e26bc865194b51ba146d2a3d206` | **unchanged** — it predates `STATUS.md`, so its tree was untouched and its SHA and dates survived |
| `PREREGISTRATION.md` | **unchanged** — its blob SHA at `HEAD` (`ed695cb6f45b8e31de1a22ec9a54c34271ce64a3`) is identical to its blob in the freeze commit |
| `DEVIATIONS.md` content | unchanged by the rewrite; only the SHAs of later commits that edited it moved |

The rewrite was performed **before** the instruction to preserve commit SHAs was
given, and is recorded here rather than quietly left out. **No secrets were
ever committed**: Modal API tokens were only ever passed as environment
variables and never written into the repository. The redaction concerned
account/workspace *names*, not credentials.

No history rewriting has occurred since, and none is planned. The freeze is
intact and independently verifiable:

```bash
git rev-parse 033ee1a:PREREGISTRATION.md    # -> ed695cb6f45b8e31de1a22ec9a54c34271ce64a3
git rev-parse HEAD:PREREGISTRATION.md       # -> same
```

## Commit identity changed (2026-10-10)

New commits from 2026-10-10 onward are authored as
`faisalhacks <141169331+faisalhacks@users.noreply.github.com>`. Earlier commits
retain their original authorship; rewriting it would change the
pre-registration SHA and destroy the freeze, so it was not done.

## Pivot: registered experiment parked (2026-10-10)

| date | what | why |
|---|---|---|
| 2026-10-10 | The registered Phase 2 experiment is **parked, not cancelled or amended**. No data has been collected under `PREREGISTRATION.md` v1, which remains frozen and in force, untouched. | GPU access, not science. Modal refuses GPU functions on a workspace with no payment method, independent of budget; five workspaces were checked and four are refused. The run costs ~1.0 GPU-hour / ~$1.00 and is launch-ready. See `docs/PARKED_EXPERIMENTS.md` entry P1. |
| 2026-10-10 | A second study was **drafted** (`prereg/v2_compression_DRAFT.md`) on compression and detection latency. It is explicitly **not frozen**, its equivalence margins are left as TODO, and no data may be collected under it. | Drafting a follow-up does not amend v1. Recorded here so the existence and date of the draft are on the record, and so it cannot later be presented as having been registered earlier than it was. |
| 2026-10-10 | The paper is being completed as a protocol-and-pitfalls contribution with results reported as pending. | Writing up the methodology while the registered result is outstanding is legitimate; presenting anything in place of that result would not be. No placeholder, estimated or invented numbers appear anywhere in the manuscript. |

## Registered run executed (2026-10-10)

| | |
|---|---|
| Launched | 2026-10-10 17:42:37 IST (run tag `phase2-20261010-174237`) |
| Compute | Modal, **NVIDIA A10G**, fp32/TF32 |
| Modal profile | `riyann00b` |
| Scale | 1200 clips (1000 anomalous, 100 calibration controls, 100 evaluation controls) |
| Cost | 7,554 GPU-seconds (2.10 GPU-hours), 963 s wall across 24 containers |
| Protocol | the frozen v1 protocol, unmodified |

Two corrections to how this was initially described:

**The GPU was an A10G, not a T4.** A T4 was used only for a seconds-long access
probe to confirm the account was not payment-gated. The registered run used an
A10G throughout, as recorded in the run tag.

**The analysis was not held back.** The intent had been to publish the frozen
pre-registration commit before any analysis ran. That did not happen:
`modal run modal_app.py` invokes `analyse_remote` as the final step of the same
entrypoint, so the merge, the frozen analysis and the Phase 3 gate all executed
automatically as part of the launch, and the verdict was read. There was no
separate analysis step to withhold.

What this does and does not compromise: the pre-registration genuinely precedes
the data and this is verifiable from the repository itself. Commit
`033ee1ac2ca27e26bc865194b51ba146d2a3d206` is dated 2026-10-06, four days before
the run, and the blob hash of `PREREGISTRATION.md` is `ed695cb6f45b8e31de1a22ec9a54c34271ce64a3`
at both that commit and HEAD. What is absent is an independent, third-party
timestamp: the freeze was not published before the result was seen, so a reader
relying only on the public record has the author's word for the ordering rather
than a notarised one. We record that plainly rather than implying otherwise.

No threshold, metric, test, gate criterion or probe definition was altered at
any point.

## Precision choice, measured rather than assumed (2026-10-10, exploratory)

Before the registered run, fp32 was compared against bf16 and fp16 on four
clips of real weights. Reduced precision degrades the two monitor families
asymmetrically: under bf16 the worst geometry probe shifts by a median 6.74%
while the model's own loss shifts by 0.23%, about a factor of thirty. The loss
is a mean over 256x1024 values so rounding error averages out, whereas probe A
is a second difference and probe B a cosine against a mean of near-identical
vectors.

Choosing bf16 for throughput would therefore have handicapped the probes while
leaving the baseline intact: a directional bias against the hypothesis. The
registered run used fp32, which is what the frozen protocol already specified,
so this is a confirmation of an existing choice and not a change to it. The
comparison is **exploratory** and says nothing about the hypothesis.

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

## Modal account provenance (2026-10-12)

| date | topic | entry |
|---|---|---|
| 2026-10-12 | Modal account | The registered run used profile riyann00b, owned by a friend who consented to one-time verbal use. The author's own accounts were payment-gated. Per reviewer ruling, permission is not the same as approved provenance; this is flagged as unresolved for v2. An accountable compute route must be agreed before any v2 run. |

See `PROVENANCE.md` for the full account record.
