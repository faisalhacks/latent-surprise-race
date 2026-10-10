# Exploratory CPU pilot at 128 px

## This is NOT the registered experiment

Nothing in this folder bears on the hypothesis. It exists to size the registered
run and to sanity-check the pipeline against real pretrained weights. It must
not be pooled with, substituted for, or reported alongside registered results.

| | registered experiment | this pilot |
|---|---|---|
| resolution | **256 px** (frozen) | **128 px** |
| patch grid | 16 x 16 | 8 x 8 |
| clips | 1200 | 280 (reduced) |
| hardware | GPU | CPU |
| status | **not run** | incomplete |
| counts as evidence | yes, once run | **no** |

## What is and is not changed

Changed: spatial resolution and scale, nothing else.

Unchanged: context length (16 frames), tubelet size (2), all three probe
definitions, the two-pass `L0` construction, the threshold rules, the metrics,
the statistics and the gate. 96 px was rejected because the disc (10 px
diameter at source) would then be smaller than a single 16 px patch.

## Status

**No outcome numbers have been computed, printed, plotted or analysed.**

Thresholds are calibrated on control clips. Until both control splits are
complete, no threshold exists, so no alarm time is defined and no detection
latency can be computed for any monitor. The run order was deliberately
arranged to extract **controls first**, so that an interrupted run becomes a
smaller but coherent experiment rather than an incoherent one — but that
threshold has not yet been crossed.

Live counts are regenerated into the manuscript at build time by
`scripts/make_paper.py` (`pilot_macros`), read from the chunk files on disk, so
the paper always reports what actually exists rather than a number someone
typed once.

To see the current state:

```bash
python scripts/make_paper.py          # prints "pilot: N / 280 clips (state)"
```

## If you do analyse it later

```bash
python scripts/run_pilot_cpu.py --analyse-only
```

This merges whatever chunks are on disk and runs the standard analysis. Any
output **must** be labelled exploratory and reported at its true clip counts,
including which anomaly families are present and in what proportion. The
analysis writes its own note field recording that it is a pilot.

## Why 128 px was necessary

Measured on the development machine: **9.66 s per encoder forward** at the
frozen 256 px, i.e. ~478 s per clip and **~159 hours** for the full registered
design. At 128 px it is ~55 s per clip. The registered run on a GPU is ~1
GPU-hour; the CPU pilot exists only because that GPU-hour was unavailable.

## Contents

Chunk files, if present, live in `../results/pilot_cpu_128/chunks/` and are not
committed — they are regenerable from the frozen seed and would otherwise add
hundreds of megabytes of latents to the repository.
