# Parked experiments

**Status: parked 2026-10-10. Nothing here runs until the repository owner says
"credits are available, go".**

Everything below is blocked on GPU credit, not on code. Each entry is
launch-ready: the harness, the adapters and the analysis are written, tested and
committed. Costs are derived from a measured figure — ~86 TFLOPs per clip
(~1.75 TFLOPs per encoder forward at 326M parameters x 2048 tokens, plus
~0.5 TFLOPs per predictor forward, 45 + 15 forwards per clip) — and an assumed
~30 TFLOPS effective on an A10G at fp32/TF32, with A10G at roughly $1.10/hour.
Treat them as estimates with maybe 50% error, not quotes.

---

## P1. The registered Phase 2 experiment (highest priority)

**Question.** Does the geometry of a frozen V-JEPA 2's internal state cross an
alarm threshold at a lower detection latency than the model's own prediction
error, on the pre-registered benchmark?

**Conditions.** The frozen protocol exactly: 1200 clips at 256 px
(1000 anomalous across 4 anomaly families, 100 calibration controls,
100 evaluation controls), `T_ctx = 16`, fp32/TF32, five monitors, both threshold
regimes, both metric families.

**Why it matters.** This is the only run that can answer the registered
question. Everything else in this file is secondary to it. Without it the paper
has a protocol and no verdict.

**Estimated cost.** ~1.0 GPU-hour, **~$1.00**. Wall clock ~5–10 minutes with
24-way fan-out.

**Launch.**
```bash
cd latent-geometry-warning
export PYTHONIOENCODING=utf-8          # Windows: cmd/PowerShell `set`
modal run modal_app.py --chunk 50
modal volume get lgw-results <tag>/ ./results
python scripts/make_paper.py && cd paper && tectonic main.tex
```

**Prerequisite.** A Modal workspace with a payment method on file. GPU functions
are refused outright without one, independent of budget or credits.

---

## P2. Precision check (run before P1 if bf16 is wanted)

**Question.** Does reduced precision preserve the probe values, in particular
probe A? It is a second difference of nearby vectors, `||z_t - 2z_{t-1} +
z_{t-2}||`, which is the textbook setting for catastrophic cancellation, so
reduced precision is not a free speed-up here.

**Conditions.** 4 clips x {fp32, bf16, fp16}, reporting median and maximum
relative error plus Spearman and Pearson correlation against fp32, per monitor.

**Estimated cost.** ~3 GPU-minutes, **~$0.05**.

**Launch.**
```bash
modal run modal_app.py --precision-check-only
```

**Decision rule.** Use fp32 for P1 unless bf16 keeps probe A's median relative
error below ~1% with Spearman > 0.99. Default is fp32.

---

## P3. Compression and quantization study (v2)

**Question.** Does post-training compression change *which* anomalies a world
model detects and *how fast*, rather than just degrading it uniformly? And is
the encoder or the predictor the sensitive component?

**Design.** See `prereg/v2_compression_DRAFT.md` — **draft, not frozen**.
Six precisions x three placements, one primary contrast per anomaly family, a
fresh seed for the confirmatory clips, and a second-GPU arm.

**Motivation.** Fu et al. (2026) find asymmetric sensitivity between components
under world-model quantization and failure modes that further optimization does
not fix. Our harness can ask whether that asymmetry shows up as a *timing*
difference per anomaly family, which a mean-performance metric would miss.

**Estimated cost.** 16 cells (1 fp32 reference + 5 precisions x 3 placements) at
1200 clips each is ~16 GPU-hours, **~$17**. With the second-GPU arm, ~32
GPU-hours, **~$35**. Reducing confirmatory clips to 400 per cell cuts this to
roughly **$6 / $12**. Costs here are the reason this is parked and not merely
queued.

**Launch (after margins are set and the draft is frozen).**
```bash
# not yet implemented: needs a --precision-matrix driver in modal_app.py
modal run modal_app.py --chunk 50 --precision bf16
modal run modal_app.py --chunk 50 --precision fp16
# ... per cell, with --tag naming the cell
```

---

## P4. Cross-architecture arm (VideoMAE)

**Question.** Is any detection-latency difference a property of world models
generally, or of the joint-embedding predictive objective specifically?

**Conditions.** The same 1200-clip benchmark through the VideoMAE adapter, which
already passes the full contract suite. VideoMAE ViT-B is roughly 4x cheaper
per forward than V-JEPA 2 ViT-L.

**Estimated cost.** ~0.3 GPU-hour, **~$0.30**.

**Status.** Adapter written and tested; never run. An adapter is a capability,
not a result.

**Launch.** Needs a `--adapter videomae` flag in `modal_app.py` (not yet
written; the extraction function currently hard-codes the V-JEPA 2 loader).

---

## P5. Mechanistic analysis of the latency gap (Phase 4.3)

**Question.** In the frames between a geometry alarm and the prediction-loss
alarm, what do the latents encode — which anomaly fired, or the disc's position
and velocity?

**Cost.** **Free.** CPU only; runs on latents already saved by P1.

**Contingent on P1** producing a non-empty latency gap. If no probe alarms
before a baseline there is no gap, and the script says so rather than probing
frames chosen some other way.

**Launch.**
```bash
python scripts/run_mechanistic.py results/<tag>
```

---

## Not parked: what is already finished

The CPU pilot at 128 px is exploratory, free, and allowed to finish. It is not
the registered experiment and is reported as such everywhere. See
`STATUS.md` for its live clip counts.
