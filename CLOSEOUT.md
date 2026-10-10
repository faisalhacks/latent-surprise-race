# Closeout

**Status: registered run EXECUTED 2026-10-10; write-up pending.**

Superseded: this document previously read BLOCKED (compute). GPU access was
obtained on 2026-10-10 and the registered experiment ran. The history of that
block is kept in `DEVIATIONS.md` rather than rewritten here.

## The registered run

| | |
|---|---|
| Launched | 2026-10-10 17:42:37 IST, run tag `phase2-20261010-174237` |
| Compute | Modal, NVIDIA **A10G**, fp32/TF32 (a T4 was used only for an access probe) |
| Modal profile | `riyann00b` |
| Scale | 1200 clips, exactly as frozen |
| Cost | 7,554 GPU-seconds (2.10 GPU-hours), 963 s wall |
| Protocol | frozen v1, unmodified |

Analysis artifacts (`results.json`, `REPORT.md`, survival curves) were produced
automatically by the same entrypoint and are on disk. **Their contents are
deliberately not reproduced in this document**; the write-up is pending the
author's decision. See `DEVIATIONS.md` for why the intended
publish-before-analysis ordering did not hold.

## What was previously blocked

## Why it had been blocked

No GPU credit. Modal refuses GPU functions on any workspace without a payment
method on file, and this refusal happens at app-hydration time, before any
compute is requested. It is an access gate, not a quota or a budget ceiling:
five separate workspaces were checked and four were refused identically.

**This is not a code blocker.** The harness is complete and tested. The
orchestration was validated end-to-end at the full 1200-clip scale using
randomly initialised weights, so the only step in the registered run that has
never executed is the GPU forward itself.

## Cost, as estimated beforehand

| item | estimate |
|---|---|
| Precision check (fp32 / bf16 / fp16) | ~3 GPU-minutes, ~$0.05 |
| **Registered Phase 2 run** | **~1.0 GPU-hour, ~$0.50–1.00** |
| Wall clock with 24-way fan-out | ~5–10 minutes |

Any of A10G, L4 or L40S is sufficient.

## What exists instead

- The protocol, benchmark, probes, baselines, statistics and decision gate,
  all tested.
- A second architecture adapter (VideoMAE) passing the same contract, never run.
- A mechanistic lag-window probe, contingent on a gate result.
- A manuscript that compiles with every registered number marked pending.
- An incomplete exploratory CPU pilot at reduced resolution, with **no outcome
  numbers computed** — see `exploratory_pilot_128px/README.md`.

## What was deliberately not done

- No registered numbers were estimated, simulated, or filled in.
- No GPU job was run on any account.
- No pilot outcome was analysed, plotted or reported.
- The frozen v1 pre-registration was not modified, reworded or re-dated.

## To close this out

Steps 1-3 are done. Remaining:

4. Decide whether to proceed to the write-up, then
   `python scripts/make_paper.py` and rebuild the PDF.
5. Run the mechanistic probe only if the gate shows a latency gap.

The gate verdict is computed by code and written into the report; it is not a
judgement call made while reading tables.
