# Closeout

**Status: BLOCKED — compute.**

## What is blocked

The registered Phase 2 experiment: 1200 clips at 256 px through frozen
V-JEPA 2, under the protocol frozen in `PREREGISTRATION.md`.

## Why

No GPU credit. Modal refuses GPU functions on any workspace without a payment
method on file, and this refusal happens at app-hydration time, before any
compute is requested. It is an access gate, not a quota or a budget ceiling:
five separate workspaces were checked and four were refused identically.

**This is not a code blocker.** The harness is complete and tested. The
orchestration was validated end-to-end at the full 1200-clip scale using
randomly initialised weights, so the only step in the registered run that has
never executed is the GPU forward itself.

## Cost to unblock

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

1. Enable billing on a Modal workspace.
2. `modal run modal_app.py --precision-check-only`
3. `modal run modal_app.py --chunk 50`
4. `python scripts/make_paper.py && cd paper && tectonic main.tex`

The gate verdict is computed by code and written into the report; it is not a
judgement call made while reading tables.
