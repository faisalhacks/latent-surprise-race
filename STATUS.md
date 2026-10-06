# Status: 2026-10-07

## Summary

The harness, the benchmark, the adapter contract and the manuscript are
complete and tested (39 tests). Two things remain outstanding, one of which
needs you:

1. **The registered GPU run is still blocked** on a Modal payment method. See
   "Where this stands" below. Nothing in the code is waiting on anything else.
2. **A CPU pilot at 128 px is running** (`scripts/run_pilot_cpu.py`), to answer
   whether the paid run is worth running and in what form. It is not the
   registered experiment and its output is labelled as a pilot everywhere it
   appears.

The paper compiles today with visible PENDING placeholders. The moment either
run is analysed, `python scripts/make_paper.py && cd paper && tectonic main.tex`
fills in every table, figure and numeric sentence.

---

# Original status: 2026-10-06

## Where this stands

Phases 1 and 2 are **built, tested and validated**. Phase 2's actual V-JEPA 2
inference is **blocked on one external thing**: the Modal workspace
(`workspace-C`) has no payment method, so every GPU class is refused at
app-hydration time:

```
InvalidError: Please add a payment method to use A10G GPU functions.
```

This is not a code problem. Everything that can be verified without a GPU has
been.

## What is verified

| check | how | result |
|---|---|---|
| Pre-registration frozen before any code ran | `PREREGISTRATION.md`, `DEVIATIONS.md` | done |
| Protocol constants cannot drift | `tests/test_config_frozen.py` | pinned |
| Probes equal their closed-form definitions | `tests/test_probes_correct.py` | exact to 1e-9 |
| Fast covariance identity equals the naive D x D computation | validated at freeze time | machine precision |
| Probes cannot see the future | prefix invariance + future inversion | bit-identical |
| L0 context cannot see the predicted frames | tested against real V-JEPA 2 architecture | bit-identical |
| Thresholds never touch evaluation data | `tests/test_leakage.py` | enforced |
| t-major token layout (tubelet j owns `[j*hw, (j+1)*hw)`) | tested on the real `Conv3d` patch embedder | confirmed |
| Predictor returns target tokens only | read from `VJEPA2Predictor.forward` | `hidden_states[:, N_ctxt:]` |
| Whole pipeline, CPU, end to end | `scripts/run_local_validation.py` | runs, reports, gates |
| Modal orchestration at full 1200-clip scale | `modal run modal_app.py --arch-dryrun` | 48 chunks, 55s wall, report generated |
| Real ViT-L weights load and run through the adapter | `scripts/bench_cpu.py` | 326M params, 2048 tokens/window |

22 tests pass.

## What the blocker costs

Measured on this machine: **9.66 s per encoder forward** (batch 1, 16 frames,
256 px, fp32, 10 threads). Each clip needs 45 encoder forwards plus 15
predictor forwards, so:

| route | estimate |
|---|---|
| this CPU, full 1200 clips | **~159 hours** - not viable |
| this CPU, 100 clips | ~13 hours - still not viable, and underpowered |
| A10G, fp32/TF32, full run | ~1.0 GPU-hour, **~$1.0-1.5** |
| A10G or L4, bf16, full run | ~0.4 GPU-hour, **~$0.4-0.7** |

Derivation: ~1.75 TFLOPs per encoder forward (326M params x 2048 tokens, plus
24 layers of 2048-token attention), ~0.5 TFLOPs per predictor forward, so
~86 TFLOPs per clip and ~1.0e17 FLOPs for 1200 clips.

So the full Phase 2 fits the $2 budget either way, with real headroom in bf16.

## To unblock

1. Add a payment method at <https://modal.com/settings/billing>. The included
   monthly credits may cover this run outright; the card is a gate on GPU
   access, not necessarily a charge.
2. Then, in order:

```bash
cd latent-geometry-warning
set PYTHONIOENCODING=utf-8                     # Windows only

# ~$0.05: does bf16 preserve probe A? It is a second difference, so this is
# the one number that decides whether the cheap precision is usable.
modal run modal_app.py --precision-check-only

# ~$0.4-1.5 depending on the answer above. chunk=50 halves cold-start overhead.
modal run modal_app.py --chunk 50                        # fp32 (safe default)
modal run modal_app.py --chunk 50 --precision bf16       # if precision check is clean

modal volume get lgw-results <tag>/ ./results            # latents + report
```

The run prints the Phase 3 gate verdict and writes `REPORT.md`,
`results.json` and `survival_curves.png`.

## Decide the precision from evidence, not convenience

Probe A is `||z_t - 2 z_{t-1} + z_{t-2}||`, a second difference of nearby
vectors, which is the textbook setting for catastrophic cancellation. bf16
carries ~3 significant decimal digits. `--precision-check-only` measures fp32
vs bf16 vs fp16 on the actual probe values and reports median and max relative
error plus rank correlation, so the choice is made on data. Default is fp32.

## What is deliberately not built yet

Phase 4 (cross-architecture, MuJoCo, mechanistic probing). The protocol gates it
on the Phase 3 verdict and requires explicit approval before spending compute on
it, so it stays unwritten until Phase 2 produces a verdict.

## Real weights now run end to end (CPU, 2 clips)

`scripts/smoke_real_cpu.py` confirmed the full path against the **real**
pretrained ViT-L, not just the architecture: `D = 1024`, 2048 tokens per window,
every monitor finite, 1190 s for 2 clips. The predictor and the L0 loss behave.

It also surfaced two power risks, detailed in `LIMITATIONS.md`:

* **`loss_L0` spans only +/-2%** (7.57 to 7.94). An MSE averaged over
  256 x 1024 values dilutes a spatially local anomaly. If that holds at scale,
  L0 alarms at chance - and beating a baseline that detects nothing is **not**
  evidence for H1. The report now carries a `lift` column, flags such baselines,
  and marks a PROCEED verdict `verdict_is_vacuous` in that case.
* **Probes A and C vary more between clips than within one** (probe C: ~191 vs
  ~206 across two clips, but only ~3 units of movement inside each). With
  pooled thresholds, that makes them partly per-clip constants rather than
  change detectors. Probe B, scale-invariant, does not show this.

Neither is patched. Both postdate the freeze, and the second is a design
decision rather than a bug - see the open question below.

## Open question for you

If the real run shows probes A and C near chance while B is not, the fix is to
standardise each probe per clip against its own pre-injection frames. Probe C is
undefined before frame 15, so it has no within-clip reference and `Sigma_base`
would have to serve instead. That is a change to a frozen probe definition, so
it is your call, not mine: either report A and C as specified and note the
weakness, or add a per-clip-standardised variant as a clearly-labelled
post-freeze robustness analysis.

## One thing to expect from the verdict

The orchestration dry run reported `primary median degenerate: True`. That is an
artifact of random weights, but it is also the failure mode flagged in
`LIMITATIONS.md`: if fewer than half the clips alarm within the 15-frame window,
the pre-registered median gate **cannot fire whatever the true effect**. The
RMTA secondary exists for precisely that case and is reported unconditionally.
If the real run lands there, the honest reading is that the pre-registered
primary was uninformative and the secondary carries the evidence.
