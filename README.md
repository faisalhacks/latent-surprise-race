# Does latent geometry warn before prediction error?

A pre-registered, leakage-free benchmark for a specific claim about video world
models: that the **geometry** of a frozen model's internal states registers a
sudden physical or visual anomaly *before* the model's own **prediction error**
does.

If true it matters, because prediction error is the standard online surprise
signal for model-based control, and anything earlier buys reaction time. The
claim is also easy to confirm by accident, which is why this repository is as
much a protocol as an experiment.

```bash
pip install -r requirements.txt
make test            # 32 tests, no GPU
make validate        # whole pipeline on CPU, ~30 s
make paper           # -> paper/main.pdf
```

## Status

| phase | state |
|---|---|
| 1. Pre-registration frozen | done — [PREREGISTRATION.md](PREREGISTRATION.md) |
| 2. Harness + V-JEPA 2 inference | code complete and validated; **GPU run blocked** on Modal billing, see [STATUS.md](STATUS.md) |
| 3. Decision gate | automated in [analysis.py](src/lgw/analysis.py) |
| 4. Expansion | gated on Phase 3 by the protocol |
| 5. Manuscript | [paper/](paper/) compiles; numbers auto-populate from a run |

## Three ways this comparison goes wrong, and what we do about it

**The baseline can cheat.** V-JEPA 2's reference forward runs one encoder over
the whole clip and then masks tokens into context and target — so the context
has already attended to the frames it predicts. That makes the loss partly
self-fulfilling and the baseline unfairly *strong*, biasing against the
hypothesis. We take context from a second encoder pass whose final tubelet is
overwritten, and assert the prediction is unchanged when only the predicted
frames change — while still responding to `f_{t-2}`, so the guarantee isn't
vacuous.

**Thresholds can leak.** Control clips are split into disjoint calibration
(sets thresholds) and evaluation (measures false alarms) sets. The learned
baseline is cross-fitted within calibration so even its own threshold is honest.

**The headline metric can be degenerate.** Time-to-alarm is right-censored. Once
fewer than half the clips alarm, the median is pinned at the censoring value
*for every monitor* and the registered test returns "no difference" whatever the
truth. We kept the registered median as primary and registered RMTA alongside it
*before running anything*, so a degenerate primary is visible rather than
silently null.

**And a fourth: the verdict can be vacuous.** If a baseline's detection rate
equals its own false-alarm rate, it fires at chance — and beating it says
nothing about the hypothesis. Every monitor reports a **lift**; a passing
verdict that rests on such a baseline is marked `verdict_is_vacuous`.

## Causality is structural, not conventional

Monitors are streaming objects fed one latent at a time, so referencing a future
frame is not something the implementation can express. Then it is tested
anyway: invert every frame after `t` and assert `z_t`, the loss and all three
probes are bit-identical — including against the real V-JEPA 2 architecture.

## Monitors

| id | definition |
|---|---|
| `loss_L0` | V-JEPA 2 predicting its own final-tubelet tokens from frames `<= t-2` — the pretraining objective, evaluated causally |
| `loss_L1` | ridge next-step predictor on the same latent sequence the probes see — information-matched |
| `probe_A` | trajectory curvature, `‖z_t − 2z_{t−1} + z_{t−2}‖` |
| `probe_B` | representational distance, `1 − cos(z_t, mean(z_{t−5..t−1}))` |
| `probe_C` | covariance shift, `‖Cov(z_{t−4..t}) − Σ_base‖_F` |

A probe is an **Early Warner** only if it beats **both** baselines by ≥1 frame
of median time-to-alarm at Holm-adjusted `p < 0.01`.

## Using it for your own model

Two methods, about twenty-five lines — see [ADAPTERS.md](ADAPTERS.md):

```python
@register("my_model")
class MyAdapter(BaseWindowAdapter):
    def encode_window(self, windows):        # -> (latent, loss target)
    def predict_final_step(self, ctx):       # -> prediction
```

The windows are built by the caller, so an adapter cannot leak the future even
by mistake. Every registered adapter is run through the same contract tests;
registering one without submitting it to them fails the suite by design. That is
what makes a cross-architecture row comparable rather than merely adjacent.

## Layout

```
PREREGISTRATION.md   the frozen protocol — the contract
DEVIATIONS.md        dated log of every change, with provenance
LIMITATIONS.md       boundary conditions, including measured ones
STATUS.md            current blocker, costs, exact commands
ADAPTERS.md          how to add a world model
REPRODUCING.md       step-by-step, with measured runtimes
paper/               manuscript; every number generated from results.json
src/lgw/             dataset, probes, adapters, statistics, analysis, gate
modal_app.py         Phase 2 on Modal: fan-out, precision check, analysis
scripts/             validation, pilot, benchmarks, figure and paper builders
tests/               causality, leakage, probe correctness, adapter contract
```

## Reporting

Every run writes `results.json`, a `REPORT.md` and Kaplan–Meier curves under
**two threshold regimes** (per-frame-matched 1% FPR, the registered primary;
clip-matched 5% FPR) and **two metric families** (median time-to-alarm, and the
censoring-robust RMTA). All four cells are reported whatever the outcome, so a
degenerate median cannot hide an effect and a favourable cell cannot be
cherry-picked.

The paper regenerates entirely from that JSON. With no run analysed it compiles
with visible `PENDING` placeholders and a DRAFT banner — a skeleton that
silently contained invented numbers would be worse than one that is obviously
incomplete.

## License

MIT (see [LICENSE](LICENSE)) — a default chosen so the harness is reusable;
change it if it does not suit you.
