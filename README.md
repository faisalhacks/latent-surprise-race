# Does latent geometry detect anomalies sooner than prediction error?

When a frozen video world model meets a sudden physical or visual anomaly, two
internal signals could flag it: the **geometry** of its representations, and the
model's **own prediction error**. Both are known to carry physical information —
prediction error is the published mechanism by which V-JEPA detects physics
violations ([Garrido et al., 2025](https://arxiv.org/abs/2502.11831)), and
embedding geometry carries physical plausibility on its own
([Internò et al., 2026](https://arxiv.org/abs/2606.20707)). What is not known is
which one crosses an alarm threshold **sooner**. This repository is a
pre-registered, leakage-free protocol for measuring exactly that — a question
that turns out to be unusually easy to decide in advance by accident.

> **The registered experiment ran 2026-10-10; it is parked as exploratory per
> reviewer ruling 2026-10-12** (see [docs/V1_EXPLORATORY.md](docs/V1_EXPLORATORY.md)).
> The frozen analysis executed automatically with the inference launch, before
> independent review, so v1 cannot stand as confirmatory evidence. Its results
> are preserved and reportable as exploratory. A confirmatory v2 protocol with
> decoupled inference and analysis is proposed in
> [prereg/v2_PROPOSED.md](prereg/v2_PROPOSED.md), not frozen.

```bash
pip install -r requirements.txt
make test            # full suite, no GPU
make validate        # whole pipeline on CPU with a stand-in model, ~30 s
make paper           # -> paper/main.pdf
```

## Status

| phase | state |
|---|---|
| 1. Pre-registration frozen | ✅ **done** — [PREREGISTRATION.md](PREREGISTRATION.md), frozen before any code was written |
| 2. Harness, benchmark, adapters, statistics | ✅ **done** — tested end-to-end; orchestration validated at full 1200-clip scale |
| 2b. Registered 256px V-JEPA 2 run | ⚠️ **exploratory** — v1 ran but is parked as exploratory per reviewer ruling; v2 proposed, not frozen. See [docs/V1_EXPLORATORY.md](docs/V1_EXPLORATORY.md) and [prereg/v2_PROPOSED.md](prereg/v2_PROPOSED.md) |
| 3. Decision gate | ✅ **done** — automated in [analysis.py](src/lgw/analysis.py); verdict produced by the automated pipeline; v1 classified as exploratory |
| 4a. Second architecture (VideoMAE) | ✅ adapter **done** & passing contract; ⏸ **never run** |
| 4b. Mechanistic lag-window probe | ✅ **done**; contingent on a gate result |
| 4c. Compression / quantization study | ⏸ **parked, draft only** — [prereg/v2_PROPOSED.md](prereg/v2_PROPOSED.md), not frozen |
| 5. Manuscript | compiles; v1 results on disk, classified as exploratory |
| - exploratory 128px CPU pilot | stopped 2026-10-10; *not* the registered experiment, no outcome computed |

## Five ways this comparison goes wrong

**1. The baseline can cheat.** V-JEPA 2's reference forward runs one encoder over
the whole clip then masks tokens into context and target — so the context has
already attended to the frames it predicts. That makes the loss partly
self-fulfilling and the baseline unfairly *strong*, biasing *against* the
hypothesis. We take context from a second encoder pass whose final tubelet is
overwritten, and assert the prediction is unchanged when only the predicted
frames change, while still responding to `f_{t-2}`.

**2. Thresholds can leak.** Control clips split into disjoint calibration (sets
thresholds) and evaluation (measures false alarms) sets; the learned baseline is
cross-fitted within calibration so even its own threshold is honest.

**3. The registered metric can be degenerate.** Detection latency is
right-censored. Once fewer than half the clips alarm, the median pins at the
censoring value *for every monitor* and the test returns "no difference"
whatever the truth. RMDL was registered alongside it *before* running anything.

**4. The verdict can be vacuous.** If a baseline's detection rate equals its own
false-alarm rate it fires at chance, and beating it says nothing — a documented
collapse for JEPA novelty signals
([Pavuluri et al., 2026](https://arxiv.org/abs/2608.01336)). Every monitor
reports a **lift**; a passing verdict resting on such a baseline is marked
`verdict_is_vacuous`.

**5. The leakage trap is architecture-specific.** VideoMAE leaks by a completely
different route — it patchifies targets from whatever tensor it's handed. Same
bias, different mechanism. "Is this loss causal?" must be re-asked per model.

## Causality is structural, not conventional

Monitors are streaming objects fed one latent at a time, so referencing a future
frame isn't something the implementation can express. Then it's tested anyway:
invert every frame after `t` and assert `z_t`, the loss and all three probes are
bit-identical — including against the real V-JEPA 2 architecture.

## Monitors

| id | definition |
|---|---|
| `loss_L0` | V-JEPA 2 predicting its own final-tubelet tokens from frames `<= t-2` — the pretraining objective, evaluated causally |
| `loss_L1` | ridge next-step predictor on the same latent sequence the probes see — information-matched |
| `probe_A` | trajectory curvature, `‖z_t − 2z_{t−1} + z_{t−2}‖` |
| `probe_B` | representational distance, `1 − cos(z_t, mean(z_{t−5..t−1}))` |
| `probe_C` | covariance shift, `‖Cov(z_{t−4..t}) − Σ_base‖_F` |

## Reproducing

```bash
python -m pytest tests -q                  # full suite, no GPU
python scripts/run_local_validation.py 40  # whole pipeline, CPU, ~30 s
python scripts/run_pilot_cpu.py 40 70 50   # exploratory 128px pilot, checkpointed
python scripts/run_pilot_cpu.py --analyse-only   # analyse whatever is on disk
make paper                                 # regenerate figures, tables, PDF
```

Full detail, including measured runtimes per resolution and the GPU commands
for the parked run, is in [REPRODUCING.md](REPRODUCING.md).

## Using it for your own model

Two methods, ~25 lines — see [ADAPTERS.md](ADAPTERS.md):

```python
@register("my_model")
class MyAdapter(BaseWindowAdapter):
    def encode_window(self, windows):        # -> (latent, loss target)
    def predict_final_step(self, ctx):       # -> prediction
```

The windows are built by the caller, so an adapter can't leak the future even by
mistake. Every registered adapter runs through the same contract tests;
registering one without submitting it to them fails the suite by design.

## Key documents

| file | what it is |
|---|---|
| [PREREGISTRATION.md](PREREGISTRATION.md) | the frozen v1 protocol — the contract |
| [DEVIATIONS.md](DEVIATIONS.md) | dated log of every change, with provenance |
| [docs/PARKED_EXPERIMENTS.md](docs/PARKED_EXPERIMENTS.md) | each parked run: question, cost, launch command |
| [prereg/v2_PROPOSED.md](prereg/v2_PROPOSED.md) | compression study — **draft, not frozen** |
| [LIMITATIONS.md](LIMITATIONS.md) | boundary conditions, including measured ones |
| [STATUS.md](STATUS.md) | current blocker and costs |
| [ADAPTERS.md](ADAPTERS.md) | how to add a world model |
| [REPRODUCING.md](REPRODUCING.md) | step-by-step with measured runtimes |

## Reporting

Every run writes `results.json`, a `REPORT.md` and Kaplan–Meier curves under
**two threshold regimes** and **two metric families**. All four cells are
reported whatever the outcome, so a degenerate median can't hide an effect and a
favourable cell can't be cherry-picked.

The paper regenerates entirely from that JSON. With no run analysed it compiles
with visible `PENDING` markers — a skeleton that silently contained invented
numbers would be worse than one that is obviously incomplete.

## License

MIT (see [LICENSE](LICENSE)) — a default chosen so the harness is reusable;
change it if it doesn't suit you. Citation metadata in
[CITATION.cff](CITATION.cff).
