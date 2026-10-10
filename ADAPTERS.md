# Adding a world model

The causal loop, the probes, the thresholds, the survival statistics and the
decision gate are shared across every architecture. A new model supplies two
methods. That is the whole contract, and it is what makes a cross-architecture
result a *comparison* rather than a pile of incomparable pipelines.

## The contract

```python
from lgw.adapters import BaseWindowAdapter, register

@register("my_model")
class MyAdapter(BaseWindowAdapter):
    def __init__(self, **kw):
        super().__init__(dim=512, name="my_model", **kw)
        self.net = load_my_model().eval()

    def encode_window(self, windows):
        """(B, t_ctx, H, W) uint8 -> (latent (B, dim), L0 target)

        `windows` is already the causal window ending at t. The latent is what
        the geometry probes consume; the target is what the prediction loss is
        scored against. Return `(z, None)` if the model has no native
        prediction loss.
        """
        h = self.net.encode(windows)          # (B, n_tokens, D)
        return h.mean(1), h

    def predict_final_step(self, ctx_windows):
        """Prediction of the final step, in the same space as the target.

        `ctx_windows` has its final tubelet overwritten, so nothing here has
        seen frame t-1 or t. Return None if there is no prediction head.
        """
        return self.net.predict_next(ctx_windows)
```

Then register a cheap constructor in `tests/test_adapter_contract.py` and run
`pytest tests -q`.

## Why the caller builds the windows

The information boundary is enforced by `BaseWindowAdapter.process`, not by your
code. You are handed `causal_window(clip, t)` and `context_window(clip, t)`
and cannot reach past them, so an adapter cannot leak the future by accident.
This is deliberate: "remember to be causal" is not a guarantee, it is a hope.

## What your adapter must pass

`tests/test_adapter_contract.py` runs these against **every** registered
adapter:

| test | what it rules out |
|---|---|
| `test_shapes_and_loss_window` | wrong latent dimension; a loss scored outside the evaluation window |
| `test_future_perturbation_does_not_change_the_past` | any lookahead: every frame after a cut is inverted, and `z_t`, `L0` and all probes must be unchanged |
| `test_l0_prediction_is_blind_to_the_predicted_frames` | a self-fulfilling loss whose context already saw its target |
| `test_runs_through_the_shared_harness` | NaNs or degenerate probe values |
| `test_every_registered_adapter_is_covered` | registering a model without submitting it to these checks |

The third test is the subtle one. It asserts the prediction does **not** move
when only `f_{t-1}, f_t` change, *and* that it **does** move when `f_{t-2}`
changes — so the guarantee cannot be satisfied trivially by a prediction that
ignores its input.

## Worked examples in this repository

| adapter | architecture | native loss | file |
|---|---|---|---|
| `vjepa2` | joint-embedding predictive, latent prediction | masked-token representation prediction | `src/lgw/vjepa2_adapter.py` |
| `videomae` | masked autoencoder, pixel reconstruction | masked-patch pixel MSE | `src/lgw/videomae_adapter.py` |
| `surrogate` | fixed random projection (plumbing only) | persistence | `src/lgw/surrogate.py` |

The first two sit on opposite sides of the pixel/latent divide, which is the
point: if an effect appears in one and not the other, that is informative about
the training objective rather than about the harness.

## The trap this exists to prevent

V-JEPA 2's reference implementation runs one encoder over the whole clip and
then masks tokens into context and target. Its own source says why that differs
from training:

```
# this is implemented here as in VJEPA training a separate encoder is
# used for target
```

With a single pass, context tokens have attended to the frames being predicted.
The loss is then partly self-fulfilling, the baseline looks stronger than it is,
and the comparison is biased *against* the hypothesis under test. If your model
has bidirectional attention over the window, you need the same two-pass
treatment — see `vjepa2_adapter.py` for a worked example.

**The trap is architecture-specific. Re-ask it for your model.** VideoMAE fails
differently: `VideoMAEForPreTraining` builds its regression targets by
patchifying *the same tensor it is handed*, so passing it the context window
scores the prediction against the overwritten pixels — a near-zero loss and a
baseline that looks far stronger than it is. Same bias, completely different
route. `videomae_adapter.py` predicts from the context window while patchifying
the target from the full window. The contract test catches both — but only for
adapters that are in it.

## Tubelets and temporal granularity

If your model embeds more than one frame per token step, set `tubelet`
accordingly. `context_window` overwrites the final `tubelet` frames, so the
context ends at `t - tubelet`. If your model is frame-wise, pass `tubelet=1`
and the context ends at `t-1`.

Note that `z_t` then summarises `tubelet` frames, so time-to-alarm is quantised
at that granularity. Both monitor families inherit the same quantisation, so
comparisons stay fair, but absolute latencies are coarser than the frame unit
suggests.

## Precision

Probe A is `||z_t - 2 z_{t-1} + z_{t-2}||`, a second difference of nearby
vectors — the textbook setting for catastrophic cancellation. Do not assume
bf16 is a free speed-up. `modal run modal_app.py --precision-check-only`
measures fp32 against bf16 and fp16 on the actual probe values and reports
relative error and rank correlation.

## Running your adapter

```python
from lgw.adapters import build_adapter
from lgw.extract import run_extraction
from lgw.analysis import analyse

adapter = build_adapter("my_model")
scores = {split: run_extraction(clips[split], adapter) for split in splits}
scores["anomaly"]["condition"] = conditions
analyse(scores, "results/my_model", model_name="my_model")
```

`analyse` writes `results.json`, `REPORT.md` and the survival curves, and
`scripts/make_paper.py` turns any `results.json` into the paper's tables.
