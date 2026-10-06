# Reproducing

Everything here is deterministic from one seed (`config.SEED = 20261006`). No
dataset is distributed because none needs to be: the clips are regenerated
bit-identically from the frozen code.

## 0. Install

```bash
pip install -r requirements.txt
python -m pytest tests -q            # 32 tests, no GPU required, ~20 s
```

If the tests do not pass, stop. They are the guarantees, not a formality.

## 1. Validate the harness on CPU (free, ~30 s)

```bash
python scripts/run_local_validation.py 40
```

Runs the complete pipeline — generation, extraction, probes, thresholds,
survival analysis, gate, report — with a random-projection stand-in model. It
proves the plumbing. Its numbers are **not** a result and the report says so.

## 2. The registered experiment (GPU)

Requires a Modal account with a payment method; GPU functions are refused
without one.

```bash
export PYTHONIOENCODING=utf-8                    # Windows: the Modal CLI prints
                                                 # glyphs cp1252 cannot encode
modal run modal_app.py --precision-check-only    # ~$0.05, decides fp32 vs bf16
modal run modal_app.py --chunk 50                # full Phase 2
modal volume get lgw-results <tag>/ ./results    # latents + report
```

| | |
|---|---|
| clips | 1200 (1000 anomalous, 100 calibration controls, 100 evaluation controls) |
| forwards | 45 encoder + 15 predictor per clip |
| A10G, fp32/TF32 | ~1.0 GPU-hour, ~\$1.0–1.5 |
| A10G or L4, bf16 | ~0.4 GPU-hour, ~\$0.4–0.7 |

Derivation: ~1.75 TFLOPs per encoder forward (326M parameters × 2048 tokens,
plus 24 layers of 2048-token attention) and ~0.5 TFLOPs per predictor forward,
so ~86 TFLOPs per clip and ~1.0e17 FLOPs overall.

`--chunk 50` halves container cold-start overhead relative to the default.
Raise it further to cut cost, lower it to cut wall-clock.

## 3. Without a GPU

Measured on a 16-core CPU: **9.66 s per encoder forward** at the frozen 256 px,
i.e. ~478 s per clip and **~159 hours** for the full design. Not viable.

A reduced-resolution pilot is:

```bash
python scripts/run_pilot_cpu.py 40 70 50        # 128 px, ~4–5 h, checkpointed
```

This is **not** the registered experiment. Only the spatial grid changes
(16×16 → 8×8 patches); context length, tubelet, probe definitions, the L0
construction, thresholds and metrics are untouched. 96 px was rejected because
the disc would then be smaller than one patch. Use it to decide whether the paid
run is worth running, not as a result.

Measured operating points on this machine:

| res | t_ctx | tokens | s/forward | s/clip | 240 clips |
|---|---|---|---|---|---|
| 256 | 16 | 2048 | 9.66 | 478 | 32 h |
| 128 | 16 | 512 | 1.10 | 55 | 3.7 h |
| 96 | 16 | 288 | 0.65 | 33 | 2.2 h |
| 64 | 16 | 128 | 0.32 | 16 | 1.1 h |

## 4. Rebuild the paper

```bash
python scripts/make_figures.py       # dataset + method figures, dataset stats
python scripts/make_paper.py         # tables, macros, prose from results.json
cd paper && tectonic main.tex        # -> main.pdf
```

`make_paper.py` picks the most recent real run automatically; pass a path to
choose one, or `--list` to see them. With no run analysed it emits visible
`PENDING` placeholders and a DRAFT banner rather than plausible numbers.

## What is pinned, and what that buys

| pinned by | guarantee |
|---|---|
| `config.SEED` | identical clips on every machine |
| `tests/test_config_frozen.py` | the protocol constants cannot drift unnoticed |
| `tests/test_causality.py` | no monitor can read a future frame |
| `tests/test_adapter_contract.py` | every model is held to the same contract |
| `tests/test_leakage.py` | thresholds never see the data they are scored on |
| `scripts/make_paper.py` | no number is transcribed into the paper by hand |

## Known environment notes

- `transformers==4.57.3` is pinned because the adapter depends on
  `VJEPA2Model` exposing its predictor and accepting `context_mask` /
  `target_mask`.
- A tiny `VJEPA2Config` needs `head_dim >= 12`: the 3D RoPE uses
  `d_dim = 2*((head_dim//3)//2)` per axis and degenerates below that. The test
  configs use `hidden_size=96, num_attention_heads=4`.
- On Windows, set `PYTHONIOENCODING=utf-8` before `modal run`.
