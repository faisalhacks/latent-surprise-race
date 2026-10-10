# Claims and their evidence

Every claim this project makes, mapped to the artifact that supports it.
Anything without evidence is marked **UNSUPPORTED**. The headline hypothesis is
one of them.

Status key: **SUPPORTED** (evidence exists and is reproducible) ·
**UNSUPPORTED/PENDING** (no evidence yet; experiment not run) ·
**PARTIAL** (evidence exists but is limited in a way stated inline).

---

## The headline hypothesis

| claim | status | evidence |
|---|---|---|
| Representation geometry crosses an alarm threshold sooner than the model's own prediction error in a frozen video world model. | **UNSUPPORTED / PENDING** | **None.** The registered experiment has not been run. No detection latency, no gate verdict, no survival curve, no $p$-value exists for this claim. It is open. |

Nothing in this repository should be read as evidence for or against it.

---

## Claims about the protocol and apparatus

| claim | status | evidence |
|---|---|---|
| Monitors cannot read a future frame. | SUPPORTED | `tests/test_causality.py::test_probe_prefix_invariance`, `::test_probe_future_perturbation_does_not_change_past`; and against the real architecture, `tests/test_adapter_contract.py::test_future_perturbation_does_not_change_the_past` (runs for every registered adapter). Inverts all frames after a cut and asserts bit-identity. |
| The $\mathcal{L}_0$ prediction is blind to the frames it predicts, and not vacuously so. | SUPPORTED | `tests/test_adapter_contract.py::test_l0_prediction_is_blind_to_the_predicted_frames` — asserts no change when $f_{t-1},f_t$ change, **and** a change when $f_{t-2}$ changes. |
| Thresholds never see the data they are evaluated on. | SUPPORTED | `tests/test_leakage.py` (3 tests); disjoint `control_calib` / `control_eval` seed ranges in `datagen.build_split`; 5-fold cross-fitting for $\mathcal{L}_1$ in `analysis.add_l1_baseline`. |
| Probe values equal their closed forms in the pre-registration. | SUPPORTED | `tests/test_probes_correct.py` — agreement to $10^{-9}$. |
| The $O(k^2d)$ covariance identity equals the naive $d\times d$ computation. | SUPPORTED | Verified at freeze time to $\le 2\times10^{-16}$ relative; the identity is implemented in `probes._fro_sq_diff_lowrank`. |
| V-JEPA 2's token layout is t-major (tubelet $j$ owns indices $[j\cdot hw,(j{+}1)\cdot hw)$). | SUPPORTED | `tests/test_vjepa2_adapter.py::test_token_index_block_maps_to_tubelet` — tested on the real `Conv3d` patch embedder, which is local, so the claim is checkable exactly. |
| The frozen protocol constants cannot drift unnoticed. | SUPPORTED | `tests/test_config_frozen.py` pins every constant; editing one fails CI. |
| The orchestration works at full 1200-clip scale. | SUPPORTED | Modal dry run with randomly initialised weights: 48 chunks, 55 s wall, report generated. Only the GPU forward itself is unexercised. |
| No number in the manuscript is transcribed by hand. | SUPPORTED | `scripts/make_paper.py` generates all macros, tables and numeric prose; `tests/test_paper_generation.py` (8 tests) pins the behaviour, including that a missing diagnostic never becomes a passing claim. |

## Claims about the benchmark

| claim | status | evidence |
|---|---|---|
| Anomalous and control clips share pre-injection history exactly, so comparisons are paired. | SUPPORTED | `tests/test_datagen.py::test_anomaly_matches_control_before_injection`. |
| The occlusion condition hides the disc completely in 77.5% of post-injection frames, partially in 17.2%, and never reveals it again in 28.0% of clips. | SUPPORTED | Measured over all 250 clips of the condition by `scripts/make_figures.py::dataset_macros`; regenerated into the paper at build time. |
| The occlusion condition therefore measures object permanence rather than partial corruption. | SUPPORTED (interpretation of the above) | Follows from the measurement. Interpreted against Xie & Alanwar (2026), who document an encoder/predictor dissociation in V-JEPA 2 in exactly this regime. |
| Each anomaly is observable from the first post-injection frame. | PARTIAL | True by construction for friction (per-frame drag), occlusion and frame drop. For `gravity_flip` the acceleration changes at once but position diverges gradually — stated in the paper as the subtlest condition at short latency, not asserted as equally observable. |

## Claims about the adapter contract

| claim | status | evidence |
|---|---|---|
| The contract is architecture-agnostic, not V-JEPA-shaped. | SUPPORTED | Two adapters from opposite sides of the pixel/latent divide (V-JEPA 2, VideoMAE) pass the identical parameterised suite — `tests/test_adapter_contract.py`, 14 tests. |
| The leakage trap is architecture-specific. | SUPPORTED | V-JEPA 2 leaks via a single encoder pass; VideoMAE leaks by patchifying targets from whatever tensor it is handed. Both are handled in their adapters and pinned by the same test. |
| VideoMAE behaves differently from V-JEPA 2 on this benchmark. | **UNSUPPORTED** | The VideoMAE adapter has **never been run** on the benchmark. An adapter is a capability, not a result. |

## Claims about the statistics

| claim | status | evidence |
|---|---|---|
| A censored median is degenerate below 50% detection and cannot separate monitors. | SUPPORTED | Follows from the definition; demonstrated in the orchestration dry run, which reported `primary_median_degenerate: true`. |
| A 1% per-frame threshold compounds to roughly 14% clip-level over 15 frames. | SUPPORTED | Measured directly in `stats` validation; the clip-matched regime recovers exactly 5%. |
| RMDL separates monitors where the median cannot. | SUPPORTED | Demonstrated on synthetic signals with known ground truth during `stats` validation. |
| A baseline firing at chance makes a passing verdict vacuous. | SUPPORTED (as a logical point, implemented as a diagnostic) | `analysis._run_regime` computes lift and `verdict_is_vacuous`; `tests/test_paper_generation.py` pins that a missing lift value is recomputed rather than defaulted to a benign one. The underlying collapse is documented by Pavuluri et al. (2026). |
| `loss_L0` will fire at chance on the registered run. | **UNSUPPORTED** | Only a $\pm2\%$ dynamic range observed on **two** clips of real weights. That is a risk flagged in `LIMITATIONS.md`, not a finding. |
| Probes A and C are dominated by per-clip offsets. | **UNSUPPORTED** | Based on **two** clips. Flagged as a risk; not measured at scale. |

## Claims we explicitly do not make

- We do not claim to be first at anything. Geometry of frozen embeddings
  carrying physical information is established (Internò et al., 2026);
  prediction error as the physics-violation mechanism is established
  (Garrido et al., 2025); the at-chance collapse of self-supervised novelty
  signals is documented (Pavuluri et al., 2026). Where we believe a practice is
  uncommon — the lift/vacuity check — we say "to our knowledge".
- We do not claim **anticipation**. The benchmark injects anomalies abruptly, so
  it measures reaction time. Any anticipation claim would need a gradual-onset
  design we do not have.
- We make no claim about compression. That study is drafted, unfrozen and unrun.
- We make no claim about real-world or robotic settings.
