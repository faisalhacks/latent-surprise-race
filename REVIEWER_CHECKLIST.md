# Reviewer checklist

A compact set of checks for the three ways a result like this goes wrong:
**leakage**, **post-outcome tuning**, and **unreported failures**. Each item
says what to run or read, and what a bad answer looks like.

This repository is **post-run**: v1 executed on 2026-10-10 and is parked as
exploratory per reviewer ruling 2026-10-12. Outcome-side items are therefore
live, but the evidence they bear on is exploratory rather than confirmatory.
See `docs/V1_EXPLORATORY.md`.

---

## A. Leakage — can any monitor see what it should not?

| # | check | how | bad answer |
|---|---|---|---|
| A1 | Does any monitor read a future frame? | `pytest tests/test_causality.py tests/test_adapter_contract.py -q` | Any failure. Tests invert every frame after a cut and assert bit-identity for earlier frames. |
| A2 | Is the guarantee vacuous — does the prediction simply ignore its input? | `test_l0_prediction_is_blind_to_the_predicted_frames` | A prediction that also fails to move when `f_{t-2}` changes. The test asserts it *does*. |
| A3 | Does the prediction-error baseline see the frames it predicts? | Read `src/lgw/vjepa2_adapter.py` docstring; confirm two encoder passes | A single-pass implementation. V-JEPA 2's reference forward encodes the whole clip then masks, which leaks. |
| A4 | Same question for every other adapter. | `src/lgw/videomae_adapter.py` | Assuming the V-JEPA 2 fix transfers. VideoMAE leaks differently: it patchifies targets from whatever tensor it is handed. |
| A5 | Are thresholds fitted on data they are scored against? | `pytest tests/test_leakage.py -q`; check `control_calib` / `control_eval` seed ranges are disjoint | Shared clips, or a threshold computed on anomalous data. |
| A6 | Is the learned baseline's own threshold optimistic? | `analysis.add_l1_baseline` — 5-fold cross-fitting | In-sample scoring of $\mathcal{L}_1$ on the clips that fitted it. |
| A7 | Do probe frames leak into each other in the mechanistic analysis? | `tests/test_mechanistic.py::test_probe_does_not_hallucinate_signal_from_clip_identity` | Random (not clip-grouped) splits. Frames from one clip are near-duplicates. |

## B. Post-outcome tuning — was anything chosen after seeing the answer?

| # | check | how | bad answer |
|---|---|---|---|
| B1 | Does the frozen protocol match its state at the freeze commit? | `diff --strip-trailing-cr <(git show $(jq -r .freeze_commit_sha FREEZE.json):PREREGISTRATION.md) PREREGISTRATION.md` | Any content difference. Should be zero. |
| B2 | Is every post-freeze change dated and justified? | `DEVIATIONS.md` | An undocumented change, or a change dated vaguely. |
| B3 | Were the secondary metrics added before or after real inference? | `DEVIATIONS.md`, rows dated 2026-10-06 | A secondary metric introduced after an outcome was known. The log must say which. |
| B4 | Was the test tail (one- vs two-sided) fixed in advance? | `DEVIATIONS.md`, Section 3 row | A tail chosen post hoc. Both tails are reported in every table so the choice cannot be hidden. |
| B5 | Were equivalence margins set before or after seeing effects? | `prereg/v2_PROPOSED.md` §3 | Margins filled in. They are deliberately left TODO; a filled-in margin with no justification is the failure mode. |
| B6 | Can protocol constants drift silently? | `pytest tests/test_config_frozen.py -q` | Passing after an edit. The test pins every constant. |
| B7 | Is the confirmatory set distinct from the development set? | v2 draft §4 specifies a fresh seed | Reusing the seed the harness was debugged on for confirmatory clips. |

## C. Unreported failures — what is being left out?

| # | check | how | bad answer |
|---|---|---|---|
| C1 | Does the baseline actually detect anything? | `lift` column in the report; `verdict_is_vacuous` in `results.json` | A baseline with lift $\approx 0$ and a "probe wins" conclusion drawn against it. Beating a monitor that fires at chance shows nothing. |
| C2 | Is the headline metric degenerate? | `primary_median_degenerate` in `results.json` | Median gains all `+0.0` reported as "no difference". That is censoring, not equality. |
| C3 | Are both threshold regimes reported? | `REPORT.md` has both | Only the favourable regime shown. Both are emitted unconditionally. |
| C4 | Are both metric families reported? | median **and** RMDL in every table | Only the one that gives the nicer answer. |
| C5 | Are per-condition results shown, not just pooled? | per-condition table in `REPORT.md` | Pooling across anomaly families with very different effect magnitudes. |
| C6 | Are any numbers hand-transcribed? | `scripts/make_paper.py`; `pytest tests/test_paper_generation.py -q` | A number in the PDF that is not in `results.json`. |
| C7 | Do pending results look like real ones? | Build the paper with no run analysed | Plausible placeholder values. It must emit visible `PENDING` markers and delete stale figures. |
| C8 | Is an adapter presented as a result? | `CLAIMS.md` | "We evaluate two architectures" when one was only implemented. VideoMAE has never been run. |
| C9 | Is the exploratory pilot presented as the registered experiment? | `exploratory_pilot_128px/README.md`; paper §Results | Pilot numbers in a results table, or pooled with registered ones. |
| C10 | Are known risks disclosed? | `LIMITATIONS.md` | Omitting the $\pm2\%$ $\mathcal{L}_0$ range or the per-clip offsets in probes A and C. |

## D. Post-run specific (this snapshot)

| # | check | expected |
|---|---|---|
| D1 | Has the registered experiment run? | **Yes, ran 2026-10-10** (tag `phase2-20261010-174237`). v1 is exploratory, not confirmatory. |
| D2 | Are registered outcome numbers present? | **Yes**, in the tables and on disk. They are reported as **exploratory** evidence per the ruling. |
| D3 | Is the headline hypothesis claimed as confirmatory? | **No.** `CLAIMS.md` marks it NOT CONFIRMATORY and labels it exploratory. |
| D4 | Has any pilot outcome been analysed? | **No.** The 128px CPU pilot was stopped on 2026-10-10 with its control splits incomplete, so no threshold exists and no latency is defined. |

## Fastest path to an opinion

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # A1-A7, B6, C6-C7
cat CLAIMS.md                             # what is and is not claimed
cat DEVIATIONS.md                         # B2-B4
diff --strip-trailing-cr <(git show 033ee1a:PREREGISTRATION.md) PREREGISTRATION.md   # B1
```

---

Note: `FREEZE.json` field `registered_experiment_run` was false at freeze time
(correct); superseded by the run on 2026-10-10. The frozen file is not
modified.
