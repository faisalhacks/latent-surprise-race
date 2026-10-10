# Reproduce

One command, no GPU, no downloads, ~3 minutes.

**bash / zsh / Git Bash:**

```bash
pip install -r requirements.txt && python -m pytest tests -q && python scripts/run_local_validation.py 40
```

**Windows PowerShell 5.1** — `&&` is a parser error there, so chain with `;`
and check `$?`:

```powershell
pip install -r requirements.txt; if ($?) { python -m pytest tests -q }; if ($?) { python scripts/run_local_validation.py 40 }
```

or, equivalently, as three lines:

```powershell
pip install -r requirements.txt
python -m pytest tests -q
python scripts\run_local_validation.py 40
```

(PowerShell 7+ supports `&&`, so the bash form works there unchanged.)

That runs the full test suite (causality, leakage, probe correctness, adapter
contract, paper generation) and then the entire pipeline end-to-end on CPU with
a stand-in model: clip generation, extraction, probes, thresholds, survival
analysis, decision gate, report and survival curves.

The stand-in model is a fixed random projection. **Its numbers are not a
scientific result** and the generated report says so on its first line. The
point is to prove the apparatus runs.

## Rebuild the paper

```bash
make paper        # figures + tables + PDF -> paper/main.pdf
```

With no registered run analysed, the PDF compiles with visible `PENDING`
markers rather than placeholder values.

## The registered experiment

**Not run.** It needs ~1 GPU-hour on a Modal workspace with billing enabled.
See `docs/PARKED_EXPERIMENTS.md` (entry P1) for the exact launch command, and
`CLOSEOUT.md` for why it is blocked.

## The exploratory pilot

```bash
python scripts/run_pilot_cpu.py 40 70 50        # 128px, CPU, checkpointed
python scripts/run_pilot_cpu.py --analyse-only  # analyse whatever is on disk
```

On Windows use backslashes (`python scripts\run_pilot_cpu.py 40 70 50`); the
forward-slash form also works from Git Bash.

Not the registered experiment: reduced resolution and scale. See
`exploratory_pilot_128px/README.md`.

## What you need

Python 3.11 or 3.12, ~2 GB disk for the test suite. `torch` and
`transformers==4.57.3` are required for the adapter tests (CPU build is fine).
No dataset download: clips are regenerated deterministically from
`config.SEED = 20261006`.
