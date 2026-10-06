"""Harness validation on CPU with the surrogate model.

Runs the complete Phase 2 -> Phase 3 path end-to-end so that the plumbing, the
probes, the thresholds, the survival statistics and the gate are all exercised
before any GPU budget is spent.

The numbers it produces are NOT a scientific result: the surrogate is a fixed
random projection, not V-JEPA 2. It exists to prove the harness works.

Usage:  python scripts/run_local_validation.py [n_per_condition]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgw import config as C                      # noqa: E402
from lgw.analysis import analyse                 # noqa: E402
from lgw.datagen import make_clip                # noqa: E402
from lgw.extract import run_extraction           # noqa: E402
from lgw.surrogate import SurrogateAdapter       # noqa: E402


def build(n_per_cond: int, n_calib: int, n_eval: int):
    clips, conds = [], []
    idx = 0
    for cond in C.CONDITIONS:
        for _ in range(n_per_cond):
            f, _, _ = make_clip(idx, cond)
            clips.append(f)
            conds.append(cond)
            idx += 1
    anomaly = (np.stack(clips), np.array(conds))

    def controls(base, n):
        return np.stack([make_clip(base + j, None)[0] for j in range(n)])

    return anomaly, controls(1_000_000, n_calib), controls(2_000_000, n_eval)


def main() -> None:
    n_per_cond = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    t0 = time.time()
    (ax, aconds), cal, ev = build(n_per_cond, C.N_CONTROL_CALIB, C.N_CONTROL_EVAL)
    print("clips: anomaly={} calib={} eval={}  ({:.1f}s)".format(
        ax.shape[0], cal.shape[0], ev.shape[0], time.time() - t0))

    adapter = SurrogateAdapter()
    scores = {}
    for name, X in (("anomaly", ax), ("control_calib", cal), ("control_eval", ev)):
        t1 = time.time()
        out = run_extraction(X, adapter, batch_size=64, progress=False)
        scores[name] = out
        print("  extracted {:14s} {} in {:.1f}s".format(name, out["Z"].shape, time.time() - t1))
    scores["anomaly"]["condition"] = aconds

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "surrogate_validation")
    r = analyse(scores, out_dir, model_name="surrogate (harness validation only)",
                note="NOT a scientific result. Fixed random projection, not V-JEPA 2.")
    print("\n--- gate: {} | early warners: {} ---".format(
        r["gate"]["verdict"], r["gate"]["early_warners"] or "none"))
    print("wrote {}".format(os.path.abspath(out_dir)))
    print("total {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
