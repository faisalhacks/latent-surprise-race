"""Phase 4.3 - probe what the latents encode in the lag window.

Runs against any completed run directory containing per-chunk npz files with
latents, plus that run's results.json for the alarm times.

    python scripts/run_mechanistic.py results/pilot_cpu_128

Contingent on the Phase 3 gate: if no probe alarms before the loss, there is no
lag window and the script says so rather than probing arbitrary frames.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgw import config as C                       # noqa: E402
from lgw import stats as S                        # noqa: E402
from lgw.analysis import add_l1_baseline          # noqa: E402
from lgw.datagen import make_clip                 # noqa: E402
from lgw.mechanistic import (analyse_lag_window,  # noqa: E402
                             format_report, lag_window_slices)

KEYS = ("Z", "loss_L0", "probe_A", "probe_B", "probe_C")


def load_split(run_dir: str, split: str) -> dict:
    files = sorted(glob.glob(os.path.join(run_dir, "chunks", split + "_*.npz")))
    if not files:
        raise FileNotFoundError("no chunks for " + split + " in " + run_dir)
    parts = [np.load(f, allow_pickle=False) for f in files]
    merged = {k: np.concatenate([p[k] for p in parts], axis=0)
              for k in KEYS + ("condition", "clip_id")}
    _, keep = np.unique(merged["clip_id"], return_index=True)
    merged = {k: v[np.sort(keep)] for k, v in merged.items()}
    order = np.argsort(merged["clip_id"], kind="stable")
    return {k: v[order] for k, v in merged.items()}


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_cpu_128"
    scores = {s: load_split(run_dir, s)
              for s in ("anomaly", "control_calib", "control_eval")}
    add_l1_baseline(scores)
    anom, calib = scores["anomaly"], scores["control_calib"]

    thresholds = {m: S.threshold(calib[m]) for m in C.MONITORS}
    dts, evs = {}, {}
    for m in C.MONITORS:
        dts[m], evs[m] = S.alarm_times(anom[m], thresholds[m])

    traj = np.stack([make_clip(int(i), c if c != "control" else None)[1]
                     for i, c in zip(anom["clip_id"], anom["condition"])])

    out, lines = {}, ["# Phase 4.3 - mechanistic analysis of the lag window", ""]
    lines.append("Run: `{}`  (n_anomaly={})".format(run_dir, anom["Z"].shape[0]))
    lines.append("")
    any_lag = False
    for probe in C.PROBES:
        for base in C.BASELINES:
            sl = lag_window_slices(dts[probe], evs[probe], dts[base], evs[base])
            n = sum(s is not None for s in sl)
            key = probe + "_vs_" + base
            if n == 0:
                lines.append("## `{}`\n\nNo clip has a lag window: this probe "
                             "never alarms strictly before this baseline. There "
                             "is nothing to probe, which is itself the "
                             "finding.\n".format(key))
                out[key] = {"n_clips_with_lag": 0}
                continue
            any_lag = True
            res = analyse_lag_window(anom["Z"], anom["condition"], sl, traj)
            out[key] = res
            lines.append("## `{}`\n".format(key))
            lines.append(format_report(res))

    if not any_lag:
        lines.append("\n**No probe alarmed before any baseline on this run, so "
                     "the mechanistic question does not arise.** Reporting that "
                     "plainly is the honest outcome; probing frames chosen some "
                     "other way would be answering a different question.\n")

    os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
    with open(os.path.join(run_dir, "analysis", "mechanistic.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    rep = os.path.join(run_dir, "analysis", "MECHANISTIC.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", os.path.abspath(rep))


if __name__ == "__main__":
    main()
