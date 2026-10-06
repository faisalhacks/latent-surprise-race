"""A real V-JEPA 2 PILOT on CPU, at reduced spatial resolution.

This is NOT the pre-registered experiment. The frozen protocol runs at 256 px,
which costs ~159 h on this machine. This runs at 128 px, where the 16-frame
context, the 2-frame tubelet, every probe definition and the whole L0
construction are untouched - only the patch grid shrinks from 16x16 to 8x8.
96 px was rejected because the disc would then be smaller than one patch.

Purpose: answer the two questions that decide whether the paid run is worth
running, and in what form.
  1. Does `loss_L0` fire above chance at all? Its dynamic range on 2 clips was
     +/-2%, which would make it a degenerate baseline.
  2. Do probes A and C have between-clip offsets that swamp their within-clip
     movement, making pooled thresholds behave as per-clip constants?

Checkpointed per chunk, so it can be interrupted and resumed.

Usage:  python scripts/run_pilot_cpu.py [n_per_condition] [n_calib] [n_eval]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgw import config as C                      # noqa: E402
from lgw.analysis import analyse                 # noqa: E402
from lgw.datagen import make_clip                # noqa: E402
from lgw.extract import run_extraction           # noqa: E402
from lgw.vjepa2_adapter import VJEPA2Adapter     # noqa: E402

PILOT_RES = 128
CHUNK = 4
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "pilot_cpu_128")
KEYS = ("Z", "loss_L0", "probe_A", "probe_B", "probe_C")


def plan(n_per_cond: int, n_calib: int, n_eval: int):
    specs = []
    for ci, cond in enumerate(C.CONDITIONS):
        base = ci * C.N_PER_CONDITION
        for j in range(n_per_cond):
            specs.append(("anomaly", base + j, cond))
    for j in range(n_calib):
        specs.append(("control_calib", 1_000_000 + j, None))
    for j in range(n_eval):
        specs.append(("control_eval", 2_000_000 + j, None))
    jobs, by_split = [], {}
    for split, i, cond in specs:
        by_split.setdefault(split, []).append((i, cond))
    for split, items in by_split.items():
        for k in range(0, len(items), CHUNK):
            jobs.append((split, items[k:k + CHUNK], len(jobs)))
    return jobs


def main() -> None:
    from transformers import VJEPA2Model

    a = sys.argv[1:]
    n_per_cond = int(a[0]) if len(a) > 0 else 40
    n_calib = int(a[1]) if len(a) > 1 else 70
    n_eval = int(a[2]) if len(a) > 2 else 50

    torch.set_num_threads(os.cpu_count() or 8)
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    jobs = plan(n_per_cond, n_calib, n_eval)
    total = sum(len(s) for _, s, _ in jobs)
    print("PILOT res={} t_ctx={} clips={} chunks={}".format(
        PILOT_RES, C.T_CTX, total, len(jobs)), flush=True)

    todo = [j for j in jobs
            if not os.path.exists(os.path.join(
                OUT, "chunks", "{}_{:04d}.npz".format(j[0], j[2])))]
    print("{} chunks already done, {} to go".format(
        len(jobs) - len(todo), len(todo)), flush=True)

    adapter = None
    t_start = time.time()
    for n_done, (split, specs, cid) in enumerate(todo):
        if adapter is None:
            model = VJEPA2Model.from_pretrained(C.MODEL_ID, dtype=torch.float32,
                                                attn_implementation="sdpa")
            adapter = VJEPA2Adapter(model, device="cpu", dtype=torch.float32,
                                    t_ctx=C.T_CTX, res=PILOT_RES,
                                    name=C.MODEL_ID + "@128px")
            print("model ready, D={}, tokens/window={}".format(
                adapter.dim, adapter.n_tub * adapter.hw), flush=True)

        clips = np.stack([make_clip(i, c)[0] for i, c in specs])
        out = run_extraction(clips, adapter, batch_size=len(specs), progress=False)
        np.savez_compressed(
            os.path.join(OUT, "chunks", "{}_{:04d}.npz".format(split, cid)),
            condition=np.array([c if c is not None else "control" for _, c in specs]),
            clip_id=np.array([i for i, _ in specs], dtype=np.int64),
            **{k: out[k] for k in KEYS})
        el = time.time() - t_start
        rate = el / (n_done + 1)
        print("[{:3d}/{:3d}] {:14s} {:.0f}s elapsed, eta {:.1f}h".format(
            n_done + 1, len(todo), split, el,
            rate * (len(todo) - n_done - 1) / 3600), flush=True)

    # ---- merge and analyse ----
    import glob
    scores = {}
    for split in ("anomaly", "control_calib", "control_eval"):
        files = sorted(glob.glob(os.path.join(OUT, "chunks", split + "_*.npz")))
        parts = [np.load(f, allow_pickle=False) for f in files]
        merged = {k: np.concatenate([p[k] for p in parts], axis=0)
                  for k in KEYS + ("condition", "clip_id")}
        order = np.argsort(merged["clip_id"], kind="stable")
        scores[split] = {k: v[order] for k, v in merged.items()}
        print("{:15s} {} clips".format(split, scores[split]["Z"].shape[0]))

    r = analyse(scores, os.path.join(OUT, "analysis"),
                model_name=C.MODEL_ID + " @128px (PILOT)",
                note=("PILOT, NOT the pre-registered experiment: run at 128 px "
                      "instead of the frozen 256 px, on CPU, at reduced n. "
                      "Context length, tubelet, probe definitions, L0 "
                      "construction, thresholds and metrics are unchanged. "
                      "Use it to size and sanity-check the paid run, not as "
                      "the result."))
    g = r["gate"]
    print("\n=============== PILOT gate ===============")
    print("verdict:                   {}".format(g["verdict"]))
    print("early warners (primary):   {}".format(g["early_warners"] or "none"))
    print("median degenerate:         {}".format(g["primary_median_degenerate"]))
    print("baselines at chance:       {}".format(
        g["baselines_firing_at_chance"] or "none"))
    print("verdict vacuous:           {}".format(g["verdict_is_vacuous"]))
    print("RMTA warners:              {}".format(
        g["secondary_rmta_early_warners"] or "none"))
    print("==========================================")
    print("wrote", os.path.abspath(os.path.join(OUT, "analysis")))


if __name__ == "__main__":
    main()
