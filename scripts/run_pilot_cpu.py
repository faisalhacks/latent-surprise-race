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
    """Work order: controls first, then anomalies round-robin by condition.

    Two properties matter for a run that may be interrupted:
      * thresholds come from control clips, so without them nothing can be
        analysed at all. Controls therefore go first, and any interruption
        after they finish still leaves a usable (smaller) experiment.
      * anomalies are interleaved across conditions, so a partial anomaly set
        is balanced rather than being all of condition one.
    """
    jobs = []

    def add(split, items):
        for k in range(0, len(items), CHUNK):
            jobs.append((split, items[k:k + CHUNK]))

    add("control_calib", [(1_000_000 + j, None) for j in range(n_calib)])
    add("control_eval", [(2_000_000 + j, None) for j in range(n_eval)])

    interleaved = []
    for j in range(n_per_cond):
        for ci, cond in enumerate(C.CONDITIONS):
            interleaved.append((ci * C.N_PER_CONDITION + j, cond))
    add("anomaly", interleaved)
    return jobs


def completed_clip_ids(split: str) -> set:
    """Clip ids already on disk for a split, whatever file they live in.

    Resume keys on content, not on filename, so the work order can change
    between runs without discarding finished clips.
    """
    import glob
    done = set()
    for f in glob.glob(os.path.join(OUT, "chunks", split + "_*.npz")):
        try:
            done.update(int(x) for x in np.load(f, allow_pickle=False)["clip_id"])
        except Exception:
            continue
    return done


def main() -> None:
    from transformers import VJEPA2Model

    a = sys.argv[1:]
    n_per_cond = int(a[0]) if len(a) > 0 else 40
    n_calib = int(a[1]) if len(a) > 1 else 70
    n_eval = int(a[2]) if len(a) > 2 else 50

    torch.set_num_threads(os.cpu_count() or 8)
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    jobs = plan(n_per_cond, n_calib, n_eval)
    total = sum(len(s) for _, s in jobs)
    print("PILOT res={} t_ctx={} clips={} chunks={}".format(
        PILOT_RES, C.T_CTX, total, len(jobs)), flush=True)

    done_by_split = {s: completed_clip_ids(s)
                     for s in ("anomaly", "control_calib", "control_eval")}
    todo = []
    for split, specs in jobs:
        rest = [(i, c) for i, c in specs if i not in done_by_split[split]]
        if rest:
            todo.append((split, rest))
    n_done_clips = sum(len(v) for v in done_by_split.values())
    print("{} clips already done, {} chunks to go".format(
        n_done_clips, len(todo)), flush=True)

    adapter = None
    t_start = time.time()
    for n_done, (split, specs) in enumerate(todo):
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
        # named by first clip id, so the filename is stable under reordering
        np.savez_compressed(
            os.path.join(OUT, "chunks", "{}_{:07d}.npz".format(split, specs[0][0])),
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
        # dedupe then sort, so a re-run that renamed files cannot double-count
        _, keep = np.unique(merged["clip_id"], return_index=True)
        scores[split] = {k: v[np.sort(keep)] for k, v in merged.items()}
        order = np.argsort(scores[split]["clip_id"], kind="stable")
        scores[split] = {k: v[order] for k, v in scores[split].items()}
        n = scores[split]["Z"].shape[0]
        if n == 0:
            raise SystemExit("split {} has no clips; nothing to analyse".format(split))
        print("{:15s} {} clips".format(split, n))
        if split == "anomaly":
            uniq, cnt = np.unique(scores[split]["condition"], return_counts=True)
            print("                 " + ", ".join(
                "{}={}".format(u, c) for u, c in zip(uniq, cnt)))

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
