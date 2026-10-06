"""End-to-end check of the real pretrained V-JEPA 2 on CPU, a couple of clips.

The tiny-model tests prove the plumbing against the real *architecture* with
random weights. This proves the same path against the real *weights*: that the
predictor produces a sane loss, that the latents are well conditioned, and that
the probes return finite, non-degenerate values. Slow (minutes per clip) and
far too small to mean anything statistically - it is a correctness check.

Usage:  python scripts/smoke_real_cpu.py [n_clips]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgw import config as C                      # noqa: E402
from lgw.datagen import make_clip                # noqa: E402
from lgw.extract import run_extraction           # noqa: E402
from lgw.vjepa2_adapter import VJEPA2Adapter     # noqa: E402


def main() -> None:
    from transformers import VJEPA2Model

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    torch.set_num_threads(os.cpu_count() or 8)

    specs = [(0, "gravity_flip"), (1_000_000, None)][:n]
    clips = np.stack([make_clip(i, c)[0] for i, c in specs])

    t0 = time.time()
    model = VJEPA2Model.from_pretrained(C.MODEL_ID, dtype=torch.float32,
                                        attn_implementation="sdpa")
    ad = VJEPA2Adapter(model, device="cpu", dtype=torch.float32, name=C.MODEL_ID)
    print("model ready in {:.0f}s; D={} tokens/window={}".format(
        time.time() - t0, ad.dim, ad.n_tub * ad.hw))

    t1 = time.time()
    out = run_extraction(clips, ad, batch_size=n, progress=False)
    print("extracted {} clips in {:.0f}s\n".format(n, time.time() - t1))

    Z = out["Z"]
    print("latent stats: norm mean={:.2f} std={:.2f} | finite={}".format(
        float(np.linalg.norm(Z, axis=2).mean()),
        float(np.linalg.norm(Z, axis=2).std()),
        bool(np.isfinite(Z).all())))
    # how much of z changes frame to frame; probe A lives on the second difference
    d1 = np.linalg.norm(np.diff(Z, axis=1), axis=2)
    print("||z_t - z_t-1|| relative to ||z||: {:.4f}".format(
        float(d1.mean() / np.linalg.norm(Z, axis=2).mean())))

    ev = slice(C.EVAL_START, C.EVAL_END)
    for k in ("loss_L0", "probe_A", "probe_B", "probe_C"):
        v = out[k][:, ev]
        print("{:9s} finite={} min={:.4g} median={:.4g} max={:.4g}".format(
            k, bool(np.isfinite(v).all()), float(np.nanmin(v)),
            float(np.nanmedian(v)), float(np.nanmax(v))))

    for j, (i, cond) in enumerate(specs):
        tag = cond or "control"
        print("\nclip {} ({}):".format(i, tag))
        for k in ("loss_L0", "probe_A", "probe_B", "probe_C"):
            s = out[k][j, C.EVAL_START:C.EVAL_START + 8]
            print("  {:9s} t=15..22: {}".format(
                k, " ".join("{:8.4g}".format(x) for x in s)))

    os.makedirs("results/real_cpu_smoke", exist_ok=True)
    np.savez_compressed("results/real_cpu_smoke/out.npz",
                        **{k: out[k] for k in
                           ("Z", "loss_L0", "probe_A", "probe_B", "probe_C")})
    print("\nwrote results/real_cpu_smoke/out.npz")


if __name__ == "__main__":
    main()
