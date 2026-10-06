"""Measure real V-JEPA 2 ViT-L throughput on this CPU.

Used to decide whether a local fallback for Phase 2 is feasible at all, rather
than guessing. Downloads ~1.2 GB of weights on first run.
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
from lgw.vjepa2_adapter import VJEPA2Adapter     # noqa: E402


def main() -> None:
    from transformers import VJEPA2Model

    torch.set_num_threads(os.cpu_count() or 8)
    t0 = time.time()
    model = VJEPA2Model.from_pretrained(C.MODEL_ID, dtype=torch.float32,
                                        attn_implementation="sdpa")
    print("loaded {} in {:.1f}s".format(C.MODEL_ID, time.time() - t0))
    n_par = sum(p.numel() for p in model.parameters())
    print("parameters: {:.1f}M".format(n_par / 1e6))

    ad = VJEPA2Adapter(model, device="cpu", dtype=torch.float32, name="cpu-bench")
    print("tokens per window: {} ({} tubelets x {} spatial)".format(
        ad.n_tub * ad.hw, ad.n_tub, ad.hw))

    clip, _, _ = make_clip(0, "gravity_flip")
    w = np.stack([clip[:C.T_CTX]])

    with torch.inference_mode():
        ad._encode(w)                                    # warm up
        t1 = time.time()
        reps = 3
        for _ in range(reps):
            ad._encode(w)
        per_fwd = (time.time() - t1) / reps
    print("\nencoder forward (batch 1, 16 frames, 256px): {:.2f}s".format(per_fwd))

    # 30 encoder passes + 15 context passes + 15 predictor passes per clip
    per_clip = 45 * per_fwd * 1.1
    print("estimated per clip:      {:.1f}s".format(per_clip))
    print("estimated 1200 clips:    {:.1f} hours".format(1200 * per_clip / 3600))
    print("estimated 100 clips:     {:.1f} hours".format(100 * per_clip / 3600))


if __name__ == "__main__":
    main()
