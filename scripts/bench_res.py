"""Time the real V-JEPA 2 encoder at several input sizes, to size a CPU pilot.

The frozen protocol runs at 256 px / 16 context frames. That costs ~9.7 s per
forward on this machine, i.e. ~159 h for the full design - not runnable here.
This measures the cheaper operating points so a pilot can be sized honestly
instead of guessed at.
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

# (resolution, context frames, batch)
POINTS = [(128, 16, 1), (128, 16, 4), (96, 16, 4), (128, 8, 4), (64, 16, 4)]


def main() -> None:
    from transformers import VJEPA2Model

    torch.set_num_threads(os.cpu_count() or 8)
    model = VJEPA2Model.from_pretrained(C.MODEL_ID, dtype=torch.float32,
                                        attn_implementation="sdpa")
    clip, _, _ = make_clip(0, "gravity_flip")

    print("{:>5} {:>4} {:>3} {:>7} {:>9} {:>9} {:>11}".format(
        "res", "tctx", "B", "tokens", "s/fwd", "s/clip", "h/240clips"))
    for res, tctx, B in POINTS:
        try:
            ad = VJEPA2Adapter(model, device="cpu", dtype=torch.float32,
                               t_ctx=tctx, res=res, name="bench")
            w = np.stack([clip[:tctx]] * B)
            with torch.inference_mode():
                ad._encode(w)
                t0 = time.time()
                for _ in range(2):
                    ad._encode(w)
                per_fwd_batch = (time.time() - t0) / 2
            per_fwd_clip = per_fwd_batch / B
            # 30 full windows + 15 context windows + 15 predictor passes
            per_clip = 45 * per_fwd_clip * 1.12
            print("{:>5} {:>4} {:>3} {:>7} {:>9.3f} {:>9.1f} {:>11.1f}".format(
                res, tctx, B, ad.n_tub * ad.hw, per_fwd_clip, per_clip,
                240 * per_clip / 3600))
        except Exception as e:
            print("{:>5} {:>4} {:>3}  FAILED: {}".format(res, tctx, B,
                                                         str(e)[:60]))


if __name__ == "__main__":
    main()
