"""Frozen V-JEPA 2 wrapped as an online world-model adapter (prereg Sec. 2, 4).

Token layout
------------
`VJEPA2PatchEmbeddings3D` is a `Conv3d` with stride `(tubelet, patch, patch)`
followed by `.flatten(2).transpose(1, 2)`, so the token sequence is t-major:
token index `i` belongs to tubelet `i // (H' * W')`. Tubelet `j` therefore owns
the contiguous block `[j * H'W', (j+1) * H'W')`. Everything below depends on
that, and `tests/test_vjepa2_adapter.py` asserts it against the real
`Conv3d` patch embedder.

Why two encoder passes
----------------------
`VJEPA2Model.forward` runs **one** encoder over the whole video and then uses
token masks to pick context and target. Its own source says as much:

    # this is implemented here as in VJEPA training a separate encoder is
    # used for target

With a single pass the context tokens have attended to the very frames being
predicted, so the loss is partly self-fulfilling and the baseline is unfairly
strong - which would bias *against* the hypothesis under test. So the context
comes from a separate pass over a window whose final tubelet is overwritten
with a repeat of `f_{t-2}`, and only the target comes from the full window.
`test_l0_prediction_cannot_see_the_target_frames` pins this.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .adapters import BaseWindowAdapter, register

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@register("vjepa2")
class VJEPA2Adapter(BaseWindowAdapter):
    """Online probing wrapper around a frozen `VJEPA2Model`."""

    def __init__(self, model, device="cuda", dtype=torch.float32,
                 t_ctx: int = C.T_CTX, res: int = C.MODEL_RES,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 loss_from_frame: int = C.EVAL_START, name: str = "vjepa2"):
        cfg = model.config
        super().__init__(dim=int(cfg.hidden_size), name=name, t_ctx=t_ctx,
                         tubelet=int(cfg.tubelet_size),
                         loss_from_frame=loss_from_frame)
        self.model = model.eval().to(device=device, dtype=dtype)
        for p in self.model.parameters():          # frozen: belt and braces
            p.requires_grad_(False)

        self.patch = int(cfg.patch_size)
        self.device, self.dtype = device, dtype
        self.res = res

        if res % self.patch or t_ctx % self.tubelet:
            raise ValueError("resolution/context length must divide patch/tubelet")
        self.hw = (res // self.patch) ** 2          # tokens per tubelet
        self.n_tub = t_ctx // self.tubelet
        n_tok = self.n_tub * self.hw
        # final tubelet = frames (t-1, t); everything before it is the context
        self.tgt_lo, self.tgt_hi = n_tok - self.hw, n_tok
        self._idx_ctx = torch.arange(0, self.tgt_lo, device=device).unsqueeze(0)
        self._idx_tgt = torch.arange(self.tgt_lo, self.tgt_hi,
                                     device=device).unsqueeze(0)

        self._mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(std, device=device, dtype=dtype).view(1, 1, 3, 1, 1)

    # ---- preprocessing ----
    def _prep(self, windows: np.ndarray) -> torch.Tensor:
        """(B, T, H, W) uint8 -> (B, T, 3, res, res) normalised tensor."""
        B, T = windows.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(windows)).to(self.device)
        x = x.to(self.dtype).div_(255.0).reshape(B * T, 1, *windows.shape[2:])
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear",
                          align_corners=False)
        x = x.reshape(B, T, 1, self.res, self.res).expand(B, T, 3, self.res, self.res)
        return (x - self._mean) / self._std

    def _encode(self, windows: np.ndarray) -> torch.Tensor:
        out = self.model(pixel_values_videos=self._prep(windows),
                         skip_predictor=True)
        return out.last_hidden_state

    def _predict_final_tubelet(self, ctx_windows: np.ndarray) -> torch.Tensor:
        B = ctx_windows.shape[0]
        out = self.model(
            pixel_values_videos=self._prep(ctx_windows),
            context_mask=[self._idx_ctx.expand(B, -1)],
            target_mask=[self._idx_tgt.expand(B, -1)],
        )
        return out.predictor_output.last_hidden_state

    # ---- adapter contract ----
    @torch.inference_mode()
    def encode_window(self, windows: np.ndarray):
        enc = self._encode(windows)
        target = enc[:, self.tgt_lo:self.tgt_hi]          # true final-tubelet tokens
        z = target.mean(dim=1).float().cpu().numpy()
        return z, target.float().cpu().numpy()

    @torch.inference_mode()
    def predict_final_step(self, ctx_windows: np.ndarray):
        return self._predict_final_tubelet(ctx_windows).float().cpu().numpy()

    @torch.inference_mode()
    def process(self, clips: np.ndarray):
        return super().process(clips)


def load_frozen_vjepa2(model_id: str = C.MODEL_ID, device="cuda",
                       dtype=torch.float32, **kw):
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(model_id, dtype=dtype,
                                        attn_implementation="sdpa")
    return VJEPA2Adapter(model, device=device, dtype=dtype, name=model_id, **kw)
