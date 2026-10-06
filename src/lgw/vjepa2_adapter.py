"""Frozen V-JEPA 2 wrapped as an online world-model adapter (prereg Sec. 2 and 4).

Token layout
------------
`VJEPA2PatchEmbeddings3D` is a `Conv3d` with stride `(tubelet, patch, patch)`
followed by `.flatten(2).transpose(1, 2)`, so the token sequence is t-major:
token index `i` corresponds to tubelet `i // (H' * W')`. Tubelet `j` therefore
owns the contiguous index block `[j * H'W', (j+1) * H'W')`. Everything below
depends on that fact, and `tests/test_vjepa2_adapter.py` asserts it against a
real (randomly initialised) `VJEPA2Model`.

Per frame t, at most three forwards are run:
  1. encoder on the causal window `W_t`           -> z_t and the L0 target
  2. encoder on the context window `W'_t`          -> context tokens (<= t-2)
  3. predictor(context tokens -> target positions) -> L0 prediction
Steps 2 and 3 are skipped for `t < EVAL_START`, because L0 is never scored
outside the evaluation window. This changes no reported quantity and removes
half of the context passes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .extract import causal_window, context_window

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VJEPA2Adapter:
    """Online probing wrapper around a frozen `VJEPA2Model`."""

    def __init__(self, model, device="cuda", dtype=torch.float32,
                 t_ctx: int = C.T_CTX, res: int = C.MODEL_RES,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 loss_from_frame: int = C.EVAL_START, name: str = "vjepa2"):
        self.model = model.eval().to(device=device, dtype=dtype)
        for p in self.model.parameters():          # frozen: belt and braces
            p.requires_grad_(False)

        cfg = model.config
        self.patch = int(cfg.patch_size)
        self.tub = int(cfg.tubelet_size)
        self.dim = int(cfg.hidden_size)
        self.device, self.dtype = device, dtype
        self.res, self.t_ctx = res, t_ctx
        self.name = name
        self.loss_from_frame = loss_from_frame

        if res % self.patch or t_ctx % self.tub:
            raise ValueError("resolution/context length must divide patch/tubelet")
        self.hw = (res // self.patch) ** 2          # tokens per tubelet
        self.n_tub = t_ctx // self.tub
        n_tok = self.n_tub * self.hw
        # final tubelet = frames (t-1, t); everything before it is the context
        self.tgt_lo, self.tgt_hi = n_tok - self.hw, n_tok
        self._idx_ctx = torch.arange(0, self.tgt_lo, device=device).unsqueeze(0)
        self._idx_tgt = torch.arange(self.tgt_lo, self.tgt_hi, device=device).unsqueeze(0)

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

    # ---- forwards ----
    def _encode(self, windows: np.ndarray) -> torch.Tensor:
        out = self.model(pixel_values_videos=self._prep(windows), skip_predictor=True)
        return out.last_hidden_state

    def _predict_final_tubelet(self, ctx_windows: np.ndarray) -> torch.Tensor:
        B = ctx_windows.shape[0]
        out = self.model(
            pixel_values_videos=self._prep(ctx_windows),
            context_mask=[self._idx_ctx.expand(B, -1)],
            target_mask=[self._idx_tgt.expand(B, -1)],
        )
        return out.predictor_output.last_hidden_state

    # ---- public API ----
    @torch.inference_mode()
    def process(self, clips: np.ndarray):
        B, T = clips.shape[0], clips.shape[1]
        Z = np.empty((B, T, self.dim), dtype=np.float32)
        L0 = np.full((B, T), np.nan, dtype=np.float64)

        for t in range(T):
            full = np.stack([causal_window(clips[b], t, self.t_ctx) for b in range(B)])
            enc_full = self._encode(full)
            target = enc_full[:, self.tgt_lo:self.tgt_hi]      # true tubelet-7 tokens
            Z[:, t] = target.mean(dim=1).float().cpu().numpy()

            if t >= self.loss_from_frame:
                ctx = np.stack([context_window(clips[b], t, self.t_ctx, self.tub)
                                for b in range(B)])
                pred = self._predict_final_tubelet(ctx)
                mse = ((pred.float() - target.float()) ** 2).mean(dim=(1, 2))
                L0[:, t] = mse.cpu().numpy()

        return Z, L0


def load_frozen_vjepa2(model_id: str = C.MODEL_ID, device="cuda",
                       dtype=torch.float32):
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(model_id, dtype=dtype)
    return VJEPA2Adapter(model, device=device, dtype=dtype, name=model_id)
