"""VideoMAE as a second architecture (protocol Phase 4.1).

Why this model specifically
---------------------------
V-JEPA 2 is a joint-embedding predictive architecture: it predicts
*representations* of masked regions. VideoMAE is a masked autoencoder: it
reconstructs *pixels*. They share a tubelet size, a patch size and a ViT
backbone, so the adapter maps cleanly, but their training objectives sit on
opposite sides of the pixel/latent divide. That is exactly the contrast the
protocol asks for: is any early-warning effect a property of world models in
general, or an artefact of the joint-embedding predictive objective?

The leakage trap here is different from V-JEPA 2's
--------------------------------------------------
`VideoMAEForPreTraining` builds its regression targets by patchifying **the same
`pixel_values` it was handed**. Feeding it the context window would therefore
score the prediction against the *overwritten* pixels rather than the true
frames `t-1, t` -- a loss that is trivially near zero and a baseline that looks
far stronger than it is. So the prediction comes from the context window while
the target is patchified from the full window, and the two are combined by the
shared `l0`. We replicate the model's own patchify-and-normalise so the target
is the quantity VideoMAE was actually trained to produce.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .adapters import BaseWindowAdapter, register

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL = "MCG-NJU/videomae-base"


def patchify(pixel_values: torch.Tensor, tubelet: int, patch: int,
             norm_pix_loss: bool) -> torch.Tensor:
    """(B, T, C, H, W) -> (B, n_patches, tubelet*patch*patch*C).

    Mirrors `VideoMAEForPreTraining.forward`'s target construction, including
    the per-patch normalisation, so the L0 target is the model's own objective
    rather than something merely similar to it.
    """
    B, T, Ch, H, W = pixel_values.shape
    x = pixel_values.view(B, T // tubelet, tubelet, Ch, H // patch, patch,
                          W // patch, patch)
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    n_patches = (T // tubelet) * (H // patch) * (W // patch)
    if norm_pix_loss:
        x = x.view(B, n_patches, tubelet * patch * patch, Ch)
        x = (x - x.mean(dim=-2, keepdim=True)) / (
            x.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    return x.view(B, n_patches, tubelet * patch * patch * Ch)


@register("videomae")
class VideoMAEAdapter(BaseWindowAdapter):
    """Online probing wrapper around a frozen `VideoMAEForPreTraining`."""

    def __init__(self, model, device="cuda", dtype=torch.float32,
                 res: int | None = None, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 loss_from_frame: int = C.EVAL_START, name: str = "videomae"):
        cfg = model.config
        t_ctx = int(cfg.num_frames)
        super().__init__(dim=int(cfg.hidden_size), name=name, t_ctx=t_ctx,
                         tubelet=int(cfg.tubelet_size),
                         loss_from_frame=loss_from_frame)
        self.model = model.eval().to(device=device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.patch = int(cfg.patch_size)
        self.res = int(res if res is not None else cfg.image_size)
        self.device, self.dtype = device, dtype
        self.norm_pix_loss = bool(getattr(cfg, "norm_pix_loss", True))

        if self.res % self.patch or t_ctx % self.tubelet:
            raise ValueError("resolution/frames must divide patch/tubelet")
        self.hw = (self.res // self.patch) ** 2
        self.n_tub = t_ctx // self.tubelet
        n_tok = self.n_tub * self.hw
        self.tgt_lo, self.tgt_hi = n_tok - self.hw, n_tok
        self.n_tokens = n_tok

        self._mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(std, device=device, dtype=dtype).view(1, 1, 3, 1, 1)

    def _prep(self, windows: np.ndarray) -> torch.Tensor:
        B, T = windows.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(windows)).to(self.device)
        x = x.to(self.dtype).div_(255.0).reshape(B * T, 1, *windows.shape[2:])
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear",
                          align_corners=False)
        x = x.reshape(B, T, 1, self.res, self.res).expand(B, T, 3, self.res, self.res)
        return (x - self._mean) / self._std

    def _final_tubelet_mask(self, B: int) -> torch.Tensor:
        m = torch.zeros((B, self.n_tokens), dtype=torch.bool, device=self.device)
        m[:, self.tgt_lo:self.tgt_hi] = True
        return m

    # ---- adapter contract ----
    @torch.inference_mode()
    def encode_window(self, windows: np.ndarray):
        px = self._prep(windows)
        enc = self.model.videomae(pixel_values=px).last_hidden_state
        z = enc[:, self.tgt_lo:self.tgt_hi].mean(dim=1).float().cpu().numpy()
        target = patchify(px, self.tubelet, self.patch, self.norm_pix_loss)
        target = target[:, self.tgt_lo:self.tgt_hi]
        return z, target.float().cpu().numpy()

    @torch.inference_mode()
    def predict_final_step(self, ctx_windows: np.ndarray):
        px = self._prep(ctx_windows)
        mask = self._final_tubelet_mask(px.shape[0])
        out = self.model(pixel_values=px, bool_masked_pos=mask)
        return out.logits.float().cpu().numpy()

    @torch.inference_mode()
    def process(self, clips: np.ndarray):
        return super().process(clips)


def load_frozen_videomae(model_id: str = DEFAULT_MODEL, device="cuda",
                         dtype=torch.float32, **kw):
    from transformers import VideoMAEForPreTraining
    model = VideoMAEForPreTraining.from_pretrained(model_id, dtype=dtype)
    return VideoMAEAdapter(model, device=device, dtype=dtype, name=model_id, **kw)
