"""Pre-flight validation of the V-JEPA 2 adapter on CPU.

Uses a tiny, randomly initialised `VJEPA2Model` so the real architecture, the
real predictor and the real masking code all run, without downloading weights
or touching a GPU. The point is to catch a wrong token index or a broken
predictor contract here, for free, rather than after paying for inference.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import VJEPA2Config, VJEPA2Model      # noqa: E402

from lgw import config as C                             # noqa: E402
from lgw.datagen import make_clip                       # noqa: E402
from lgw.extract import run_extraction                  # noqa: E402
from lgw.vjepa2_adapter import VJEPA2Adapter            # noqa: E402

T_CTX = 8          # 4 tubelets, keeps the CPU test fast
RES = 32           # 2x2 = 4 tokens per tubelet -> 16 tokens total


def _tiny_model():
    # head_dim must be >= 12: the 3D RoPE uses d_dim = 2*((head_dim//3)//2) per
    # axis and degenerates when that is < 4. 96/4 = 24 is the smallest tidy size.
    cfg = VJEPA2Config(
        patch_size=16, crop_size=RES, frames_per_clip=T_CTX, tubelet_size=2,
        hidden_size=96, num_attention_heads=4, num_hidden_layers=2, mlp_ratio=2.0,
        pred_hidden_size=96, pred_num_attention_heads=4, pred_num_hidden_layers=2,
        pred_num_mask_tokens=4,
    )
    torch.manual_seed(0)
    return VJEPA2Model(cfg).eval()


def _adapter():
    return VJEPA2Adapter(_tiny_model(), device="cpu", dtype=torch.float32,
                         t_ctx=T_CTX, res=RES, name="tiny")


def test_token_index_block_maps_to_tubelet():
    """The t-major layout assumption: tubelet j owns indices [j*hw, (j+1)*hw).

    Tested on the patch embedding alone, which is a strided Conv3d and therefore
    spatially/temporally local, so the claim is checkable exactly.
    """
    model = _tiny_model()
    emb = model.encoder.embeddings.patch_embeddings
    hw = (RES // 16) ** 2
    n_tub = T_CTX // 2
    base = torch.zeros(1, T_CTX, 3, RES, RES)
    e0 = emb(base.permute(0, 2, 1, 3, 4))
    assert e0.shape[1] == n_tub * hw

    for j in range(n_tub):
        x = base.clone()
        x[:, 2 * j:2 * j + 2] = 1.0                     # light up only tubelet j
        e = emb(x.permute(0, 2, 1, 3, 4))
        changed = (e - e0).abs().amax(dim=-1)[0] > 1e-6
        expected = torch.zeros(n_tub * hw, dtype=torch.bool)
        expected[j * hw:(j + 1) * hw] = True
        assert torch.equal(changed, expected), (j, changed.tolist())


def test_adapter_shapes_and_loss_window():
    ad = _adapter()
    clips = np.stack([make_clip(i, "gravity_flip")[0] for i in range(2)])
    Z, L0 = ad.process(clips)
    assert Z.shape == (2, C.N_FRAMES, ad.dim)
    assert np.isfinite(Z).all()
    # L0 is only needed inside the evaluation window
    assert np.all(np.isnan(L0[:, :C.EVAL_START]))
    assert np.isfinite(L0[:, C.EVAL_START:]).all()
    assert (L0[:, C.EVAL_START:] > 0).all()


def test_adapter_is_causal_against_real_architecture():
    """Perturbing frames > t must leave z_t and L0_t untouched."""
    ad = _adapter()
    clip, _, _ = make_clip(5, "occlusion")
    cut = 20
    pert = clip.copy()
    pert[cut + 1:] = np.asarray(255, np.uint8) - pert[cut + 1:]
    Za, La = ad.process(clip[None])
    Zb, Lb = ad.process(pert[None])
    np.testing.assert_allclose(Za[0, : cut + 1], Zb[0, : cut + 1], atol=1e-6)
    np.testing.assert_allclose(La[0, C.EVAL_START:cut + 1],
                               Lb[0, C.EVAL_START:cut + 1], rtol=1e-5)


def test_l0_prediction_cannot_see_the_target_frames():
    """The L0 prediction at frame t must not move when frames t-1 and t change.

    Only the *target* may depend on them; if the prediction moved too, the loss
    would be partly self-fulfilling and the baseline unfairly strong.
    """
    from lgw.extract import context_window

    ad = _adapter()
    clip, _, _ = make_clip(6, None)
    t = 22
    pred_a = ad._predict_final_tubelet(np.stack([context_window(clip, t, T_CTX, 2)]))
    mod = clip.copy()
    mod[t - 1:t + 1] = 0                      # change only frames t-1 and t
    pred_b = ad._predict_final_tubelet(np.stack([context_window(mod, t, T_CTX, 2)]))
    np.testing.assert_allclose(pred_a.numpy(), pred_b.numpy(), atol=1e-6)

    # sanity: the prediction *does* respond to a change at f_{t-2}
    mod2 = clip.copy()
    mod2[t - 2] = 0
    pred_c = ad._predict_final_tubelet(np.stack([context_window(mod2, t, T_CTX, 2)]))
    assert np.abs(pred_a.numpy() - pred_c.numpy()).max() > 1e-5


def test_full_harness_runs_with_real_architecture():
    ad = _adapter()
    clips = np.stack([make_clip(i, "frame_drop")[0] for i in range(2)])
    out = run_extraction(clips, ad, batch_size=2, progress=False)
    for k in ("Z", "loss_L0", "probe_A", "probe_B", "probe_C"):
        assert k in out
    for k in ("probe_A", "probe_B", "probe_C"):
        assert np.isfinite(out[k][:, C.EVAL_START:]).all(), k
