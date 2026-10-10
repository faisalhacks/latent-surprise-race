"""Every registered adapter must satisfy the same online contract.

This is the reusability guarantee. A new architecture is only admissible if it
passes these checks, which is what lets Phase 4.1 compare architectures rather
than compare pipelines. Adding an adapter without adding a constructor here
fails `test_every_registered_adapter_is_covered`, by design.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lgw import config as C                                  # noqa: E402
from lgw.adapters import REGISTRY, available_adapters        # noqa: E402
from lgw.datagen import make_clip                            # noqa: E402
from lgw.extract import run_extraction                       # noqa: E402
from lgw.surrogate import SurrogateAdapter                   # noqa: E402
from lgw.videomae_adapter import VideoMAEAdapter             # noqa: E402
from lgw.vjepa2_adapter import VJEPA2Adapter                 # noqa: E402

TINY_T_CTX, TINY_RES = 8, 32


def _tiny_vjepa2():
    from transformers import VJEPA2Config, VJEPA2Model
    cfg = VJEPA2Config(
        patch_size=16, crop_size=TINY_RES, frames_per_clip=TINY_T_CTX,
        tubelet_size=2, hidden_size=96, num_attention_heads=4,
        num_hidden_layers=2, mlp_ratio=2.0, pred_hidden_size=96,
        pred_num_attention_heads=4, pred_num_hidden_layers=2,
        pred_num_mask_tokens=4)
    torch.manual_seed(0)
    return VJEPA2Adapter(VJEPA2Model(cfg), device="cpu", dtype=torch.float32,
                         t_ctx=TINY_T_CTX, res=TINY_RES, name="tiny")


def _tiny_videomae():
    from transformers import VideoMAEConfig, VideoMAEForPreTraining
    cfg = VideoMAEConfig(
        image_size=TINY_RES, patch_size=16, num_frames=TINY_T_CTX,
        tubelet_size=2, hidden_size=96, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=192, decoder_hidden_size=96,
        decoder_num_hidden_layers=2, decoder_num_attention_heads=4,
        decoder_intermediate_size=192)
    torch.manual_seed(0)
    return VideoMAEAdapter(VideoMAEForPreTraining(cfg), device="cpu",
                           dtype=torch.float32, name="tiny-videomae")


# Cheap constructors, one per registered adapter.
CONSTRUCTORS = {
    "surrogate": lambda: SurrogateAdapter(),
    "vjepa2": _tiny_vjepa2,
    "videomae": _tiny_videomae,
}


def test_every_registered_adapter_is_covered():
    missing = set(available_adapters()) - set(CONSTRUCTORS)
    assert not missing, (
        "registered adapters with no contract-test constructor: {}. "
        "Add one so the new model is held to the same online contract.".format(
            sorted(missing)))


@pytest.mark.parametrize("name", sorted(CONSTRUCTORS))
def test_shapes_and_loss_window(name):
    ad = CONSTRUCTORS[name]()
    clips = np.stack([make_clip(i, "gravity_flip")[0] for i in range(2)])
    Z, L0 = ad.process(clips)
    assert Z.shape == (2, C.N_FRAMES, ad.dim)
    assert np.isfinite(Z).all()
    assert np.all(np.isnan(L0[:, :C.EVAL_START])), "L0 scored before the window"
    assert np.isfinite(L0[:, C.EVAL_START:]).all()


@pytest.mark.parametrize("name", sorted(CONSTRUCTORS))
def test_future_perturbation_does_not_change_the_past(name):
    """The core guarantee: frame t cannot see frame t+1."""
    ad = CONSTRUCTORS[name]()
    clip, _, _ = make_clip(5, "occlusion")
    cut = 20
    pert = clip.copy()
    pert[cut + 1:] = np.asarray(255, np.uint8) - pert[cut + 1:]
    Za, La = ad.process(clip[None])
    Zb, Lb = ad.process(pert[None])
    np.testing.assert_allclose(Za[0, : cut + 1], Zb[0, : cut + 1], atol=1e-6)
    np.testing.assert_allclose(La[0, C.EVAL_START:cut + 1],
                               Lb[0, C.EVAL_START:cut + 1], rtol=1e-5)


@pytest.mark.parametrize("name", sorted(CONSTRUCTORS))
def test_l0_prediction_is_blind_to_the_predicted_frames(name):
    """Changing only frames t-1 and t must not move the prediction.

    Otherwise the prediction is partly self-fulfilling and the baseline is
    unfairly strong, which would bias against the hypothesis under test.
    """
    from lgw.extract import context_window

    ad = CONSTRUCTORS[name]()
    clip, _, _ = make_clip(6, None)
    t = 22
    a = ad.predict_final_step(np.stack([context_window(clip, t, ad.t_ctx, ad.tubelet)]))
    mod = clip.copy()
    mod[t - 1:t + 1] = 0
    b = ad.predict_final_step(np.stack([context_window(mod, t, ad.t_ctx, ad.tubelet)]))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)


@pytest.mark.parametrize("name", sorted(CONSTRUCTORS))
def test_runs_through_the_shared_harness(name):
    ad = CONSTRUCTORS[name]()
    clips = np.stack([make_clip(i, "frame_drop")[0] for i in range(2)])
    out = run_extraction(clips, ad, batch_size=2, progress=False)
    for k in ("probe_A", "probe_B", "probe_C"):
        assert np.isfinite(out[k][:, C.EVAL_START:]).all(), k


def test_registry_is_populated():
    assert "vjepa2" in REGISTRY and "surrogate" in REGISTRY
