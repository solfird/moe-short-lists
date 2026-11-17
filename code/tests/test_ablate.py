"""ablate.py: scale Gemma 4's always-on dense path inside the unchanged MoE model (plans/2026-10-31-paper.md, H16)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.gemma4_text import Model, ModelArgs

import ablate


def tiny() -> Model:
    mx.random.seed(0)
    args = ModelArgs(
        hidden_size=32,
        num_hidden_layers=3,
        intermediate_size=64,
        num_attention_heads=2,
        head_dim=16,
        global_head_dim=16,
        vocab_size=100,
        vocab_size_per_layer_input=100,
        num_kv_shared_layers=0,
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=16,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


IDS = mx.array([[1, 2, 3, 4, 5]])


def test_alpha_one_leaves_the_logits_identical():
    model = tiny()
    before = model(IDS)
    ablate.scale_always_on(model, 1.0)
    assert mx.array_equal(model(IDS), before)


def test_alpha_scales_the_output_of_the_dense_path_in_every_moe_layer():
    model = tiny()
    assert ablate.scale_always_on(model, 0.5) == 3
    layer = model.model.layers[0]
    x = mx.random.normal((1, 5, 32))
    assert mx.allclose(layer.post_feedforward_layernorm_1(x), 0.5 * layer.post_feedforward_layernorm_1.inner(x))


def test_alpha_zero_matches_a_model_whose_dense_path_returns_zero():
    a, b = tiny(), tiny()
    ablate.scale_always_on(a, 0.0)
    for layer in b.model.layers:
        layer.post_feedforward_layernorm_1 = lambda x: mx.zeros_like(x)  # noqa: B023
    assert mx.array_equal(a(IDS), b(IDS))


def test_scaling_changes_the_logits():
    model = tiny()
    before = model(IDS)
    ablate.scale_always_on(model, 0.5)
    assert not mx.array_equal(model(IDS), before)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_out_of_range_is_refused(alpha):
    with pytest.raises(ValueError, match="alpha"):
        ablate.scale_always_on(tiny(), alpha)


def test_a_model_without_the_gemma_path_is_refused():
    class Other(nn.Module):
        layers: list = []

    with pytest.raises(ValueError, match="Gemma"):
        ablate.scale_always_on(Other(), 0.5)


def test_scaling_twice_is_refused():
    model = tiny()
    ablate.scale_always_on(model, 0.5)
    with pytest.raises(ValueError, match="already"):
        ablate.scale_always_on(model, 0.5)


def test_tag_names_the_arm():
    assert ablate.tag(0.5) == "gemma_a0.50"
    assert ablate.tag(0.0) == "gemma_a0.00"


def test_kept_difference_is_paired_and_points_at_the_loss():
    """Intact keeps 1.0, ablated 0.5, on the same items; the CI is entirely above zero."""
    keys = [("qa", i) for i in range(50)]
    base_i = {k: 1.0 for k in keys}
    mask_i = {k: 1.0 for k in keys}
    base_a = {k: 0.8 for k in keys}
    mask_a = {k: 0.4 for k in keys}
    d = ablate.kept_difference(mask_i, base_i, mask_a, base_a)
    assert d["mean"] == pytest.approx(0.5)
    assert d["lo"] == pytest.approx(0.5) and d["hi"] == pytest.approx(0.5)


def test_interpretable_needs_half_of_the_intact_unmasked_score():
    assert ablate.interpretable(0.5, 1.0)
    assert not ablate.interpretable(0.49, 1.0)


@pytest.mark.slow
def test_real_gemma_alpha_one_is_identical_and_alpha_half_is_not():
    from mlx_lm import load

    from models import MODELS

    model, _ = load(MODELS["gemma"])
    ids = mx.array([[2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107]])
    before = model(ids)
    mx.eval(before)
    assert ablate.scale_always_on(model, 1.0) == 30
    assert mx.array_equal(model(ids), before)
    assert ablate.scale_always_on(model, 0.5) == 30
    assert not mx.array_equal(model(ids), before)


def test_kept_difference_is_nan_when_the_arm_scores_zero_on_every_item():
    keys = range(5)
    one = dict.fromkeys(keys, 1.0)
    zero = dict.fromkeys(keys, 0.0)
    d = ablate.kept_difference(one, one, zero, zero)
    assert all(np.isnan(d[x]) for x in ("mean", "lo", "hi"))
