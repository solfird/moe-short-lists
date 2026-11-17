"""Adapter tests on tiny random weights: no model download."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.nemotron_h import MoEGate, group_expert_select

import segroute
from adapters import GateTap, find_routers, n_experts

HIDDEN, EXPERTS, TOP_K = 16, 32, 4


def make_gate(seed=0, bias_scale=0.5):
    cfg = SimpleNamespace(
        num_experts_per_tok=TOP_K,
        norm_topk_prob=True,
        n_routed_experts=EXPERTS,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        hidden_size=HIDDEN,
    )
    gate = MoEGate(cfg)
    mx.random.seed(seed)
    gate.weight = mx.random.normal((EXPERTS, HIDDEN))
    gate.e_score_correction_bias = mx.random.uniform(-bias_scale, bias_scale, (EXPERTS,))
    return gate


def tokens(n, seed=1):
    mx.random.seed(seed)
    return mx.random.normal((1, n, HIDDEN))


def test_gatetap_matches_group_expert_select():
    gate = make_gate()
    ctl = segroute.Controller(1, EXPERTS)
    x = tokens(20)
    inds, scores = GateTap(gate, 0, ctl)(x)
    ref_inds, ref_scores = group_expert_select(x @ gate.weight.T, gate.e_score_correction_bias, TOP_K, 1, 1, 2.5, True)
    a, b = np.array(inds), np.array(ref_inds)
    assert all(set(r1) == set(r2) for r1, r2 in zip(a.reshape(-1, TOP_K), b.reshape(-1, TOP_K), strict=True))
    order_a, order_b = np.argsort(a, axis=-1), np.argsort(b, axis=-1)
    sa = np.take_along_axis(np.array(scores), order_a, axis=-1)
    sb = np.take_along_axis(np.array(ref_scores), order_b, axis=-1)
    assert np.allclose(sa, sb, atol=1e-5)


def test_nemotron_mask_never_selects_masked():
    gate = make_gate()
    ctl = segroute.Controller(1, EXPERTS)
    ctl.mode, ctl.k, ctl.s = "mask", 8, None
    tap = GateTap(gate, 0, ctl)
    tap(tokens(30, seed=2))  # prefill picks the allowed set
    allowed = ctl.allowed[0]
    assert len(allowed) == 8
    outside = [e for e in range(EXPERTS) if e not in allowed]
    boost = np.zeros(EXPERTS, dtype=np.float32)
    boost[outside] = 5.0
    gate.e_score_correction_bias = gate.e_score_correction_bias + mx.array(boost)
    inds, _ = tap(tokens(64, seed=3))
    assert set(np.array(inds).reshape(-1).tolist()) <= allowed


def test_logged_weights_are_unmasked_selection():
    gate = make_gate(bias_scale=0.05)
    gate.e_score_correction_bias = gate.e_score_correction_bias + 40.0  # real Nemotron: large bias, small spread
    ctl = segroute.Controller(1, EXPERTS)
    ctl.mode = "log"
    x = tokens(40, seed=4)
    inds, _ = GateTap(gate, 0, ctl)(x)
    logged = np.array(ctl.log[0][0])  # (L, E)
    assert logged.shape == (40, EXPERTS)
    assert np.allclose(logged.sum(axis=-1), 1.0, atol=1e-5)
    assert ((logged > 0).sum(axis=-1) == TOP_K).all()
    sel = np.array(inds)[0]
    assert all(set(np.nonzero(row)[0]) == set(i) for row, i in zip(logged, sel, strict=True))
    scores = np.array(mx.sigmoid(x @ gate.weight.T))[0]
    expect = np.zeros_like(scores)
    np.put_along_axis(expect, sel, np.take_along_axis(scores, sel, axis=-1), axis=-1)
    assert np.allclose(logged, expect / expect.sum(axis=-1, keepdims=True), atol=1e-5)


def test_masked_expert_can_reenter_on_reselect():
    gate = make_gate(bias_scale=0.05)
    gate.e_score_correction_bias = gate.e_score_correction_bias + 40.0
    ctl = segroute.Controller(1, EXPERTS)
    ctl.mode, ctl.k, ctl.s = "mask", 8, 1
    tap = GateTap(gate, 0, ctl)
    tap(tokens(30, seed=2))
    outside = next(e for e in range(EXPERTS) if e not in ctl.allowed[0])
    boost = np.zeros(EXPERTS, dtype=np.float32)
    boost[outside] = 5.0
    gate.e_score_correction_bias = gate.e_score_correction_bias + mx.array(boost)
    tap(tokens(1, seed=3))  # masked out, but the unmasked router now picks it: logged and reselected
    assert outside in ctl.allowed[0]
    inds, _ = tap(tokens(1, seed=3))
    assert outside in np.array(inds).reshape(-1).tolist()


def fake_layers():
    gemma = SimpleNamespace(router=SimpleNamespace(proj=nn.Linear(HIDDEN, 128, bias=False)))
    qwen = SimpleNamespace(mlp=SimpleNamespace(gate=nn.Linear(HIDDEN, 64, bias=False), switch_mlp=object()))
    nemotron = SimpleNamespace(mixer=SimpleNamespace(gate=make_gate()))
    attention = SimpleNamespace(mixer=SimpleNamespace(q_proj=object()))
    dense = SimpleNamespace(mlp=SimpleNamespace(up_proj=object()))
    return gemma, qwen, nemotron, attention, dense


def test_find_routers_kinds_and_skips():
    gemma, qwen, nemotron, attention, dense = fake_layers()
    model = SimpleNamespace(layers=[gemma, attention, qwen, dense, nemotron])
    found = find_routers(model)
    assert [(owner, attr, kind) for owner, attr, kind in found] == [
        (gemma.router, "proj", "softmax_linear"),
        (qwen.mlp, "gate", "softmax_linear"),
        (nemotron.mixer, "gate", "sigmoid_bias"),
    ]
    assert [n_experts(*r) for r in found] == [128, 64, EXPERTS]


def test_find_routers_layer_paths():
    gemma = fake_layers()[0]
    nested = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[gemma])))
    hybrid = SimpleNamespace(backbone=SimpleNamespace(layers=[gemma]))
    assert len(find_routers(nested)) == 1
    assert len(find_routers(hybrid)) == 1


def test_install_wraps_each_kind():
    gemma, qwen, nemotron, _, _ = fake_layers()
    ctl = segroute.install(SimpleNamespace(layers=[gemma, qwen]))
    assert isinstance(gemma.router.proj, segroute.Tap)
    assert (ctl.n_layers, ctl.n_experts) == (2, 128)
    segroute.install(SimpleNamespace(layers=[nemotron]))
    assert isinstance(nemotron.mixer.gate, GateTap)


def test_observe_returns_bias_only_in_mask_mode():
    probs = mx.softmax(mx.random.normal((5, EXPERTS)), axis=-1)
    off = segroute.Controller(1, EXPERTS)
    assert off.observe(0, probs) is None
    log = segroute.Controller(1, EXPERTS)
    log.mode = "log"
    assert log.observe(0, probs) is None
    assert len(log.log[0]) == 1
    mask = segroute.Controller(1, EXPERTS)
    mask.mode, mask.k = "mask", 6
    bias = np.array(mask.observe(0, probs))
    assert bias.shape == (EXPERTS,)
    assert (bias == 0).sum() == 6 and (bias == -1e9).sum() == EXPERTS - 6


def test_grouped_routing_is_rejected():
    gate = make_gate()
    gate.n_group = 4
    with pytest.raises(AssertionError):
        GateTap(gate, 0, segroute.Controller(1, EXPERTS))


def test_controller_rejects_batch_above_one():
    ctl = segroute.Controller(1, EXPERTS)
    ctl.mode, ctl.k = "mask", 6
    with pytest.raises(ValueError, match="batch size 1"):
        ctl(0, mx.random.normal((2, 5, EXPERTS)))
    ctl.mode = "log"
    with pytest.raises(ValueError, match="batch size 1"):
        ctl(0, mx.random.normal((2, 5, EXPERTS)))
    assert ctl.log[0] == []


def test_gatetap_rejects_batch_above_one():
    ctl = segroute.Controller(1, EXPERTS)
    mx.random.seed(5)
    with pytest.raises(ValueError, match="batch size 1"):
        GateTap(make_gate(), 0, ctl)(mx.random.normal((2, 3, HIDDEN)))
