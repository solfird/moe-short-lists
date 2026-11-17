import mlx.core as mx
import mlx.nn as nn
import numpy as np

import controls


def probs(n: int, e: int, hot: int) -> mx.array:
    p = np.full((n, e), 0.01, dtype=np.float32)
    p[:, hot] = 1.0
    return mx.array(p / p.sum(axis=1, keepdims=True))


def test_fixed_set_comes_from_calibration_mass_and_never_changes():
    ctl = controls.FixedController(n_layers=1, n_experts=8, tail=lambda: None, rule="ends")
    ctl.fixed = [np.array([0, 0, 0, 0, 0, 3.0, 2.0, 0])]
    ctl.mode, ctl.k, ctl.s = "mask", 2, 4
    ctl.observe(0, probs(5, 8, hot=0))  # prompt prefers expert 0; the fixed set wins
    assert ctl.allowed[0] == {5, 6}
    for _ in range(20):  # past several S windows of tokens that prefer expert 1
        ctl.observe(0, probs(1, 8, hot=1))
    assert ctl.allowed[0] == {5, 6}
    assert ctl.loads[0] == 2 and ctl.reselects[0] == 0


def test_calibration_mass_is_the_mean_of_per_prompt_token_means():
    a = np.array([[[1.0, 0.0], [0.0, 1.0]]])  # one layer, two tokens
    b = np.array([[[1.0, 0.0]]])
    assert np.allclose(controls.calibration_mass([a, b]), [[0.75, 0.25]])


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = object()
        self.post_feedforward_layernorm_2 = nn.Identity()


def test_scale_routed_wraps_the_gemma_routed_norm():
    layers = [Block(), Block()]
    assert controls.scale_routed(layers, 0.75) == 2
    out = layers[0].post_feedforward_layernorm_2(mx.array([4.0]))
    assert float(out[0]) == 3.0
