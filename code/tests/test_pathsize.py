import mlx.core as mx
import mlx.nn as nn
import numpy as np

import pathsize as ps


class Double(nn.Module):
    def __call__(self, x):
        return 2 * x


def test_tap_records_token_norms_and_returns_the_output_unchanged():
    tap = ps.Tap(Double())
    x = mx.array([[[3.0, 4.0], [0.0, 1.0]]])
    assert np.allclose(np.array(tap(x)), [[[6.0, 8.0], [0.0, 2.0]]])
    assert np.allclose(tap.take(), [10.0, 2.0])
    assert tap.take().size == 0


def test_routed_part_is_the_block_output_minus_the_always_on_part():
    total = np.array([[3.0, 4.0], [1.0, 1.0]])
    always = np.array([[3.0, 0.0], [1.0, 1.0]])
    assert np.allclose(ps.norms(total - always), [4.0, 0.0])


def test_ratio_per_layer_is_summed_always_on_over_summed_routed_norms():
    prompts = [np.array([[2.0, 1.0], [1.0, 1.0]]), np.array([[2.0, 3.0], [3.0, 1.0]])]  # (layers, [a, r])
    assert np.allclose(ps.layer_ratios(prompts), [1.0, 2.0])


def test_model_summary_is_the_median_over_layers_with_a_bootstrap_ci():
    prompts = [np.array([[2.0, 1.0], [1.0, 1.0], [3.0, 1.0]])] * 5
    s = ps.summary(prompts)
    assert s["median"] == 2.0 and s["lo"] == 2.0 and s["hi"] == 2.0


def test_h61_needs_gemma_one_and_a_half_times_both_others_with_cis_apart():
    g = {"median": 3.0, "lo": 2.8, "hi": 3.2}
    low = {"median": 1.0, "lo": 0.9, "hi": 1.1}
    near = {"median": 2.1, "lo": 1.9, "hi": 2.9}
    assert ps.h61_verdict({"gemma": g, "nemotron": low, "qwen36": low}) == "holds"
    assert ps.h61_verdict({"gemma": g, "nemotron": low, "qwen36": near}) == "fails"
