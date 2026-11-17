"""Router adapters: find and wrap the MoE router of each model family for the segment-routing controller."""

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nemotron_h import MoEGate

if TYPE_CHECKING:
    from segroute import Controller

SOFTMAX_LINEAR = "softmax_linear"  # Gemma 4 router.proj, Qwen3 and Qwen3.6 mlp.gate
SIGMOID_BIAS = "sigmoid_bias"  # Nemotron-H mixer.gate (MoEGate)


def check_batch(x: mx.array) -> None:
    """Raise unless x holds one sequence: the controller keeps one routing state, so it supports batch size 1 only."""
    if x.shape[0] != 1:
        raise ValueError(f"segment routing supports batch size 1 only, got a batch of {x.shape[0]}")


def _layers(model: nn.Module) -> list[nn.Module]:
    """Return the decoder layers of a model, whichever way mlx-lm nests them."""
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    return model.backbone.layers


def find_routers(model: nn.Module) -> list[tuple[nn.Module, str, str]]:
    """List (owner, attribute, kind) for the router of every MoE layer; other layers are skipped."""
    found = []
    for layer in _layers(model):
        mixer = getattr(layer, "mixer", None)
        if getattr(layer, "router", None) is not None:  # Gemma 4
            found.append((layer.router, "proj", SOFTMAX_LINEAR))
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):  # Qwen3, Qwen3.6
            found.append((layer.mlp, "gate", SOFTMAX_LINEAR))
        elif isinstance(getattr(mixer, "gate", None), (MoEGate, GateTap)):  # Nemotron-H, before or after install
            found.append((mixer, "gate", SIGMOID_BIAS))
    return found


def n_experts(owner: nn.Module, attribute: str, kind: str) -> int:
    """Number of routed experts behind a router."""
    router = getattr(owner, attribute)
    return router.n_routed_experts if kind == SIGMOID_BIAS else router.weight.shape[0]


class GateTap(nn.Module):
    """Wraps a Nemotron MoEGate: restricts selection to the resident set and logs the unmasked top-k routing weights."""

    def __init__(self, gate: nn.Module, layer: int, ctl: "Controller") -> None:
        super().__init__()
        assert gate.n_group in (None, 1), "grouped routing is not supported"
        self.gate = gate
        self.layer = layer
        self._ctl = [ctl]  # list keeps it out of the parameter tree

    def logits(self, x: mx.array) -> mx.array:
        """Float32 gate logits before the sigmoid (a hook for wrappers that change the gate logits)."""
        return (x @ self.gate.weight.T).astype(mx.float32)

    def bias(self, weights: mx.array, scores: mx.array) -> mx.array | None:
        """Additive selection bias from the controller, given the (L, E) unmasked top-k weights."""
        return self._ctl[0].observe(self.layer, weights)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        check_batch(x)
        gate = self.gate
        scores = mx.sigmoid(self.logits(x))
        r = scores + gate.e_score_correction_bias
        free = mx.argpartition(-r, kth=gate.top_k - 1, axis=-1)[..., : gate.top_k]  # selection without the mask
        weights = mx.take_along_axis(scores, free, axis=-1)
        weights = mx.put_along_axis(mx.zeros_like(scores), free, weights / weights.sum(axis=-1, keepdims=True), axis=-1)
        bias = self.bias(weights[0], scores[0])
        if bias is not None:
            r = r + bias
        inds = mx.stop_gradient(mx.argpartition(-r, kth=gate.top_k - 1, axis=-1)[..., : gate.top_k])
        top = mx.take_along_axis(scores, inds, axis=-1)
        if gate.top_k > 1 and gate.norm_topk_prob:
            top = top / (top.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, top * gate.routed_scaling_factor
