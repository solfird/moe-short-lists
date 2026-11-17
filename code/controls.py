"""Review controls (pass rules written 2026-11-03 06:20): H55 fixed set, H56 routed-path cut.

controls.py calibrate --model-tag gemma           # results/static_mass_gemma.npy, router mass on suite items 0 to 299
controls.py static --model-tag gemma --k 24       # tag gemma_d300_static, suite items 300 to 599
controls.py routed --model-tag gemma --k 24       # tag gemma_r0.75_ends64, items 0 to 299; --k 0 unmasked
"""

import argparse
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import curriculum as cu
import segroute as sr
import selrule as sl
import short
from adapters import _layers

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALPHA = 0.75


class FixedController(sl.TailController):
    """One set per layer from calibration mass, chosen once per answer and never reselected."""

    fixed: list[np.ndarray] = []

    def observe(self, layer: int, probs: Any) -> Any:
        if self.mode != "mask":
            return super().observe(layer, probs)
        if self.bias[layer] is None:
            self._select(layer, self.fixed[layer])
        return self.bias[layer]


def calibration_mass(per_prompt: list[np.ndarray]) -> np.ndarray:
    """Mean over prompts of each prompt's (layers, tokens, experts) router probabilities averaged over tokens."""
    return np.mean([p.mean(axis=1) for p in per_prompt], axis=0)


class Scaled(nn.Module):
    """Multiplies the output of a module by a constant, whatever its arguments."""

    def __init__(self, inner: nn.Module, alpha: float) -> None:
        super().__init__()
        self.inner, self.alpha = inner, alpha

    def __call__(self, *args: Any) -> mx.array:
        return self.inner(*args) * self.alpha


def scale_routed(layers: list[nn.Module], alpha: float) -> int:
    """Scale the routed-expert output of every MoE layer: Gemma 4 routed norm, Qwen3.6 switch MLP."""
    n = 0
    for layer in layers:
        if getattr(layer, "router", None) is not None:
            layer.post_feedforward_layernorm_2 = Scaled(layer.post_feedforward_layernorm_2, alpha)
            n += 1
        elif hasattr(getattr(layer, "mlp", None), "switch_mlp"):
            layer.mlp.switch_mlp = Scaled(layer.mlp.switch_mlp, alpha)
            n += 1
    return n


def cmd_calibrate(a: argparse.Namespace) -> None:
    from mlx_lm import load

    from models import MODELS

    model, tok = load(MODELS[a.model_tag])
    ctl = sr.install(model)
    ctl.mode = "log"
    per_prompt = []
    for _, _, prompt, _ in short.items(300, 0):
        ctl.reset()
        mx.eval(model(mx.array(sr.chat(tok, prompt))[None]))
        per_prompt.append(np.stack([np.array(ctl.log[lay][0]) for lay in range(ctl.n_layers)]))
    np.save(RESULTS / f"static_mass_{a.model_tag}.npy", calibration_mass(per_prompt))
    print("prompts", len(per_prompt), flush=True)


def cmd_static(a: argparse.Namespace) -> None:
    mass = np.load(RESULTS / f"static_mass_{a.model_tag}.npy")

    def install(m: nn.Module) -> sl.TailController:
        ctl = cu.install_with(m, FixedController)
        ctl.fixed = list(mass)
        return ctl

    cu._suite(a.model_tag, a.k, f"{a.model_tag}_d300_static", install, 300)


def cmd_routed(a: argparse.Namespace) -> None:
    load = short.load

    def scaled(*args: Any, **kw: Any) -> Any:
        out = load(*args, **kw)
        print("layers scaled:", scale_routed(_layers(out[0]), ALPHA), flush=True)
        return out

    tag = f"{a.model_tag}_r{ALPHA:.2f}_ends64"
    cu._suite(a.model_tag, a.k, tag, lambda m: cu.install_with(m, sl.TailController), 0, load=scaled)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("calibrate", "static", "routed"))
    p.add_argument("--model-tag", required=True)
    p.add_argument("--k", type=int, default=0)
    a = p.parse_args()
    {"calibrate": cmd_calibrate, "static": cmd_static, "routed": cmd_routed}[a.cmd](a)


if __name__ == "__main__":
    main()
