"""H61: size of the always-on path against the routed path (pass rules written 2026-11-03 09:16, before any run).

Full models, no mask, prefill only. Per MoE layer and prompt token, the norm of the always-on output a and of the
routed output r as each enters the layer's sum.

pathsize.py run --model-tag gemma     # results/pathsize_gemma.json
pathsize.py report                    # results/pathsize.json, H61 verdict
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_PROMPTS = 64  # per task, items 0 to 63
FACTOR = 1.5


def norms(x: Any) -> np.ndarray:
    """L2 norm of each token vector, last axis, flattened over the other axes."""
    a = np.asarray(x, dtype=np.float32)
    return np.linalg.norm(a.reshape(-1, a.shape[-1]), axis=-1)


def cosines(a: np.ndarray, r: np.ndarray) -> np.ndarray:
    a, r = a.reshape(-1, a.shape[-1]), r.reshape(-1, r.shape[-1])
    return (a * r).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(r, axis=-1), 1e-12)


class Tap(nn.Module):
    """Passes the output of a module through and keeps the last one."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.last: np.ndarray | None = None

    def __call__(self, *args: Any, **kw: Any) -> mx.array:
        out = self.inner(*args, **kw)
        self.last = np.array(out.astype(mx.float32))
        return out

    def take(self) -> np.ndarray:
        out = norms(self.last) if self.last is not None else np.zeros(0)
        self.last = None
        return out


class BlockTap(nn.Module):
    """An MoE block whose shared part is an attribute; keeps the shared output and the rest of the block output."""

    def __init__(self, inner: nn.Module, shared: Any) -> None:
        super().__init__()
        self.inner = inner
        self.shared = shared
        self.a: np.ndarray | None = None
        self.r: np.ndarray | None = None

    def __call__(self, x: mx.array) -> mx.array:
        out = self.inner(x)
        s = self.shared(self.inner, x)
        self.a = np.array(s.astype(mx.float32))
        self.r = np.array((out - s).astype(mx.float32))
        return out


def _qwen_shared(block: nn.Module, x: mx.array) -> mx.array:
    return mx.sigmoid(block.shared_expert_gate(x)) * block.shared_expert(x)


def _nemotron_shared(block: nn.Module, x: mx.array) -> mx.array:
    return block.shared_experts(x)


def install(model: nn.Module) -> list[Any]:
    """One probe per MoE layer: (Tap, Tap) on Gemma 4, a BlockTap on Qwen3.6 and Nemotron."""
    from adapters import _layers

    probes: list[Any] = []
    for layer in _layers(model):
        if getattr(layer, "router", None) is not None:
            layer.post_feedforward_layernorm_1 = Tap(layer.post_feedforward_layernorm_1)
            layer.post_feedforward_layernorm_2 = Tap(layer.post_feedforward_layernorm_2)
            probes.append((layer.post_feedforward_layernorm_1, layer.post_feedforward_layernorm_2))
        elif hasattr(getattr(layer, "mlp", None), "shared_expert"):
            layer.mlp = BlockTap(layer.mlp, _qwen_shared)
            probes.append(layer.mlp)
        elif hasattr(getattr(layer, "mixer", None), "shared_experts"):
            layer.mixer = BlockTap(layer.mixer, _nemotron_shared)
            probes.append(layer.mixer)
    return probes


def collect(probe: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(probe, tuple):
        a, r = probe[0].last, probe[1].last
        probe[0].last = probe[1].last = None
    else:
        a, r = probe.a, probe.r
        probe.a = probe.r = None
    return a, r


def layer_ratios(prompts: list[np.ndarray]) -> np.ndarray:
    """Per layer, summed always-on norm over summed routed norm; each prompt is (layers, [sum a, sum r])."""
    tot = np.sum(prompts, axis=0)
    return tot[:, 0] / tot[:, 1]


def summary(prompts: list[np.ndarray], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Median over layers of the per-layer ratio, with a 95% bootstrap interval over prompts."""
    arr = np.asarray(prompts)
    idx = np.random.default_rng(seed).integers(0, len(arr), size=(n_boot, len(arr)))
    boot = [float(np.median(layer_ratios(list(arr[i])))) for i in idx]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"median": float(np.median(layer_ratios(list(arr)))), "lo": float(lo), "hi": float(hi)}


def h61_verdict(models: dict[str, dict[str, float]]) -> str:
    g = models["gemma"]
    others = [v for m, v in models.items() if m != "gemma"]
    ok = all(g["median"] >= FACTOR * o["median"] and g["lo"] > o["hi"] for o in others)
    return "holds" if ok else "fails"


def cmd_run(a: argparse.Namespace) -> None:
    from mlx_lm import load

    import segroute as sr
    import short
    from models import MODELS

    model, tok = load(MODELS[a.model_tag])
    probes = install(model)
    sums, cos = [], []
    for _, _, prompt, _ in short.items(N_PROMPTS, 0):
        mx.eval(model(mx.array(sr.chat(tok, prompt))[None]))
        rows, crow = [], []
        for p in probes:
            av, rv = collect(p)
            rows.append([norms(av).sum(), norms(rv).sum()])
            crow.append(float(cosines(av, rv).mean()))
        sums.append(np.array(rows))
        cos.append(crow)
    out = {
        "model": a.model_tag,
        "n_prompts": len(sums),
        "layers": len(probes),
        "ratio_per_layer": layer_ratios(sums).tolist(),
        "cosine_per_layer": np.mean(cos, axis=0).tolist(),
        "summary": summary(sums),
    }
    np.save(RESULTS / f"pathsize_{a.model_tag}_sums.npy", np.asarray(sums))
    (RESULTS / f"pathsize_{a.model_tag}.json").write_text(json.dumps(out, indent=1))
    print(a.model_tag, out["summary"], flush=True)


def cmd_report(_: argparse.Namespace) -> None:
    models = {m: json.loads((RESULTS / f"pathsize_{m}.json").read_text()) for m in ("gemma", "nemotron", "qwen36")}
    out = {"verdict": h61_verdict({m: r["summary"] for m, r in models.items()}), "models": models}
    (RESULTS / "pathsize.json").write_text(json.dumps(out, indent=1))
    print(out["verdict"], {m: r["summary"] for m, r in models.items()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("run", "report"))
    p.add_argument("--model-tag")
    a = p.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
