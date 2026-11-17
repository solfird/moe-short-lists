"""Do experts share a curriculum? Random sets (H45), arithmetic (H46), output overlap (H47), router overlap (H48),
always-on ablation under ends64 (H43, T2), mask exposure (T3), larger budgets on summarize (H52).

curriculum.py random --model-tag gemma --k 24
curriculum.py arith --model-tag gemma --k 24 --arm long --rule ends     # rule: off, ends, random
curriculum.py overlap --model-tag gemma
curriculum.py routers --model-tag gemma --k 24
curriculum.py ablate --model-tag gemma --k 24 --alpha 0.75              # --k 0: unmasked
curriculum.py exposure --k 16
curriculum.py frontier --model-tag gemma --k 48
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import ablate
import segroute as sr
import selrule as sl
import short
from adapters import SOFTMAX_LINEAR, GateTap, _layers, find_routers, n_experts

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 0
SUITE_N, GSM_N, ARITH_N, ROUTER_N, OVERLAP_N = 300, 200, 300, 100, 8
FRONTIER_N, ARITH_LONG_N = 100, 100
BUDGETS = {"gemma": 24, "nemotron": 16, "qwen36": 24}


def arith_short(n: int) -> list[tuple[str, str]]:
    """One operation, number only: a two-digit sum or a two-digit by one-digit product."""
    rng = random.Random(SEED)
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            x, gold = f"{a} + {b}", a + b
        else:
            a, b = rng.randint(10, 99), rng.randint(2, 9)
            x, gold = f"{a} * {b}", a * b
        out.append((f"What is {x}? Reply with the number only.", str(gold)))
    return out


def arith_long(n: int, count: int = 24) -> list[tuple[str, str]]:
    """Same skill, long answer: count numbers added with one running total per line."""
    rng = random.Random(SEED + 1)
    out = []
    for _ in range(n):
        nums = [rng.randint(10, 99) for _ in range(count)]
        written = ", ".join(map(str, nums))
        out.append(
            (
                f"Add these numbers: {written}. Write one running total per line. End with a line 'The answer is N.'",
                str(sum(nums)),
            )
        )
    return out


def exposed_layers(n_layers: int, n_masked: int) -> list[int]:
    """n_masked evenly spaced layer indices."""
    return sorted({int(i * n_layers / n_masked) for i in range(n_masked)})


def top_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    x, y = set(np.argsort(-a)[:k].tolist()), set(np.argsort(-b)[:k].tolist())
    return len(x & y) / len(x | y)


def shared_energy(x: np.ndarray) -> float:
    """Share of the energy of the expert outputs (experts, dims) held by their mean."""
    return float((x.mean(axis=0) ** 2).sum() * len(x) / (x**2).sum())


def pair_cosine(x: np.ndarray) -> float:
    """Mean cosine over distinct pairs of rows."""
    u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    c = u @ u.T
    n = len(x)
    return float((c.sum() - np.trace(c)) / (n * (n - 1)))


def token_type(text: str) -> str:
    t = text.strip()
    if t.startswith("<") and t.endswith(">"):
        return "template"
    if re.fullmatch(r"[\d.,]+", t) and any(c.isdigit() for c in t):
        return "number"
    if t and not any(c.isalnum() for c in t):
        return "punct"
    return "word"


def h46_reading(short_kept: dict[str, float], long_kept: dict[str, float]) -> str:
    s, ln = sl.at_bar([short_kept]), sl.at_bar([long_kept])
    if s and ln:
        return "word_problems"
    if s:
        return "length"
    return "ability"


class RandomController(sl.TailController):
    """Same life cycle as ends64, but each selection takes K' experts at random."""

    rng = np.random.default_rng(SEED)

    def _select(self, layer: int, prob_sum: Any) -> None:
        super()._select(layer, self.rng.random(self.n_experts))


class ExposureController(sl.TailController):
    """ends64 on the chosen layers only; the other layers route freely."""

    masked: set[int] = set()

    def observe(self, layer: int, probs: Any) -> Any:
        return super().observe(layer, probs) if layer in self.masked else None


def install_with(model: nn.Module, cls: type[sl.TailController]) -> sl.TailController:
    targets = find_routers(model)
    ctl = cls(len(targets), n_experts(*targets[0]), lambda: sl.TAIL, "ends")
    for idx, (owner, attr, kind) in enumerate(targets):
        router = getattr(owner, attr)
        setattr(owner, attr, sr.Tap(router, idx, ctl) if kind == SOFTMAX_LINEAR else GateTap(router, idx, ctl))
    return ctl


def _suite(model_tag: str, k: int, tag: str, install: Any, start: int, load: Any = None) -> None:
    orig = short.items, sr.install, short.load
    if start:
        short.items = lambda n: orig[0](n, start)
    sr.install = install
    if load:
        short.load = load
    try:
        n_exp = 256 if model_tag == "qwen36" else 128
        short.cmd_run(argparse.Namespace(model_tag=model_tag, tag=tag, n=SUITE_N, frac=k / n_exp, s=32))
    finally:
        short.items, sr.install, short.load = orig


def cmd_random(a: argparse.Namespace) -> None:
    from models import MODELS

    _suite(a.model_tag, a.k, f"{a.model_tag}_d300_random", lambda m: install_with(m, RandomController), 300)
    orig = sr.install
    sr.install = lambda m: install_with(m, RandomController)
    try:
        sr.cmd_gsm8k(
            argparse.Namespace(
                model=MODELS[a.model_tag],
                tag=f"{a.model_tag}_random",
                k=a.k,
                s=32,
                n=GSM_N,
                max_tokens=512,
                router=None,
            )
        )
    finally:
        sr.install = orig


def cmd_arith(a: argparse.Namespace) -> None:
    from mlx_lm import load

    from models import MODELS

    path = RESULTS / f"arith_{a.model_tag}_{a.arm}_{a.rule}_k{a.k}.jsonl"
    items = {
        "short": lambda: arith_short(ARITH_N),
        "mid": lambda: arith_long(ARITH_LONG_N, 8),
        "long": lambda: arith_long(ARITH_LONG_N, 24),
    }[a.arm]()
    limit = 8 if a.arm == "short" else 256
    model, tok, config = load(MODELS[a.model_tag], return_config=True)
    ctl = install_with(model, RandomController if a.rule == "random" else sl.TailController)
    ctl.mode, ctl.k, ctl.s = ("off", None, None) if a.rule == "off" else ("mask", a.k, 32)
    with path.open("w") as f:
        for i, (prompt, gold) in enumerate(items):
            ctl.reset()
            gen = sr.generate(model, tok, sr.chat(tok, prompt), limit, config)
            text = tok.decode(gen)
            pred = sr.gsm_answer(text)
            ok = pred is not None and float(pred) == float(gold)
            f.write(json.dumps({"i": i, "ok": ok, "gold": gold, "out": text, "n_gen": len(gen)}) + "\n")
    print(path.name, flush=True)


def cmd_frontier(a: argparse.Namespace) -> None:
    """H52: summarize at larger budgets, items 300 to 399, ends64."""
    from mlx_lm import load

    from models import MODELS

    path = RESULTS / f"frontier_{a.model_tag}_k{a.k}.jsonl"
    model, tok, config = load(MODELS[a.model_tag], return_config=True)
    ctl = install_with(model, sl.TailController)
    ctl.mode, ctl.k, ctl.s = "mask", a.k, 32
    rows = [x for x in short.items(FRONTIER_N, 300) if x[0] == "summarize"]
    with path.open("w") as f:
        for task, i, prompt, ref in rows:
            ctl.reset()
            gen = sr.generate(model, tok, sr.chat(tok, prompt), short.MAX[task], config)
            f.write(json.dumps({"task": task, "i": i, "ref": ref, "out": tok.decode(gen), "gen_ids": gen}) + "\n")
    print(path.name, flush=True)


def _blocks(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """(owner, attribute of the expert bank) per MoE layer."""
    out = [(owner, "switch_mlp") for owner, _, _ in find_routers(model) if hasattr(owner, "switch_mlp")]
    if out:
        return out
    return [(layer.experts, "switch_glu") for layer in _layers(model) if getattr(layer, "router", None) is not None]


class Capture(nn.Module):
    def __init__(self, inner: nn.Module, store: list) -> None:
        super().__init__()
        self.inner = inner
        self._store = [store]

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        self._store[0].append(x)
        return self.inner(x, indices)


def _prompts(n: int) -> dict[str, list[str]]:
    from datasets import load_dataset

    out: dict[str, list[str]] = {"summarize": [], "qa": [], "classify": []}
    for t, _, p, _ in short.items(n, 300):
        if t in out:
            out[t].append(p)
    test = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    out["math"] = [sr.gsm_prompt(ex["question"]) for ex in test]
    return out


def cmd_overlap(a: argparse.Namespace) -> None:
    """H47: same-token cosine of sampled experts, shared energy, floor from different tokens."""
    from mlx_lm import load

    from models import MODELS

    model, tok = load(MODELS[a.model_tag])
    blocks = _blocks(model)
    chosen = [int(i * len(blocks) / 6) for i in range(6)]
    rng = np.random.default_rng(SEED)
    rows = []
    for li in chosen:
        owner, attr = blocks[li]
        bank = getattr(owner, attr)
        store: list[mx.array] = []
        setattr(owner, attr, Capture(bank, store))
        for prompts in _prompts(OVERLAP_N).values():
            for p in prompts:
                mx.eval(model(mx.array(sr.chat(tok, p))[None]))
        setattr(owner, attr, bank)
        xs = mx.concatenate([x[0] for x in store], axis=0)
        idx = rng.choice(xs.shape[0], size=min(256, xs.shape[0]), replace=False)
        xs = xs[mx.array(idx.tolist())]
        n_exp = n_experts(*find_routers(model)[li])
        experts = rng.choice(n_exp, size=32, replace=False).tolist()
        outs = []
        for e in experts:
            y = bank(xs[None], mx.full((1, xs.shape[0], 1), e, dtype=mx.int32))
            mx.eval(y)
            outs.append(np.array(y[0, :, 0].astype(mx.float32)))
        o = np.stack(outs, axis=1)  # (tokens, experts, dims)
        same = float(np.mean([pair_cosine(o[t]) for t in range(o.shape[0])]))
        energy = float(np.mean([shared_energy(o[t]) for t in range(o.shape[0])]))
        perm = rng.permutation(o.shape[0])
        cross = o.copy()
        cross[:, 16:] = o[perm][:, 16:]
        u = cross / np.maximum(np.linalg.norm(cross, axis=2, keepdims=True), 1e-12)
        floor = float(np.mean((u[:, :16] * u[:, 16:]).sum(axis=2)))
        rows.append({"layer": li, "same_token": same, "shared_energy": energy, "other_token_floor": floor})
        print(rows[-1], flush=True)
    (RESULTS / f"curriculum_overlap_{a.model_tag}.json").write_text(json.dumps(rows, indent=1))


def cmd_routers(a: argparse.Namespace) -> None:
    """H48: top-K' overlap between tasks within a token type, and between token types within a task."""
    from mlx_lm import load

    from models import MODELS

    model, tok = load(MODELS[a.model_tag])
    ctl = sr.install(model)
    ctl.mode = "log"
    types = ("word", "number", "punct", "template")
    mass: dict[tuple[str, str], list[np.ndarray]] = {}
    half: dict[tuple[str, int], list[np.ndarray]] = {}
    for task, prompts in _prompts(ROUTER_N).items():
        for n, p in enumerate(prompts):
            ctl.reset()
            ids = sr.chat(tok, p)
            mx.eval(model(mx.array(ids)[None]))
            kinds = [token_type(tok.decode([t])) for t in ids]
            probs = np.stack([np.array(ctl.log[lay][0]) for lay in range(ctl.n_layers)])  # (layers, L, E)
            half.setdefault((task, n % 2), []).append(probs.mean(axis=1))
            for ty in types:
                keep = [j for j, x in enumerate(kinds) if x == ty]
                if keep:
                    mass.setdefault((task, ty), []).append(probs[:, keep].mean(axis=1))
    means = {key: np.mean(v, axis=0) for key, v in mass.items()}
    tasks = sorted({t for t, _ in means})

    def overlap(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean([top_jaccard(x[lay], y[lay], a.k) for lay in range(len(x))]))

    across_tasks = [
        overlap(means[(s, ty)], means[(t, ty)])
        for ty in types
        for i, s in enumerate(tasks)
        for t in tasks[i + 1 :]
        if (s, ty) in means and (t, ty) in means
    ]
    across_types = [
        overlap(means[(t, x)], means[(t, y)])
        for t in tasks
        for i, x in enumerate(types)
        for y in types[i + 1 :]
        if (t, x) in means and (t, y) in means
    ]
    floor = [overlap(np.mean(half[(t, 0)], axis=0), np.mean(half[(t, 1)], axis=0)) for t in tasks]
    out = {
        "k": a.k,
        "across_tasks_same_type": float(np.mean(across_tasks)),
        "across_types_same_task": float(np.mean(across_types)),
        "split_half_same_task": float(np.mean(floor)),
    }
    print(out)
    (RESULTS / f"curriculum_routers_{a.model_tag}.json").write_text(json.dumps(out, indent=1))


def scale_shared(model: nn.Module, alpha: float) -> int:
    """Scale the always-on output of every MoE layer: Gemma 4 dense path, Qwen3.6 and Nemotron shared experts."""
    n = 0
    for layer in _layers(model):
        if getattr(layer, "router", None) is not None:
            layer.post_feedforward_layernorm_1 = ablate.ScaledOutput(layer.post_feedforward_layernorm_1, alpha)
            n += 1
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "shared_expert"):
            layer.mlp.shared_expert = ablate.ScaledOutput(layer.mlp.shared_expert, alpha)
            n += 1
        elif hasattr(getattr(layer, "mixer", None), "shared_experts"):
            layer.mixer.shared_experts = ablate.ScaledOutput(layer.mixer.shared_experts, alpha)
            n += 1
    return n


def cmd_ablate(a: argparse.Namespace) -> None:
    short_load = short.load

    def scaled(*args: Any, **kw: Any) -> Any:
        out = short_load(*args, **kw)
        print("layers scaled:", scale_shared(out[0], a.alpha), flush=True)
        return out

    _suite(
        a.model_tag,
        a.k,
        f"{a.model_tag}_a{a.alpha:.2f}_ends64",
        lambda m: install_with(m, sl.TailController),
        0,
        load=scaled,
    )


def cmd_exposure(a: argparse.Namespace) -> None:
    def install(m: nn.Module) -> sl.TailController:
        ctl = install_with(m, ExposureController)
        ctl.masked = set(exposed_layers(ctl.n_layers, 13))
        return ctl

    _suite("gemma", a.k, "gemma_d300_exposure13", install, 300)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("random")
    r.add_argument("--model-tag", required=True)
    r.add_argument("--k", type=int, required=True)
    ar = sub.add_parser("arith")
    ar.add_argument("--model-tag", required=True)
    ar.add_argument("--k", type=int, required=True)
    ar.add_argument("--arm", required=True, choices=["short", "mid", "long"])
    ar.add_argument("--rule", required=True, choices=["off", "ends", "random"])
    o = sub.add_parser("overlap")
    o.add_argument("--model-tag", required=True)
    ro = sub.add_parser("routers")
    ro.add_argument("--model-tag", required=True)
    ro.add_argument("--k", type=int, required=True)
    ab = sub.add_parser("ablate")
    ab.add_argument("--model-tag", required=True)
    ab.add_argument("--k", type=int, required=True)
    ab.add_argument("--alpha", type=float, required=True)
    ex = sub.add_parser("exposure")
    ex.add_argument("--k", type=int, required=True)
    fr = sub.add_parser("frontier")
    fr.add_argument("--model-tag", required=True)
    fr.add_argument("--k", type=int, required=True)
    a = p.parse_args()
    {
        "random": cmd_random,
        "arith": cmd_arith,
        "overlap": cmd_overlap,
        "routers": cmd_routers,
        "ablate": cmd_ablate,
        "exposure": cmd_exposure,
        "frontier": cmd_frontier,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
