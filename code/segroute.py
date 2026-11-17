"""Segment routing retrofit for vanilla MoE models in MLX.

Wraps each layer's router projection so the routing logits can be logged (unmasked) or masked to a
per-layer expert set chosen from the prompt and reselected every S generated tokens.

    segroute.py log      --model M --tag gemma           # phase 1: unmasked runs, router softmax saved
    segroute.py coverage --tag gemma                     # phase 1: offline sweep over K' and S
    segroute.py gsm8k    --model M --tag gemma --k 16 --s 32 --n 200   # phase 2: masked quality
"""

import argparse
import csv
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from adapters import SOFTMAX_LINEAR, GateTap, check_batch, find_routers, n_experts
from models import eos_ids
from records import exclusive, lock_path, provenance, read_records, repair_tail, write_meta

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / "runs"
RESULTS = HERE / "results"
TOP_K = 8


class Controller:
    """Per-layer state: what to log, which experts are allowed, when to reselect."""

    def __init__(self, n_layers, n_experts):
        self.n_layers, self.n_experts = n_layers, n_experts
        self.mode = "off"  # off | log | mask
        self.k = self.s = None
        self.reset()

    def reset(self):
        self.log = [[] for _ in range(self.n_layers)]
        self.bias = [None] * self.n_layers
        self.allowed = [None] * self.n_layers
        self.window = [None] * self.n_layers
        self.since = [0] * self.n_layers
        self.loads = [0] * self.n_layers  # experts read in, prompt set included
        self.reselects = [0] * self.n_layers
        self.touched = [set() for _ in range(self.n_layers)]

    def _select(self, layer, prob_sum):
        top = np.argsort(-np.array(prob_sum))[: self.k]
        new = set(top.tolist())
        old = self.allowed[layer] or set()
        self.loads[layer] += len(new - old)
        self.allowed[layer] = new
        self.touched[layer] |= new
        mask = np.full(self.n_experts, -1e9, dtype=np.float32)
        mask[list(new)] = 0.0
        self.bias[layer] = mx.array(mask)

    def observe(self, layer: int, probs: mx.array) -> mx.array | None:
        """Log, window and reselect on a normalized (L, E) distribution. Returns the additive bias in mask mode."""
        if self.mode == "log":
            self.log[layer].append(probs)
        if self.mode != "mask":
            return None
        if self.bias[layer] is None:  # prefill: choose the set from this layer's prompt scores
            self._select(layer, probs.sum(axis=0))
        else:
            self.window[layer] = (
                probs.sum(axis=0) if self.window[layer] is None else self.window[layer] + probs.sum(axis=0)
            )
            self.since[layer] += probs.shape[0]
        bias = self.bias[layer]
        if self.s and self.since[layer] >= self.s:  # applies from the next token on
            self._select(layer, self.window[layer])
            self.reselects[layer] += 1
            self.window[layer], self.since[layer] = None, 0
        return bias

    def __call__(self, layer: int, logits: mx.array) -> mx.array:
        if self.mode == "off":
            return logits
        check_batch(logits)
        bias = self.observe(layer, mx.softmax(logits.astype(mx.float32), axis=-1)[0])  # (L, E)
        return logits if bias is None else logits + bias.astype(logits.dtype)


class Tap(nn.Module):
    def __init__(self, inner, layer, ctl):
        super().__init__()
        self.inner = inner
        self.layer = layer
        self._ctl = [ctl]  # list keeps it out of the parameter tree

    def __call__(self, x):
        return self._ctl[0](self.layer, self.inner(x))


def install(model: nn.Module) -> Controller:
    """Wrap the router of every MoE layer. Returns the controller."""
    targets = find_routers(model)
    ctl = Controller(len(targets), n_experts(*targets[0]))
    for idx, (owner, name, kind) in enumerate(targets):
        router = getattr(owner, name)
        setattr(owner, name, Tap(router, idx, ctl) if kind == SOFTMAX_LINEAR else GateTap(router, idx, ctl))
    return ctl


def chat(tok: TokenizerWrapper, text: str) -> list[int]:
    """Token ids of a single user turn with thinking off and no system prompt."""
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False
    )


def generate(
    model: nn.Module, tok: TokenizerWrapper, prompt_ids: list[int], max_tokens: int, config: dict | None = None
) -> list[int]:
    """Greedy decode with our own loop so the controller sees exactly one prefill call; stops at any eos id."""
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    out = []
    eos = eos_ids(tok, config)
    for _ in range(max_tokens):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(nxt)
        t = nxt.item()
        if t in eos:
            break
        out.append(t)
        logits = model(nxt[None], cache=cache)
    return out


def prompts(n_gsm, n_code, n_dolly, seed=0):
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    items = []
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for i in rng.choice(len(gsm), n_gsm, replace=False):
        items.append(("math", gsm[int(i)]["question"]))
    he = load_dataset("openai/openai_humaneval", split="test")
    for i in rng.choice(len(he), n_code, replace=False):
        items.append(("code", "Complete this Python function.\n\n" + he[int(i)]["prompt"]))
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly = dolly.filter(
        lambda r: not r["context"] and r["category"] in ("open_qa", "creative_writing", "brainstorming", "general_qa")
    )
    for i in rng.choice(len(dolly), n_dolly, replace=False):
        items.append((dolly[int(i)]["category"], dolly[int(i)]["instruction"]))
    return items


def cmd_log(a):
    model, tok, config = load(a.model, return_config=True)
    ctl = install(model)
    ctl.mode = "log"
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    items = prompts(a.n_gsm, a.n_code, a.n_dolly)
    meta = []
    t0 = time.time()
    for j, (kind, text) in enumerate(items):
        path = out / f"{j:04d}.npz"
        ids = chat(tok, text)
        if path.exists():
            continue
        ctl.reset()
        gen = generate(model, tok, ids, a.max_tokens, config)
        probs = np.stack(
            [np.array(mx.concatenate(ctl.log[lyr], axis=0)).astype(np.float16) for lyr in range(ctl.n_layers)]
        )
        np.savez_compressed(path, probs=probs, n_prompt=len(ids))  # probs: (layers, tokens, experts)
        meta.append({"i": j, "kind": kind, "n_prompt": len(ids), "n_gen": len(gen)})
        print(f"{j + 1}/{len(items)} {kind} prompt={len(ids)} gen={len(gen)} {time.time() - t0:.0f}s", flush=True)
    with open(out / "meta.jsonl", "a") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")


def replay(probs, n_prompt, k, s):
    """Offline estimate on an unmasked trajectory. Returns per-token mass kept, hit rate, loads."""
    n_layers, n_tok, _ = probs.shape
    top = np.argpartition(-probs, TOP_K, axis=-1)[..., :TOP_K]  # original top 8
    top_p = np.take_along_axis(probs, top, axis=-1)
    mass, hits, loads, reloads, nres = [], [], 0, 0, 0
    for lyr in range(n_layers):
        p = probs[lyr]
        allowed = set(np.argsort(-p[:n_prompt].sum(0))[:k].tolist())
        loads += k
        since, window = 0, np.zeros(p.shape[1], dtype=np.float32)
        for t in range(n_tok):
            inside = np.isin(top[lyr, t], list(allowed))
            if t >= n_prompt:
                mass.append(top_p[lyr, t][inside].sum() / top_p[lyr, t].sum())
                hits.append(inside.mean())
                window += p[t]
                since += 1
                if s and since >= s:
                    new = set(np.argsort(-window)[:k].tolist())
                    reloads += len(new - allowed)
                    nres += 1
                    allowed, since, window = new, 0, np.zeros_like(window)
    return np.mean(mass), np.mean(hits), loads + reloads, (reloads / nres if nres else 0.0)


def cmd_coverage(a):
    files = sorted((RUNS / a.tag).glob("*.npz"))
    meta = {m["i"]: m for m in map(json.loads, open(RUNS / a.tag / "meta.jsonl"))}  # noqa: SIM115 handle lives for the whole script run
    RESULTS.mkdir(exist_ok=True)
    rows, touched = [], []
    for f in files:
        d = np.load(f)
        probs, n_prompt = d["probs"].astype(np.float32), int(d["n_prompt"])
        if probs.shape[1] - n_prompt < 8:
            continue
        top = np.argpartition(-probs[:, n_prompt:], TOP_K, axis=-1)[..., :TOP_K]
        touched.append(np.mean([len(np.unique(top[lyr])) for lyr in range(probs.shape[0])]))
        kind = meta.get(int(f.stem), {}).get("kind", "?")
        for k in a.ks:
            for s in a.ss:
                m, h, loads, per_res = replay(probs, n_prompt, k, s)
                rows.append(
                    {
                        "prompt": f.stem,
                        "kind": kind,
                        "k": k,
                        "s": s or "static",
                        "mass": m,
                        "hit": h,
                        "loads_per_layer": loads / probs.shape[0],
                        "new_per_reselect": per_res,
                        "n_gen": probs.shape[1] - n_prompt,
                    }
                )
    with open(RESULTS / f"{a.tag}_coverage_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(
        f"{a.tag}: {len(touched)} answers; distinct experts touched per layer, unmasked: "
        f"mean {np.mean(touched):.1f}, median {np.median(touched):.1f} of {probs.shape[2]}"
    )
    print(f"{'K':>4} {'S':>7} {'mass':>6} {'p10':>6} {'hit':>6} {'new/resel':>9}")
    summary = []
    for k in a.ks:
        for s in a.ss:
            sel = [r for r in rows if r["k"] == k and r["s"] == (s or "static")]
            ms = np.array([r["mass"] for r in sel])
            row = {
                "k": k,
                "s": s or "static",
                "mass": ms.mean(),
                "mass_p10": np.percentile(ms, 10),
                "hit": np.mean([r["hit"] for r in sel]),
                "new_per_reselect": np.mean([r["new_per_reselect"] for r in sel]),
            }
            summary.append(row)
            print(
                f"{k:>4} {str(row['s']):>7} {row['mass']:6.3f} {row['mass_p10']:6.3f} {row['hit']:6.3f} {row['new_per_reselect']:9.2f}"  # noqa: E501 prompt or log text kept byte-identical for reproducibility
            )
    with open(RESULTS / f"{a.tag}_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)


def gsm_answer(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm_prompt(question: str) -> str:
    """User turn of a GSM8K problem, as every GSM8K run and the C2 test send it."""
    return question + "\n\nSolve step by step. End with a line 'The answer is N.'"


def gsm8k_done(path: Path) -> set[int]:
    """Indices already in a GSM8K output file; a torn last line is cut first so the resume never glues onto it."""
    if not path.exists():
        return set()
    repair_tail(path)
    return {r["i"] for r in read_records(path)}


def cmd_gsm8k(a):
    from datasets import load_dataset

    name = f"{a.tag}_gsm8k_k{a.k}_s{a.s}"
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.jsonl"
    with exclusive(lock_path(path)):
        _gsm8k_locked(a, load_dataset, name, path)


def _gsm8k_locked(a: argparse.Namespace, load_dataset: Callable, name: str, path: Path) -> None:
    """Body of cmd_gsm8k, run while this process holds the output file's lock."""
    params = {"tag": a.tag, "k": a.k, "s": a.s, "n": a.n, "max_tokens": a.max_tokens}
    write_meta(path, provenance(a.model, params))
    model, tok, config = load(a.model, return_config=True)
    ctl = install(model)
    ctl.mode, ctl.k, ctl.s = ("off", None, None) if a.k == 0 else ("mask", a.k, a.s or None)
    test = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
    done = gsm8k_done(path)
    t0 = time.time()
    with open(path, "a") as f:
        for i, ex in enumerate(test):
            if i in done:
                continue
            ctl.reset()
            ids = chat(tok, gsm_prompt(ex["question"]))
            gen = generate(model, tok, ids, a.max_tokens, config)
            text = tok.decode(gen)
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            pred = gsm_answer(text)
            ok = pred is not None and float(pred) == float(gold)
            rec = {
                "i": i,
                "ok": ok,
                "pred": pred,
                "gold": gold,
                "n_gen": len(gen),
                "text": text,
                "gen_ids": gen,
                "prompt_ids_len": len(ids),
            }
            if ctl.mode == "mask":
                rec |= {
                    "loads_per_layer": float(np.mean(ctl.loads)),
                    "touched_per_layer": float(np.mean([len(t) for t in ctl.touched])),
                    "reselects": int(np.mean(ctl.reselects)),
                }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{name} {i + 1}/{a.n} ok={ok} {time.time() - t0:.0f}s", flush=True)
    recs = read_records(path)
    print(f"{name}: {sum(r['ok'] for r in recs)}/{len(recs)} correct")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("--model", required=True)
    lg.add_argument("--tag", required=True)
    lg.add_argument("--n-gsm", type=int, default=80)
    lg.add_argument("--n-code", type=int, default=40)
    lg.add_argument("--n-dolly", type=int, default=80)
    lg.add_argument("--max-tokens", type=int, default=384)
    cv = sub.add_parser("coverage")
    cv.add_argument("--tag", required=True)
    cv.add_argument("--ks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    cv.add_argument("--ss", type=int, nargs="+", default=[16, 32, 64, 0])  # 0 = static
    gs = sub.add_parser("gsm8k")
    gs.add_argument("--model", required=True)
    gs.add_argument("--tag", required=True)
    gs.add_argument("--k", type=int, default=0)  # 0 = unmasked baseline
    gs.add_argument("--s", type=int, default=32)  # 0 = static
    gs.add_argument("--n", type=int, default=200)
    gs.add_argument("--max-tokens", type=int, default=512)
    a = p.parse_args()
    {"log": cmd_log, "coverage": cmd_coverage, "gsm8k": cmd_gsm8k}[a.cmd](a)


if __name__ == "__main__":
    main()
