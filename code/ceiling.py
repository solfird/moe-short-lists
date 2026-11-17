"""Ceiling test (H33): a fixed set per layer from both prompt ends and the stored unmasked answer. Cannot deploy.

ceiling.py run --model-tag gemma --k 12 --n 300     # results/gemma_ceiling_short_n300_k12_s32.jsonl
ceiling.py report --n 300                           # results/ceiling_h33.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import promptlen as pl
import segroute as sr
import selrule as sl
import short

RESULTS = Path(__file__).resolve().parents[1] / "results"
ENDS = 32
CELLS = {"gemma": 12, "nemotron": 12, "qwen36": 16}  # one budget step below each H31 frontier


def chosen_tokens(n_prompt: int, n_tok: int) -> list[int]:
    """Token positions the set is chosen from: first and last ENDS of the prompt, then the whole answer."""
    prompt = list(range(n_prompt))
    if n_prompt > 2 * ENDS:
        prompt = [*prompt[:ENDS], *prompt[-ENDS:]]
    return [*prompt, *range(n_prompt, n_tok)]


def fix_sets(ctl: Any, model: Any, ids: list[int], gen: list[int], idx: list[int] | None = None) -> None:
    """One unmasked pass over prompt and answer, then one set per layer for the whole item, with no reselection."""
    ctl.reset()
    ctl.mode, ctl.s = "log", None
    model(mx.array(ids + gen)[None])
    idx = chosen_tokens(len(ids), len(ids) + len(gen)) if idx is None else idx
    pos = mx.array(idx)
    scores = [mx.concatenate(rows, axis=0)[pos].sum(axis=0) for rows in ctl.log]
    mx.eval(scores)
    ctl.reset()
    ctl.mode = "mask"
    for layer, score in enumerate(scores):
        ctl._select(layer, score)


def cmd_run(a: argparse.Namespace) -> None:
    """One budget through short.cmd_run, the path of every other run."""
    stored = {(r["task"], r["i"]): r["gen_ids"] for r in short.read_records(short.run_path(a.model_tag, a.n, 0, 32))}
    answers = {prompt: stored[task, i] for task, i, prompt, _ in short.items(a.n)}
    orig_chat, orig_install, orig_generate = sr.chat, sr.install, sr.generate
    state: dict[str, Any] = {}

    def chat(tok: Any, text: str) -> list[int]:
        state["answer"] = answers[text]
        return orig_chat(tok, text)

    def install(model: Any) -> Any:
        state["ctl"] = orig_install(model)
        return state["ctl"]

    def generate(model: Any, tok: Any, ids: list[int], max_tokens: int, config: Any = None) -> list[int]:
        fix_sets(state["ctl"], model, ids, state["answer"])
        return orig_generate(model, tok, ids, max_tokens, config)

    sr.chat, sr.install, sr.generate = chat, install, generate
    try:
        frac = a.k / short.N_EXPERTS[a.model_tag]
        short.cmd_run(argparse.Namespace(model_tag=a.model_tag, tag=f"{a.model_tag}_ceiling", n=a.n, frac=frac, s=32))
    finally:
        sr.chat, sr.install, sr.generate = orig_chat, orig_install, orig_generate


def h33_verdict(bars: dict[str, bool]) -> str:
    """H33: the ceiling is at the bar on at least 2 of the 3 models."""
    return "holds" if sum(bars.values()) >= 2 else "fails"


def report(n: int) -> dict:
    """Write results/ceiling_h33.json: the ceiling beside the both-ends rule at the same budget."""
    out: dict[str, Any] = {}
    for model, k in CELLS.items():
        base_recs = short.read_records(short.run_path(model, n, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        recs = {
            "ceiling": short.read_records(short.run_path(f"{model}_ceiling", n, k, 32)),
            "ends64": short.read_records(short.run_path(sl.tag(model, sl.ENDS_ARM), n, k, 32)),
        }
        scores = {arm: short.item_metrics(r, teacher) for arm, r in recs.items()}
        row = [sl.kept_ci(scores["ceiling"][m], base[m]) for m in sl.KEPT_METRICS]
        out[model] = {
            "k": k,
            "ceiling": dict(zip(sl.KEPT_METRICS, row, strict=True)),
            "ends64": {m: sl.kept_ci(scores["ends64"][m], base[m]) for m in sl.KEPT_METRICS},
            "diffs": [pl.paired_diff(scores["ceiling"][m], scores["ends64"][m]) for m in sl.KEPT_METRICS],
            "at_bar": sl.at_bar(row),
            "loops": {arm: {t: sl.loop_rate(r, t) for t in ("summarize", "rewrite")} for arm, r in recs.items()},
            "loads_per_layer": {arm: float(np.mean([x["loads_per_layer"] for x in r])) for arm, r in recs.items()},
        }
    result = {"verdict": h33_verdict({m: r["at_bar"] for m, r in out.items()}), "models": out}
    (RESULTS / "ceiling_h33.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True, choices=list(CELLS))
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--n", type=int, default=300)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("--n", type=int, required=True)
    s.set_defaults(fn=lambda a: print(json.dumps(report(a.n), indent=1)))
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
