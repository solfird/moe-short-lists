"""C2 disagreement test: masked against unmasked next-token choice, teacher-forced on the unmasked answers.

c2.py run    --model-tag gemma --task gsm8k --frac 0.125 --s 32   # one line per item, resumable
c2.py report                                                      # results/c2_report.md
"""

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

import segroute as sr
import short
from models import MODELS, N_EXPERTS
from records import exclusive, lock_path, provenance, read_records, repair_tail, write_meta
from segroute import Controller
from sweep import C2_FRACS, C2_ITEMS, IDS_TAGS, c2_path

TASKS = tuple(C2_ITEMS)
N_SEG = 10  # segments shown in the by-segment table
MATCHED_SEGS = 4  # segments 0-3: positions both tasks reach (summaries stop at 120 tokens, under 4 x 32)
S = 32


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """First index where two token sequences differ; the shorter length when one is a prefix; None when equal."""
    for t, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return t
    return None if len(a) == len(b) else min(len(a), len(b))


def _step_logprobs(model: nn.Module, prompt_ids: list[int], gen_ids: list[int]) -> Iterator[mx.array]:
    """Float32 next-token log-probs at each answer position: one prefill call, then one call per fed answer token."""
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    for t in range(len(gen_ids)):
        lp = nn.log_softmax(logits[0, -1].astype(mx.float32), axis=-1)
        mx.eval(lp)
        yield lp
        if t + 1 < len(gen_ids):
            logits = model(mx.array([[gen_ids[t]]]), cache=cache)


def teacher_forced(
    model: nn.Module, ctl: Controller, prompt_ids: list[int], gen_ids: list[int], k: int, s: int
) -> dict[str, np.ndarray]:
    """Step-by-step passes on the unmasked answer: masked top-1 differs, KL(unmasked || masked), segment index."""
    ctl.mode = "off"
    ref = list(_step_logprobs(model, prompt_ids, gen_ids))
    ctl.reset()
    ctl.mode, ctl.k, ctl.s = "mask", k, s
    disagree, kl = [], []
    try:
        for t, lm in enumerate(_step_logprobs(model, prompt_ids, gen_ids)):
            lu = ref[t]
            d = (mx.exp(lu) * (lu - lm)).sum()
            top = mx.argmax(lm)
            mx.eval(d, top)
            disagree.append(top.item() != gen_ids[t])
            kl.append(d.item())
            ref[t] = None
    finally:
        ctl.mode = "off"
    return {
        "disagree": np.array(disagree, dtype=bool),
        "kl": np.array(kl, dtype=np.float64),
        "seg": np.arange(len(gen_ids)) // s,
    }


def item_line(task: str, i: int, n_prompt: int, out: dict[str, np.ndarray]) -> dict:
    """One output line of a c2 run, KL rounded to 6 decimals."""
    return {
        "i": i,
        "task": task,
        "n_prompt": n_prompt,
        "n_gen": len(out["disagree"]),
        "disagree": [bool(x) for x in out["disagree"]],
        "kl": [round(float(x), 6) for x in out["kl"]],
        "seg": [int(x) for x in out["seg"]],
    }


def source_path(model_tag: str, task: str) -> Path:
    """Unmasked free run whose answers are teacher-forced: id-storing GSM8K k0, or the n = 300 short k0."""
    if task == "gsm8k":
        return short.RESULTS / f"{IDS_TAGS.get(model_tag, model_tag)}_gsm8k_k0_s32.jsonl"
    return short.run_path(model_tag, C2_ITEMS["summarize"], 0, 32)


def source_items(model_tag: str, task: str) -> list[dict]:
    """Unmasked records of the task with gen_ids and prompt_ids_len; exits naming the file when either is missing."""
    path = source_path(model_tag, task)
    if not path.exists():
        raise SystemExit(f"missing source run {path.name}: run the unmasked {task} free run first")
    recs = [r for r in read_records(path) if r.get("task", "gsm8k") == task]
    for field in ("gen_ids", "prompt_ids_len"):
        if not recs or any(field not in r for r in recs):
            raise SystemExit(f"source run {path.name} has no {field}: rerun it with the current code")
    return recs


def prompt_texts(task: str) -> dict[int, str]:
    """User turn of every C2 item by index, exactly as the free runs sent it."""
    if task == "gsm8k":
        from datasets import load_dataset

        test = load_dataset("openai/gsm8k", "main", split="test").select(range(C2_ITEMS["gsm8k"]))
        return {i: sr.gsm_prompt(ex["question"]) for i, ex in enumerate(test)}
    return {i: p for t, i, p, _ in short.items(C2_ITEMS["summarize"]) if t == task}


def cmd_run(a: argparse.Namespace) -> None:
    """Teacher-force every unmasked answer of a task under the mask at one fraction, resuming the output file."""
    recs = source_items(a.model_tag, a.task)
    path = c2_path(a.model_tag, a.task, a.frac, a.s)
    with exclusive(lock_path(path)):
        _run_locked(a, recs, path)


def _run_locked(a: argparse.Namespace, recs: list[dict], path: Path) -> None:
    """Body of cmd_run, run while this process holds the output file's lock."""
    k = short.budget(N_EXPERTS[a.model_tag], a.frac)
    params = {
        "task": a.task,
        "frac": a.frac,
        "k": k,
        "s": a.s,
        "n": len(recs),
        "source": source_path(a.model_tag, a.task).name,
    }
    write_meta(path, provenance(MODELS[a.model_tag], params))
    done: set[int] = set()
    if path.exists():
        repair_tail(path)
        done = {r["i"] for r in read_records(path)}
    model, tok, _ = load(MODELS[a.model_tag], return_config=True)
    ctl = sr.install(model)
    texts = prompt_texts(a.task)
    t0 = time.time()
    with open(path, "a") as f:
        for r in recs:
            if r["i"] in done:
                continue
            ids = sr.chat(tok, texts[r["i"]])
            if r["prompt_ids_len"] != len(ids):
                raise SystemExit(
                    f"item {r['i']}: prompt has {len(ids)} tokens, the source run had {r['prompt_ids_len']}"
                )
            out = teacher_forced(model, ctl, ids, r["gen_ids"], k, a.s)
            f.write(json.dumps(item_line(a.task, r["i"], len(ids), out)) + "\n")
            f.flush()
            print(
                f"{path.name} {r['i'] + 1}/{len(recs)} d={out['disagree'].mean():.3f} {time.time() - t0:.0f}s",
                flush=True,
            )


def _mean(values: list[float]) -> float | None:
    """Mean of a list, None when it is empty."""
    return float(np.mean(values)) if values else None


def cell_stats(task: str, c2_recs: list[dict] | None, free: list[dict] | None, base: list[dict] | None) -> dict:
    """Teacher-forced d pooled, on segments 0-3 and item-weighted, with denominators; divergence from the free runs."""
    st: dict = {"items": None, "tokens": None, "d": None, "d_matched": None, "tokens_matched": None, "d_item": None}
    st |= {"kl": None, "by_seg": [None] * N_SEG, "n_div": None, "n_free": None, "fd_median": None, "recovery": None}
    st["free_note"] = None
    if c2_recs is not None:
        dis = [bool(x) for r in c2_recs for x in r["disagree"]]
        seg = [int(x) for r in c2_recs for x in r["seg"]]
        matched = [float(x) for x, g in zip(dis, seg, strict=True) if g < MATCHED_SEGS]
        st["items"], st["tokens"], st["tokens_matched"] = len(c2_recs), len(dis), len(matched)
        st["d"] = _mean([float(x) for x in dis])
        st["d_matched"] = _mean(matched)
        st["d_item"] = _mean([float(np.mean(r["disagree"])) for r in c2_recs if r["disagree"]])
        st["kl"] = _mean([float(x) for r in c2_recs for x in r["kl"]])
        st["by_seg"] = [_mean([float(x) for x, g in zip(dis, seg, strict=True) if g == j]) for j in range(N_SEG)]
    if free is not None and base is not None:
        if any("gen_ids" not in r for r in (*free, *base)):
            st["free_note"] = "missing (no gen_ids)"
            return st
        by_i = {r["i"]: r for r in base}
        div = []
        for r in free:
            b = by_i.get(r["i"])
            fd = None if b is None else first_divergence(b["gen_ids"], r["gen_ids"])
            if fd is not None:
                div.append((fd, r, b))
        st["n_div"], st["n_free"] = len(div), len(free)
        st["fd_median"] = float(np.median([fd for fd, _, _ in div])) if div else None
        if task == "gsm8k":
            st["recovery"] = _mean([float(r["ok"]) for _, r, _ in div])
        else:
            from rouge_score import rouge_scorer

            rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            st["recovery"] = _mean([rs.score(b["out"], r["out"])["rougeL"].fmeasure for _, r, b in div])
    return st


def _fmt(x: float | int | None, spec: str = ".3f") -> str:
    """Table cell: a number in the given format, n/a for None."""
    return "n/a" if x is None else format(x, spec)


def _verdict(ratio: float) -> str:
    """C2 bar on d(gsm8k) / d(summarize): holds at 1.5 or less, fails at 2 or more (amended 2026-10-30)."""
    if ratio <= 1.5:
        return "holds"
    return "fails" if ratio >= 2 else "open"


def _ratio(g: float | None, m: float | None) -> str:
    """Table cell for d(gsm8k) / d(summarize): n/a when either is missing or both are 0, inf when only m is 0."""
    if g is None or m is None or (g == 0 and m == 0):
        return "n/a"
    return "inf" if m == 0 else f"{g / m:.2f}"


def _diverged(st: dict) -> str:
    """Table cell: diverged free-run items over free-run items, n/a without free runs."""
    return "n/a" if st["n_div"] is None else f"{st['n_div']}/{st['n_free']}"


def report(cells: list[dict]) -> str:
    """Markdown report over cells {model, task, frac, k, c2, free, base, missing}; a pure function of the records."""
    stats = [(c, cell_stats(c["task"], c["c2"], c["free"], c["base"])) for c in cells]
    out = [
        "# C2 report",
        "",
        f"Teacher-forced on the unmasked answers, S = {S}. Disagreement d: masked top-1 differs from the unmasked "
        "token. Three versions of d: pooled over all answer tokens of a cell (the H13 bar), pooled over segments "
        f"0-{MATCHED_SEGS - 1} only (positions both tasks reach: GSM8K answers reach up to 16 segments, summaries "
        "up to 4), and the mean of per-item d (each item weighs the same). Denominators: items and tokens of the "
        "teacher-forced run, tokens in segments 0-3 in brackets, diverged over free-run items. The EOS step is not "
        "scored in teacher-forced d: d covers the answer tokens only. A masked run that stops early or late shows "
        "in the free-run first divergence instead, where an answer that is a prefix of the other diverges at the "
        "shorter length. First divergence and recovery come from the free-running masked records against k0, over "
        "items whose answer diverged. Recovery: share still correct (GSM8K), mean ROUGE-L against the k0 answer "
        "(summarize). Generated by `c2.py report`.",
        "",
        "## Per-token disagreement and free-run divergence",
        "",
        f"| model | task | frac | K' | items | tokens | d pooled | d seg 0-{MATCHED_SEGS - 1} (tokens) | d item mean "
        "| mean KL | median first divergence | diverged / free items | recovery | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c, st in stats:
        notes = ["missing: " + ", ".join(c["missing"])] if c["missing"] else []
        notes += [st["free_note"]] if st["free_note"] else []
        matched = "n/a" if st["d_matched"] is None else f"{st['d_matched']:.3f} ({st['tokens_matched']})"
        out.append(
            f"| {c['model']} | {c['task']} | {c['frac']:g} | {c['k']} | {_fmt(st['items'], 'd')} "
            f"| {_fmt(st['tokens'], 'd')} | {_fmt(st['d'])} | {matched} | {_fmt(st['d_item'])} | {_fmt(st['kl'])} "
            f"| {_fmt(st['fd_median'], '.1f')} | {_diverged(st)} | {_fmt(st['recovery'])} | {'; '.join(notes)} |"
        )
    out += [
        "",
        f"## Disagreement by segment index (first {N_SEG} segments)",
        "",
        "| model | task | frac | " + " | ".join(f"seg {j}" for j in range(N_SEG)) + " |",
        "|---|---|---|" + "---|" * N_SEG,
    ]
    for c, st in stats:
        out.append(f"| {c['model']} | {c['task']} | {c['frac']:g} | " + " | ".join(map(_fmt, st["by_seg"])) + " |")
    out += [
        "",
        "## C2 bar: d(gsm8k) / d(summarize) at the same fraction",
        "",
        "Verdict on pooled d: holds at 1.5 or less, fails at 2 or more (pre-registered bar C2, H13). "
        f"The segment 0-{MATCHED_SEGS - 1} and item-mean ratios are reported beside it, not as a pass bar.",
        "",
        f"| model | frac | ratio pooled | verdict | ratio seg 0-{MATCHED_SEGS - 1} | ratio item mean |",
        "|---|---|---|---|---|---|",
    ]
    by = {(c["model"], c["frac"], c["task"]): st for c, st in stats}
    for model, frac in dict.fromkeys((c["model"], c["frac"]) for c in cells):
        gs, ms = by.get((model, frac, "gsm8k"), {}), by.get((model, frac, "summarize"), {})
        g, m = gs.get("d"), ms.get("d")
        if g is None or m is None:
            verdict = "missing"
        elif m == 0:
            verdict = "fails" if g > 0 else "no disagreement"
        else:
            verdict = _verdict(g / m)
        others = [_ratio(gs.get(key), ms.get(key)) for key in ("d_matched", "d_item")]
        out.append(f"| {model} | {frac:g} | {_ratio(g, m)} | {verdict} | " + " | ".join(others) + " |")
    return "\n".join(out) + "\n"


def _load(path: Path, task: str, missing: list[str]) -> list[dict] | None:
    """Records of one task from a file, or None with the file name added to missing."""
    if not path.exists():
        missing.append(path.name)
        return None
    return [r for r in read_records(path) if r.get("task", "gsm8k") == task]


def free_path(model_tag: str, task: str, k: int) -> Path:
    """Free-running record file of a model, task and budget (k 0 is unmasked)."""
    if task == "gsm8k":
        return short.RESULTS / f"{IDS_TAGS.get(model_tag, model_tag)}_gsm8k_k{k}_s{S}.jsonl"
    return short.run_path(model_tag, C2_ITEMS["summarize"], k, S)


def cmd_report(a: argparse.Namespace) -> None:
    """Write results/c2_report.md over every model, task and C2 fraction; missing files are named in the table."""
    cells = []
    for model in MODELS:
        for frac in C2_FRACS:
            k = short.budget(N_EXPERTS[model], frac)
            for task in TASKS:
                missing: list[str] = []
                cell = {"model": model, "task": task, "frac": frac, "k": k, "missing": missing}
                cell["c2"] = _load(c2_path(model, task, frac, S), task, missing)
                cell["free"] = _load(free_path(model, task, k), task, missing)
                cell["base"] = _load(free_path(model, task, 0), task, missing)
                cells.append(cell)
    path = short.RESULTS / "c2_report.md"
    path.write_text(report(cells))
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    """Command line: run or report."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    r.add_argument("--task", required=True, choices=TASKS)
    r.add_argument("--frac", type=float, required=True, help="fraction of routed experts kept")
    r.add_argument("--s", type=int, default=S, help="tokens between reselections, > 0")
    sub.add_parser("report")
    a = p.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
