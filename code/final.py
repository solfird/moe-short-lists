"""Final tests (pass rules written 2026-11-03 07:35): H57 held-out items 600 to 899, H58 XSum.

final.py summarize --dataset cnn --model-tag gemma --k 48 --start 600 --n 300   # results/sum_cnn_gemma_d600_k48.jsonl
final.py summarize --dataset xsum --model-tag gemma --k 0 --n 200                # results/sum_xsum_gemma_d0_k0.jsonl
final.py build --round heldout|xsum      # results/panel_<round>/items.jsonl and Claude blind batches
final.py local --round heldout|xsum      # local juror, resumable
final.py merge --round heldout|xsum      # Claude verdicts from the blind batch outputs
final.py report --round heldout|xsum     # results/final_<round>.json
"""

import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ("code", "analysis")]

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import baseline
import blindbatch
import copyaudit
import failures
import jury
import selrule as sl
import short

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIRST = {"gemma": 24, "nemotron": 16, "qwen36": 24}
RECOVERY = {"gemma": 48, "nemotron": 48, "qwen36": 128}
ROUNDS = {"heldout": {"dataset": "cnn", "start": 600, "n": 300}, "xsum": {"dataset": "xsum", "start": 0, "n": 200}}
LOCAL = "mlx-community/Ministral-3-8B-Instruct-2512-4bit"
INSTRUCTION = "Summarize this news article in 2 or 3 sentences.\n\n"


def summary_prompt(text: str) -> str:
    """The suite's summarize prompt: instruction, blank line, text cut to 3,000 characters."""
    return INSTRUCTION + text[:3000]


def cells() -> list[tuple[str, int]]:
    return [(m, k) for m in FIRST for k in (0, FIRST[m], RECOVERY[m])]


def sum_path(dataset: str, model_tag: str, start: int, k: int) -> Path:
    return RESULTS / f"sum_{dataset}_{model_tag}_d{start}_k{k}.jsonl"


def cell_flags(items: list[dict], claude: dict[int, dict], local: dict[int, dict]) -> dict[str, dict[int, bool]]:
    """Accepted by both jurors on all three questions, per cell and item; items missing a verdict are left out."""
    out: dict[str, dict[int, bool]] = {}
    for it in items:
        a, b = claude.get(it["id"]), local.get(it["id"])
        if a and b:
            out.setdefault(it["cell"], {})[it["i"]] = all(a[q] and b[q] for q in jury.QUESTIONS)
    return out


def summary_items(dataset: str, n: int, start: int) -> list[tuple[int, str, str]]:
    """(index, prompt, reference) for the summaries of one round."""
    if dataset == "cnn":
        return [(i, p, r) for t, i, p, r in short.items(n, start) if t == "summarize"]
    from datasets import load_dataset

    ds = load_dataset("EdinburghNLP/xsum", split="test").select(range(start, start + n))
    return [(start + j, summary_prompt(r["document"]), r["summary"]) for j, r in enumerate(ds)]


def cmd_summarize(a: argparse.Namespace) -> None:
    from mlx_lm import load

    import curriculum as cu
    import segroute as sr
    from models import MODELS

    path = sum_path(a.dataset, a.model_tag, a.start, a.k)
    model, tok, config = load(MODELS[a.model_tag], return_config=True)
    ctl = cu.install_with(model, sl.TailController)
    ctl.mode, ctl.k, ctl.s = ("off", None, None) if a.k == 0 else ("mask", a.k, 32)
    with path.open("w") as f:
        for i, prompt, ref in summary_items(a.dataset, a.n, a.start):
            ctl.reset()
            gen = sr.generate(model, tok, sr.chat(tok, prompt), short.MAX["summarize"], config)
            f.write(
                json.dumps({"task": "summarize", "i": i, "ref": ref, "out": tok.decode(gen), "gen_ids": gen}) + "\n"
            )
    print(path.name, flush=True)


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open()] if path.exists() else []


def answers(rnd: str, m: str, k: int) -> list[dict]:
    """Stored summary records of one cell; held-out first budgets and full models come from the suite runs."""
    r = ROUNDS[rnd]
    if rnd == "heldout" and k in (0, FIRST[m]):
        tag = sl.tag(m, sl.ENDS_ARM if k else sl.WHOLE_ARM, r["start"])
        return [x for x in short.read_records(short.run_path(tag, r["n"], k, 32)) if x["task"] == "summarize"]
    return _read(sum_path(r["dataset"], m, r["start"], k))


def panel_dir(rnd: str) -> Path:
    return RESULTS / f"panel_{rnd}"


def cmd_build(a: argparse.Namespace) -> None:
    r = ROUNDS[a.round]
    articles = {i: p.split("\n\n", 1)[1] for i, p, _ in summary_items(r["dataset"], r["n"], r["start"])}
    rows = [
        {"cell": f"{m}_k{k}", "i": x["i"], "article": articles[x["i"]], "answer": x["out"]}
        for m, k in cells()
        for x in answers(a.round, m, k)
    ]
    d = panel_dir(a.round)
    d.mkdir(exist_ok=True)
    rows = jury.shuffled(rows)
    (d / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    bdir, odir = d / "claude_blind_batches", d / "claude_blind_out"
    bdir.mkdir(exist_ok=True)
    odir.mkdir(exist_ok=True)
    batches = blindbatch.distinct_batches(rows, blindbatch.SIZE)
    for n, b in enumerate(batches):
        part = [{k: x[k] for k in ("id", "article", "answer")} for x in b]
        (bdir / f"batch_{n:03d}.json").write_text(json.dumps(part, indent=0))
    print(len(rows), "answers", len(batches), "batches")


def cmd_local(a: argparse.Namespace) -> None:
    from mlx_lm import generate, load

    d = panel_dir(a.round)
    path = d / "local.jsonl"
    done = {x["id"] for x in _read(path)}
    model, tok = load(LOCAL, tokenizer_config={"fix_mistral_regex": True})
    with path.open("a") as f:
        for it in _read(d / "items.jsonl"):
            if it["id"] in done:
                continue
            msgs = [{"role": "user", "content": f"{jury.SYSTEM}\n\n{jury.juror_prompt(it)}"}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
            text = generate(model, tok, prompt=prompt, max_tokens=40)
            f.write(json.dumps({"id": it["id"], "raw": text, "verdict": jury.parse_verdict(text)}) + "\n")
            f.flush()


def cmd_merge(a: argparse.Namespace) -> None:
    d = panel_dir(a.round)
    out, problems = blindbatch.merge(d / "claude_blind_batches", d / "claude_blind_out")
    print("verdicts", len(out), "problems", problems)
    if not problems:
        (d / "claude_blind.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))


def _rouge_l(recs: list[dict]) -> dict[int, float]:
    from rouge_score import rouge_scorer

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return {x["i"]: short.reference_scores({"task": "summarize", **x}, rs)["sum_rougeL_ref"] for x in recs}


def _ci(c: dict[str, float]) -> dict[str, float]:
    return {k: round(100 * v, 1) for k, v in c.items()}


def cmd_report(a: argparse.Namespace) -> dict[str, Any]:
    r = ROUNDS[a.round]
    d = panel_dir(a.round)
    items = _read(d / "items.jsonl")
    claude = {x["id"]: x["verdict"] for x in _read(d / "claude_blind.jsonl") if x.get("verdict")}
    local = {x["id"]: x["verdict"] for x in _read(d / "local.jsonl") if x.get("verdict")}
    flags = cell_flags(items, claude, local)
    refs = {i: (p, ref) for i, p, ref in summary_items(r["dataset"], r["n"], r["start"])}
    prompts = {i: p for i, (p, _) in refs.items()}
    out: dict[str, Any] = {"round": a.round, "n_judged": sum(map(len, flags.values())), "models": {}}
    for m in FIRST:
        base_recs = answers(a.round, m, 0)
        base_rouge, base_acc = _rouge_l(base_recs), {i: float(v) for i, v in flags[f"{m}_k0"].items()}
        arts = {i: p.split("\n\n", 1)[1] for i, p in prompts.items()}
        leads = {
            name: sl.kept_ci(
                _rouge_l([{"i": i, "ref": refs[i][1], "out": baseline.lead_sentences(arts[i], s)} for i in base_rouge]),
                base_rouge,
            )
            for name, s in (("lead3", 3), ("lead1", 1))
        }
        row: dict[str, Any] = {"full_accepted": float(np.mean(list(base_acc.values()))), "leads": leads, "cells": {}}
        for k in (FIRST[m], RECOVERY[m]):
            recs = answers(a.round, m, k)
            acc = {i: float(v) for i, v in flags[f"{m}_k{k}"].items()}
            row["cells"][str(k)] = {
                "rougeL_kept": sl.kept_ci(_rouge_l(recs), base_rouge),
                "accepted_kept": sl.kept_ci(acc, base_acc),
                "at_bar": sl.at_bar([sl.kept_ci(acc, base_acc)]),
                "loops": float(np.mean([failures.is_loop(x["gen_ids"]) for x in recs])),
                **copyaudit._row(recs, prompts),
            }
        out["models"][m] = row
    if a.round == "heldout":
        out["suite"] = suite_heldout()
    (RESULTS / f"final_{a.round}.json").write_text(json.dumps(out, indent=1))
    for m, row in out["models"].items():
        print(m, "leads", {k: _ci(v) for k, v in row["leads"].items()})
        for k, c in row["cells"].items():
            print(
                "  k", k, "rougeL", _ci(c["rougeL_kept"]), "jury", _ci(c["accepted_kept"]), "over_half", c["over_half"]
            )
    return out


def suite_heldout() -> dict[str, Any]:
    """QA and classify kept at the first budgets on items 600 to 899 (H57 part a)."""
    out: dict[str, Any] = {}
    for m, k in FIRST.items():
        base_recs = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, 600), 300, 0, 32))
        teacher = short.teacher_outputs(base_recs)
        base = short.item_metrics(base_recs, teacher)
        recs = short.read_records(short.run_path(sl.tag(m, sl.ENDS_ARM, 600), 300, k, 32))
        scores = short.item_metrics(recs, teacher)
        row = {t: sl.kept_ci(scores[t], base[t]) for t in sl.KEPT_METRICS}
        out[m] = {
            **row,
            "qa_cls_at_bar": sl.at_bar([row["qa_f1"], row["cls_acc"]]),
            "loads_per_layer": float(np.mean([x["loads_per_layer"] for x in recs])),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("summarize", "build", "local", "merge", "report"))
    p.add_argument("--dataset", choices=("cnn", "xsum"), default="cnn")
    p.add_argument("--model-tag", choices=sorted(FIRST))
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--round", choices=sorted(ROUNDS))
    a = p.parse_args()
    {"summarize": cmd_summarize, "build": cmd_build, "local": cmd_local, "merge": cmd_merge, "report": cmd_report}[
        a.cmd
    ](a)


if __name__ == "__main__":
    main()
