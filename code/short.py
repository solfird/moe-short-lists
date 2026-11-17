"""Short-task suite under segment routing.

short.py run   --model-tag gemma --n 300 --frac 0.25 --s 32 [--tag T]   # frac 0 = unmasked
short.py score --tag gemma [--n 300]    # with --n: bootstrap CIs per run; without: the n = 100 runs of 2026-10-26
"""

import argparse
import collections
import json
import re
import string
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from mlx_lm import load

import segroute as sr
from models import MODELS, N_EXPERTS
from records import exclusive, lock_path, provenance, read_records, repair_tail, write_meta

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
MAX = {"summarize": 120, "qa": 32, "classify": 8, "rewrite": 160}


def article_slices(n: int, start: int = 0) -> dict[str, range]:
    """CNN/DailyMail test indices used by the two article tasks: disjoint by construction while start + n <= 1000."""
    return {"summarize": range(start, start + n), "rewrite": range(1000 + start, 1000 + start + n)}


def items(n: int = 300, start: int = 0) -> list[tuple[str, int, str, Any]]:
    """The n items of each task from position start, as (task, index, prompt, reference)."""
    from datasets import load_dataset

    out: list[tuple[str, int, str, Any]] = []
    cnn = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    slices = article_slices(n, start)
    for i, j in enumerate(slices["summarize"], start):
        out.append(
            (
                "summarize",
                i,
                "Summarize this news article in 2 or 3 sentences.\n\n" + cnn[j]["article"][:3000],
                cnn[j]["highlights"],
            )
        )
    sq = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=0).select(range(start, start + n))
    for i, r in enumerate(sq, start):
        q = f"Answer the question with a short span copied from the text.\n\nText: {r['context']}\n\nQuestion: {r['question']}"  # noqa: E501 prompt or log text kept byte-identical for reproducibility
        out.append(("qa", i, q, r["answers"]["text"]))
    ag = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(start, start + n))
    for i, r in enumerate(ag, start):
        q = (
            "Classify this news item into one of: World, Sports, Business, Sci/Tech. Reply with the label only.\n\n"
            + r["text"]
        )
        out.append(("classify", i, q, LABELS[r["label"]]))
    for i, j in enumerate(slices["rewrite"], start):  # different articles from the summarize set
        para = cnn[j]["article"][:700].rsplit(".", 1)[0] + "."
        out.append(
            ("rewrite", i, "Rewrite this paragraph in plain, simple English. Keep all the facts.\n\n" + para, None)
        )
    return out


def budget(n_experts: int, frac: float) -> int:
    """Experts kept per layer for a fraction of the routed experts: nearest integer, clamped to [1, n_experts]."""
    return min(n_experts, max(1, round(n_experts * frac)))


def run_path(tag: str, n: int, k: int, s: int) -> Path:
    """Raw output file of one run; the unmasked run (k 0) does not depend on S, so it always uses s32."""
    return RESULTS / f"{tag}_short_n{n}_k{k}_s{32 if k == 0 else s}.jsonl"


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def item_record(
    task: str, i: int, ref: Any, prompt_ids: list[int], gen_ids: list[int], decode: Callable[[list[int]], str]
) -> dict:
    """One output record, with the generated token ids and the prompt length stored beside the text."""
    return {
        "task": task,
        "i": i,
        "out": decode(gen_ids),
        "ref": ref,
        "n_prompt": len(prompt_ids),
        "n_gen": len(gen_ids),
        "gen_ids": gen_ids,
        "prompt_ids_len": len(prompt_ids),
    }


def run_suite(
    path: Path,
    todo: list[tuple[str, int, str, Any]],
    step: Callable[[str, int, str, Any], dict],
    meta: dict | None = None,
) -> None:
    """Append one record per item not yet in path, as its only writer; a torn last line is cut first."""
    with exclusive(lock_path(path)):
        if meta is not None:
            write_meta(path, meta)
        done: set[tuple[str, int]] = set()
        if path.exists():
            repair_tail(path)
            done = {(r["task"], r["i"]) for r in read_records(path)}
        with open(path, "a") as f:
            for task, i, prompt, ref in todo:
                if (task, i) in done:
                    continue
                f.write(json.dumps(step(task, i, prompt, ref)) + "\n")
                f.flush()


def cmd_run(a: argparse.Namespace) -> None:
    """Run the suite for one model and budget, resuming the output file if it exists."""
    model, tok, config = load(MODELS[a.model_tag], return_config=True)
    ctl = sr.install(model)
    k = 0 if a.frac == 0 else budget(ctl.n_experts, a.frac)
    ctl.mode, ctl.k, ctl.s = ("off", None, None) if k == 0 else ("mask", k, a.s or None)
    RESULTS.mkdir(exist_ok=True)
    path = run_path(a.tag or a.model_tag, a.n, k, a.s)

    def step(task: str, i: int, prompt: str, ref: Any) -> dict:
        ctl.reset()
        ids = sr.chat(tok, prompt)
        gen = sr.generate(model, tok, ids, MAX[task], config)
        rec = item_record(task, i, ref, ids, gen, tok.decode)
        if ctl.mode == "mask":
            rec |= {
                "loads_per_layer": float(np.mean(ctl.loads)),
                "touched_per_layer": float(np.mean([len(t) for t in ctl.touched])),
                "reselects": int(np.mean(ctl.reselects)),
            }
        return rec

    t0 = time.time()
    meta = provenance(
        MODELS[a.model_tag],
        {"tag": a.tag, "frac": a.frac, "k": k, "s": a.s, "n": a.n},
    )
    run_suite(path, items(a.n), step, meta)
    print(f"{path.name}: {time.time() - t0:.0f}s, k={k} of {N_EXPERTS[a.model_tag]}")


def norm(s: str) -> str:
    """SQuAD answer normalization: lowercase, no punctuation or articles, single spaces."""
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between a prediction and a gold answer."""
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def label_of(out: str) -> str | None:
    """The single AG News label named in an output, or None when zero or several match."""
    o = out.lower()
    hits = [lab for lab in LABELS if lab.lower() in o or (lab == "Sci/Tech" and ("sci" in o or "tech" in o))]
    return hits[0] if len(hits) == 1 else None


METRICS = ["sum_rougeL_ref", "sum_rougeL_teacher", "qa_em", "qa_f1", "cls_acc", "rew_rougeL_teacher"]


def teacher_outputs(recs: list[dict]) -> dict[tuple[str, int], str]:
    """Unmasked outputs by (task, index): the reference for the teacher-agreement scores."""
    return {(r["task"], r["i"]): r["out"] for r in recs}


def reference_scores(r: dict, rs: Any = None) -> dict[str, float]:
    """The scores of one record against its dataset reference.

    The records of this repository carry them in `ref_scores` and no reference text. A record written by a new run
    carries `ref` and the scores are computed from it.
    """
    if "ref_scores" in r:
        return r["ref_scores"]
    task = r["task"]
    if task == "summarize":
        if rs is None:
            from rouge_score import rouge_scorer

            rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return {"sum_rougeL_ref": rs.score(r["ref"], r["out"])["rougeL"].fmeasure}
    if task == "qa":
        return {
            "qa_em": float(max(norm(r["out"]) == norm(g) for g in r["ref"])),
            "qa_f1": max(f1(r["out"], g) for g in r["ref"]),
        }
    if task == "classify":
        return {"cls_acc": float(label_of(r["out"]) == r["ref"])}
    return {}


def item_metrics(recs: list[dict], teacher: dict[tuple[str, int], str]) -> dict[str, dict[tuple[str, int], float]]:
    """Per-item score of every metric the records support, keyed by (task, index)."""
    from rouge_score import rouge_scorer

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    out: dict[str, dict[tuple[str, int], float]] = collections.defaultdict(dict)
    for r in recs:
        task = r["task"]
        key = (task, r["i"])
        for name, value in reference_scores(r, rs).items():
            out[name][key] = value
        if task == "summarize":
            out["sum_rougeL_teacher"][key] = rs.score(teacher[(task, r["i"])], r["out"])["rougeL"].fmeasure
        elif task == "rewrite":
            out["rew_rougeL_teacher"][key] = rs.score(teacher[(task, r["i"])], r["out"])["rougeL"].fmeasure
    return out


def score_run(name: str, recs: list[dict], teacher: dict[tuple[str, int], str]) -> dict:
    """Point estimates of one run: the row layout of results/{tag}_short_scores.json."""
    row: dict[str, Any] = {"run": name}
    row |= {k: float(np.mean(list(v.values()))) for k, v in item_metrics(recs, teacher).items()}
    loads = [r["loads_per_layer"] for r in recs if "loads_per_layer" in r]
    row["loads_per_layer"] = float(np.mean(loads)) if loads else None
    row["n"] = len(recs)
    return row


def paired_kept_ci(masked: dict[tuple[str, int], float], base: dict[tuple[str, int], float]) -> dict[str, float]:
    """Masked over k0 mean on matched items, with a paired 95% bootstrap CI; resamples with k0 mean 0 are dropped."""
    keys = sorted(masked.keys() & base.keys())
    if not keys:
        raise ValueError("paired_kept_ci: no matched items between the masked and the k0 run")
    m = np.array([masked[k] for k in keys])
    b = np.array([base[k] for k in keys])
    if b.mean() == 0:
        raise ValueError("paired_kept_ci: the k0 mean is 0, so the kept ratio is undefined")
    idx = np.random.default_rng(0).integers(0, len(keys), size=(2000, len(keys)))
    m_boot, b_boot = m[idx].mean(axis=1), b[idx].mean(axis=1)
    ok = b_boot > 0
    lo, hi = np.percentile(m_boot[ok] / b_boot[ok], [2.5, 97.5])
    return {"mean": float(m.mean() / b.mean()), "lo": float(lo), "hi": float(hi)}


def ci_scores(runs: dict[str, list[dict]], teacher: dict[tuple[str, int], str]) -> dict[str, dict]:
    """{mean, lo, hi} per metric for each run, plus a paired 'kept' interval against the k0 run for masked runs."""
    out: dict[str, dict] = {}
    items_by_run = {name: item_metrics(recs, teacher) for name, recs in runs.items()}
    for name, recs in runs.items():
        entry: dict[str, Any] = {}
        for metric, vals in items_by_run[name].items():
            lo, hi = bootstrap_ci(list(vals.values()))
            entry[metric] = {"mean": float(np.mean(list(vals.values()))), "lo": lo, "hi": hi}
        loads = [r["loads_per_layer"] for r in recs if "loads_per_layer" in r]
        entry["loads_per_layer"] = float(np.mean(loads)) if loads else None
        entry["n"] = len(recs)
        out[name] = entry
    base_name = next((n for n in runs if n.startswith("k0")), None)
    if base_name:
        base = items_by_run[base_name]
        for name, entry in out.items():
            if name != base_name:
                entry["kept"] = {
                    m: paired_kept_ci(items_by_run[name][m], base[m])
                    for m in METRICS
                    if m in items_by_run[name] and m in base and np.mean(list(base[m].values())) > 0
                }
    return out


def print_rows(rows: list[dict]) -> None:
    """Fixed-width table of score rows."""
    keys = [*METRICS, "loads_per_layer", "n"]
    print("run".ljust(36) + "".join(k[:12].rjust(13) for k in keys))
    for r in rows:
        print(
            r["run"].ljust(36)
            + "".join((f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r.get(k))).rjust(13) for k in keys)
        )


def cmd_score(a: argparse.Namespace) -> None:
    """Score every run of a tag: point estimates (n = 100 files) or bootstrap CIs (--n)."""
    if a.n:
        runs = {
            p.stem.split(f"_short_n{a.n}_")[1]: read_records(p)
            for p in sorted(RESULTS.glob(f"{a.tag}_short_n{a.n}_k*_s*.jsonl"))
        }
        if "k0_s32" not in runs:
            raise SystemExit(f"missing baseline run {run_path(a.tag, a.n, 0, 32).name}: run it with --frac 0 first")
        scores = ci_scores(runs, teacher_outputs(runs["k0_s32"]))
        for name, e in scores.items():
            print(name.ljust(12), {m: round(e[m]["mean"], 3) for m in METRICS if m in e}, "kept:", e.get("kept"))
        with open(RESULTS / f"{a.tag}_short_n{a.n}_scores.json", "w") as f:
            json.dump(scores, f, indent=1)
        return
    teacher = teacher_outputs(read_records(RESULTS / f"{a.tag}_short_k0_s32.jsonl"))
    rows = [score_run(p.stem, read_records(p), teacher) for p in sorted(RESULTS.glob(f"{a.tag}*_short_k*_s*.jsonl"))]
    print_rows(rows)
    with open(RESULTS / f"{a.tag}_short_scores.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)


def parser() -> argparse.ArgumentParser:
    """Command line: run or score."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    r.add_argument("--tag", help="output tag, default the model tag")
    r.add_argument("--n", type=int, default=300)
    r.add_argument("--frac", type=float, required=True, help="fraction of routed experts kept; 0 = unmasked")
    r.add_argument("--s", type=int, default=32)
    s = sub.add_parser("score")
    s.add_argument("--tag", required=True)
    s.add_argument("--n", type=int, help="score the n-item runs with bootstrap CIs")
    return p


def main() -> None:
    """Command line: run or score."""
    a = parser().parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
