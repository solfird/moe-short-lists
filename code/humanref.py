"""H64: the paper's jurors against human labels of factual support on public data (SummEval, XSum faithfulness).

humanref.py build            # results/humanref/items.jsonl and the blind Claude batches
humanref.py run --juror luna # one API juror (luna, flash), resumable, capped
humanref.py local            # the local juror
humanref.py merge            # Claude batch verdicts into claude_blind.jsonl
humanref.py report           # results/humanref.json
"""

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

import blindbatch
import jury

DIR = Path(__file__).resolve().parents[1] / "results" / "humanref"
OUT = Path(__file__).resolve().parents[1] / "results" / "humanref.json"
PER_CLASS = 150
SEED = 0
CAP_USD = 2.0
XSUM_LABELS = "hf://datasets/google-research-datasets/xsum_factuality@refs%2Fconvert%2Fparquet/xsum_faithfulness/train/0000.parquet"
JURORS = ("claude", "local", "luna", "flash")


def summeval_label(consistency: float) -> bool | None:
    """Supported at a mean expert score of 5, unsupported at 4 or less, None for the band between."""
    if consistency >= 5.0:
        return True
    return False if consistency <= 4.0 else None


def xsum_label(worker_spans: dict[str, list[int]]) -> bool:
    """Supported when at least two annotators marked no unsupported span (type -1 is 'none')."""
    clean = sum(all(t == -1 for t in types) for types in worker_spans.values())
    return clean >= 2


def sample(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    """per_class supported and per_class unsupported rows, drawn without replacement."""
    rng = random.Random(seed)
    out: list[dict] = []
    for label in (True, False):
        pool = [r for r in rows if r["human"] is label]
        out += rng.sample(pool, per_class)
    return out


def agreement(human: list[bool], verdict: list[bool]) -> dict[str, float]:
    """caught: unsupported called unsupported; passed: supported called supported; balanced: their mean."""
    h, v = np.array(human), np.array(verdict)
    caught = float((~v[~h]).mean())
    passed = float(v[h].mean())
    return {"caught": caught, "passed": passed, "balanced": (caught + passed) / 2}


def balanced_ci(human: list[bool], verdict: list[bool], n: int = 2000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    h, v = np.array(human), np.array(verdict)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(h), len(h))
        hh, vv = h[idx], v[idx]
        if hh.all() or not hh.any():
            continue
        vals.append(((~vv[~hh]).mean() + vv[hh].mean()) / 2)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def _summeval() -> list[dict]:
    from datasets import load_dataset

    rows = []
    for a, r in enumerate(load_dataset("mteb/summeval", split="test")):
        for s, (text, c) in enumerate(zip(r["machine_summaries"], r["consistency"], strict=True)):
            rows.append(
                {
                    "set": "summeval",
                    "i": a,
                    "src": s,
                    "article": r["text"],
                    "answer": text,
                    "score": c,
                    "human": summeval_label(c),
                }
            )
    return rows


def _xsum() -> list[dict]:
    from datasets import load_dataset

    spans: dict[tuple, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load_dataset("parquet", data_files=XSUM_LABELS, split="train"):
        spans[(r["bbcid"], r["system"], r["summary"])][r["worker_id"]].append(r["hallucination_type"])
    docs = {int(d["id"]): d["document"] for d in load_dataset("EdinburghNLP/xsum", split="test")}
    rows = []
    for (bbcid, system, summary), workers in sorted(spans.items()):
        if len(workers) != 3:
            continue
        rows.append(
            {
                "set": "xsum",
                "i": 100_000 + bbcid,
                "src": system,
                "article": docs[bbcid],
                "answer": summary,
                "human": xsum_label(workers),
            }
        )
    return rows


def cmd_build() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    picked = sample(_summeval(), PER_CLASS, SEED) + sample(_xsum(), PER_CLASS, SEED)
    random.Random(SEED).shuffle(picked)
    items = [{"id": n, **r} for n, r in enumerate(picked)]
    (DIR / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    bdir, odir = DIR / "claude_blind_batches", DIR / "claude_blind_out"
    bdir.mkdir(exist_ok=True)
    odir.mkdir(exist_ok=True)
    batches = blindbatch.distinct_batches(items, blindbatch.SIZE)
    for n, b in enumerate(batches):
        (bdir / f"batch_{n:03d}.json").write_text(
            json.dumps([{k: x[k] for k in ("id", "article", "answer")} for x in b], indent=0)
        )
    print(len(items), "items", len(batches), "batches", collections.Counter((r["set"], r["human"]) for r in items))


def _point_jury() -> None:
    jury.JURY, jury.ITEMS, jury.CAP_USD = DIR, DIR / "items.jsonl", CAP_USD


def cmd_run(a: argparse.Namespace) -> None:
    _point_jury()
    jury.cmd_run(argparse.Namespace(juror=a.juror, limit=10_000, workers=a.workers))


def cmd_local() -> None:
    from mlx_lm import generate, load

    path = DIR / "local.jsonl"
    done = {x["id"] for x in jury._read(path)}
    model, tok = load("mlx-community/Ministral-3-8B-Instruct-2512-4bit", tokenizer_config={"fix_mistral_regex": True})
    with path.open("a") as f:
        for it in jury._read(DIR / "items.jsonl"):
            if it["id"] in done:
                continue
            msgs = [{"role": "user", "content": f"{jury.SYSTEM}\n\n{jury.juror_prompt(it)}"}]
            text = generate(model, tok, prompt=tok.apply_chat_template(msgs, add_generation_prompt=True), max_tokens=40)
            f.write(json.dumps({"id": it["id"], "raw": text, "verdict": jury.parse_verdict(text)}) + "\n")
            f.flush()


def cmd_merge() -> None:
    out, problems = blindbatch.merge(DIR / "claude_blind_batches", DIR / "claude_blind_out")
    print("verdicts", len(out), "problems", problems)
    if not problems:
        (DIR / "claude_blind.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))


def gate(two_member: dict[str, float]) -> str:
    """H64: holds at 75% balanced agreement on both sets, fails under 65% on either."""
    if all(v >= 0.75 for v in two_member.values()):
        return "holds"
    return "fails" if any(v < 0.65 for v in two_member.values()) else "neither"


def report() -> dict[str, Any]:
    items = jury._read(DIR / "items.jsonl")
    files = {"claude": "claude_blind.jsonl", "local": "local.jsonl", "luna": "luna.jsonl", "flash": "flash.jsonl"}
    votes: dict[str, dict[int, bool]] = {}
    unparsed = {}
    for j, name in files.items():
        rows = {r["id"]: r["verdict"] for r in jury._read(DIR / name)}
        unparsed[j] = sum(rows.get(it["id"]) is None for it in items) / len(items)
        votes[j] = {i: v["supported"] for i, v in rows.items() if v is not None}
    panels = {
        "two_member": lambda i: votes["claude"][i] and votes["local"][i],
        "three_vendor": lambda i: votes["claude"][i] + votes["luna"][i] + votes["flash"][i] >= 2,
        **{j: (lambda i, j=j: votes[j][i]) for j in JURORS},
    }
    out: dict[str, Any] = {"n": len(items), "unparsed": unparsed, "sets": {}}
    for name in ("summeval", "xsum"):
        rows = [it for it in items if it["set"] == name and all(it["id"] in votes[j] for j in JURORS)]
        human = [it["human"] for it in rows]
        cell: dict[str, Any] = {"n": len(rows)}
        for p, fn in panels.items():
            verdict = [bool(fn(it["id"])) for it in rows]
            cell[p] = {**agreement(human, verdict), "ci": balanced_ci(human, verdict)}
        out["sets"][name] = cell
    out["valid"] = all(u <= 0.02 for u in unparsed.values())
    out["verdict"] = gate({s: c["two_member"]["balanced"] for s, c in out["sets"].items()})
    OUT.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--juror", required=True, choices=sorted(jury.JURORS))
    r.add_argument("--workers", type=int, default=8)
    for name in ("local", "merge", "report"):
        sub.add_parser(name)
    a = p.parse_args()
    if a.cmd == "build":
        cmd_build()
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "local":
        cmd_local()
    elif a.cmd == "merge":
        cmd_merge()
    else:
        print(json.dumps(report(), indent=1))


if __name__ == "__main__":
    main()
